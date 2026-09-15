#!/usr/bin/env python3
"""Evaluate every SFT checkpoint on a fixed holdout subset (teacher-forced loss).

Each checkpoint dir must contain adapter_config.json / adapter_model.safetensors /
region_adapter.pt (as written by train_region_sft._save_adapter). The root of
``--sft-dir`` itself is evaluated too (the final weights).

Usage:
    python scripts/eval_sft_checkpoints.py \
        --config configs/qwen35_4b_outcome_multibox.yaml \
        --sft-dir outputs/train/region_sft_multibox_4b --n 128 --batch-size 4
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.prior_dataset import build_train_ref_pool
from data.scan import load_prior_split, split_holdout_by_class
from models.anomaly_prior import AnomalyPrior
from models.qwen35 import setup_model_and_processor
from models.region_injection import attach_region_adapter, ensure_region_token
from outcome.inputs_multibox import OutcomeMultiboxCollator, OutcomeMultiboxDataset
from train_region_sft import batch_loss, pack_sft_batch
from outcome.thinking import thinking_enabled
from utils.config import load_yaml_config


def find_checkpoints(sft_dir: Path) -> list:
    ckpts = []
    for p in sft_dir.iterdir():
        m = re.fullmatch(r'checkpoint-(\d+)', p.name)
        if m and (p / 'adapter_config.json').exists():
            ckpts.append((int(m.group(1)), p))
    ckpts.sort()
    if (sft_dir / 'adapter_config.json').exists():
        final_step = ckpts[-1][0] if ckpts else 0
        ckpts.append((final_step, sft_dir))  # final weights at the root
    return ckpts


def load_ckpt_model(cfg, ckpt_dir: Path, device):
    from peft import PeftModel
    model, processor = setup_model_and_processor(cfg, for_inference=True, freeze_vision=True)
    ensure_region_token(processor, model)
    model = PeftModel.from_pretrained(model, str(ckpt_dir), is_trainable=False).merge_and_unload()
    prior = AnomalyPrior.from_qwen(model, cfg)
    attach_region_adapter(model, prior, cfg, str(ckpt_dir))
    model.to(device)
    model.eval()
    return model, processor, prior


@torch.no_grad()
def eval_ckpt(model, collator, dataset, device, tokenizer, indices, batch_size):
    total, count = 0.0, 0
    for start in range(0, len(indices), batch_size):
        samples = [dataset[i] for i in indices[start:start + batch_size]]
        packed, _ = pack_sft_batch(collator, device, samples, tokenizer, True,
                                  thinking=thinking_enabled(cfg))
        n = len(samples)
        total += float(batch_loss(model, packed)) * n
        count += n
    return total / max(1, count)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--sft-dir', required=True)
    ap.add_argument('--n', type=int, default=128)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--seed', type=int, default=123, help='fixed subset seed (same for all ckpts)')
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)

    train, _ = load_prior_split(cfg)
    train, dev = split_holdout_by_class(train, float(cfg['data']['holdout_ratio']),
                                        seed=int(cfg['training']['seed']))
    pool = build_train_ref_pool(train)

    ckpts = find_checkpoints(Path(args.sft_dir))
    print(f'[eval] checkpoints: {[c[0] for c in ckpts]}', flush=True)

    results = []
    dataset = None
    for step, ckpt_dir in ckpts:
        model, processor, prior = load_ckpt_model(cfg, ckpt_dir, device)
        if dataset is None:
            dataset = OutcomeMultiboxDataset(dev, cfg, processor, 'eval', pool)
            rng = random.Random(int(args.seed))
            order = list(range(len(dataset)))
            rng.shuffle(order)
            indices = order[:max(1, min(int(args.n), len(order)))]
            print(f'[eval] fixed dev subset n={len(indices)} (seed={args.seed})', flush=True)
        tokenizer = getattr(processor, 'tokenizer', processor)
        collator = OutcomeMultiboxCollator(processor, prior, cfg)
        loss = eval_ckpt(model, collator, dataset, device, tokenizer, indices, args.batch_size)
        tag = 'final' if ckpt_dir == Path(args.sft_dir) else ''
        print(f'[eval] step={step} {tag} dev_loss={loss:.4f}', flush=True)
        results.append((step, loss, tag))
        del model, prior, collator
        torch.cuda.empty_cache()

    best = min(results, key=lambda r: r[1])
    print('\n[eval] === summary (fixed dev subset) ===', flush=True)
    for step, loss, tag in results:
        mark = ' <-- best' if (step, loss, tag) == best else ''
        print(f'  step={step:>5} {tag:>5} dev_loss={loss:.4f}{mark}', flush=True)


if __name__ == '__main__':
    main()
