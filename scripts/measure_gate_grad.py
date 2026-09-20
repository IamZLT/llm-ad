#!/usr/bin/env python3
"""Measure the alpha-gate gradient scale to pick a fair --gate-lr.

Loads the frozen-baseline + trainable HPriorAdapter, runs one forward+backward on a
batch of real train samples, and reports alpha.grad (and the delta magnitude) so the
mechanism-probe LR can move alpha to a meaningful value within one epoch.
"""
import os, sys
os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from utils.config import load_yaml_config
from utils.common import set_seed
from data.prior_dataset import build_train_ref_pool
from data.scan import load_prior_split, split_holdout_by_class
from outcome.inputs_multibox import OutcomeMultiboxCollator, OutcomeMultiboxDataset
from train_region_sft import load_sft_model, pack_sft_batch, batch_loss
from models.qwen35 import unwrap_model


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/qwen35_2b_annos_probe_v2_fsm_h_prior_sft.yaml'
    n_batch = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    cfg = load_yaml_config(cfg_path)
    set_seed(int(cfg['training']['seed']))
    device = torch.device('cuda', 0)

    model, processor, prior = load_sft_model(
        cfg, device, init_sft=cfg['outcome']['sft_adapter'],
        freeze_region=True, freeze_lora=True)
    tokenizer = getattr(processor, 'tokenizer', processor)
    model.train()

    train, _ = load_prior_split(cfg)
    train, dev = split_holdout_by_class(train, float(cfg['data']['holdout_ratio']),
                                        seed=int(cfg['training']['seed']))
    pool = build_train_ref_pool(train)
    dataset = OutcomeMultiboxDataset(train, cfg, processor, 'train', pool)
    collator = OutcomeMultiboxCollator(processor, prior, cfg)

    # sample a mix of anomaly + normal (matches training distribution); early break
    samples = []
    n_a = n_batch // 2
    for i in range(len(dataset)):
        s = dataset[i]
        if s.get('is_anomaly'):
            if sum(1 for x in samples if x.get('is_anomaly')) < n_a:
                samples.append(s)
        else:
            if sum(1 for x in samples if not x.get('is_anomaly')) < n_batch - n_a:
                samples.append(s)
        if len(samples) >= n_batch:
            break
    packed, _ = pack_sft_batch(collator, device, samples, tokenizer,
                               multibox=True, thinking=True, answer_only=False)

    loss = batch_loss(model, packed, backward_scale=1.0)
    hp = getattr(unwrap_model(model), 'h_prior_adapter', None)
    alpha = hp.alpha
    print(f'loss={float(loss.detach()):.4f}  n_samples={len(samples)}')
    print(f'alpha={float(alpha.detach()):.6f}  alpha.grad={alpha.grad.item():.3e}')

    # expected alpha movement per step at various lr
    g = alpha.grad.item()
    for lr in (1e-3, 1e-2, 1e-1, 0.3, 0.5, 1.0, 3.0):
        print(f'  lr={lr:<6} -> delta_alpha/step={lr*g:.3e}  ~after 625 steps={lr*g*625:.4f}')


if __name__ == '__main__':
    main()
