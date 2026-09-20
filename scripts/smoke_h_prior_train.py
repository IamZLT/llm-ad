#!/usr/bin/env python3
"""Smoke-test the H-Region Fusion mechanism-probe training path.

Freeze ViT / Qwen base / FSM LoRA / RegionAdapter; train ONLY HPriorAdapter + alpha.
Runs one forward+backward on a small batch and reports which parameter groups
actually receive gradients, so a full run is not launched on a broken graph.
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

    anom = None
    for i in range(len(dataset)):
        if dataset[i].get('is_anomaly'):
            anom = dataset[i]
            break
    packed, _ = pack_sft_batch(collator, device, [anom], tokenizer,
                               multibox=True, thinking=True, answer_only=False)

    loss = batch_loss(model, packed, backward_scale=1.0)
    print(f'loss={float(loss):.4f}')

    core = unwrap_model(model)

    def grad_report(name, module):
        if module is None:
            return
        has = sum(p.grad is not None for p in module.parameters())
        tot = sum(1 for _ in module.parameters())
        print(f'  {name}: {has}/{tot} params have grad')

    grad_report('HPriorAdapter', getattr(core, 'h_prior_adapter', None))
    grad_report('RegionAdapter', getattr(core, 'region_adapter', None))

    # LoRA params live under the PeftModel adapter; count how many (if any) got grad.
    lora_grad = 0
    lora_total = 0
    for n, p in core.named_parameters():
        if 'lora' in n.lower():
            lora_total += 1
            lora_grad += int(p.grad is not None)
    print(f'  LoRA: {lora_grad}/{lora_total} params have grad')

    hp = getattr(core, 'h_prior_adapter', None)
    if hp is not None:
        print(f'  alpha={float(hp.alpha.detach()):.6f} alpha.grad={hp.alpha.grad.item() if hp.alpha.grad is not None else None}')

    # sanity: no grads must flow to the frozen region adapter or LoRA
    ra = getattr(core, 'region_adapter', None)
    ra_grad = sum(p.grad is not None for p in ra.parameters()) if ra is not None else 0
    hp_grad = sum(p.grad is not None for p in hp.parameters()) if hp is not None else 0
    ok = hp_grad > 0 and ra_grad == 0 and lora_grad == 0
    print('SMOKE', 'OK' if ok else 'FAIL')


if __name__ == '__main__':
    main()
