#!/usr/bin/env python3
"""Gradient sanity check for the HMemory cross-attn under gradient checkpointing.

Loads the frozen FSM baseline + fresh (trainable) HMemory, runs one teacher-forced
step with backward, and reports whether the encoder + cross-attn receive finite
gradients. Catches the checkpointing/hook interaction early, before any long run.
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
    cfg = load_yaml_config(sys.argv[1] if len(sys.argv) > 1
                           else 'configs/qwen35_2b_annos_probe_v2_fsm_h_mem_sft.yaml')
    set_seed(int(cfg['training']['seed']))
    device = torch.device('cuda', 0)

    model, processor, prior = load_sft_model(
        cfg, device, init_sft=cfg['outcome']['sft_adapter'],
        freeze_region=True, freeze_lora=True)
    tokenizer = getattr(processor, 'tokenizer', processor)
    model.train()

    train, _ = load_prior_split(cfg)
    train, _ = split_holdout_by_class(train, float(cfg['data']['holdout_ratio']), seed=int(cfg['training']['seed']))
    pool = build_train_ref_pool(train)
    dataset = OutcomeMultiboxDataset(train, cfg, processor, 'train', pool)
    collator = OutcomeMultiboxCollator(processor, prior, cfg)
    ls = str((cfg.get('outcome') or {}).get('sft', {}).get('localize_target', 'h'))

    samples = [dataset[i] for i in range(2)]
    packed, n_sup = pack_sft_batch(collator, device, samples, tokenizer,
                                   multibox=True, thinking=True, answer_only=False,
                                   localize_source=ls)
    print(f'[pack] supervised_tokens={n_sup} h_maps={len(packed["h_maps"])}', flush=True)

    core = unwrap_model(model)
    hm = core.h_memory
    assert hm is not None
    # freeze everything except HMemory
    for n, p in model.named_parameters():
        p.requires_grad = (n.startswith('h_memory') if 'h_memory' in n else False)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f'[opt] trainable HMemory params={len(trainable)}', flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.0)

    for step in range(3):
        opt.zero_grad(set_to_none=True)
        loss = batch_loss(model, packed, backward_scale=1.0)
        grads = {}
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                grads[n] = p.grad.abs().max().item()
        gnames = list(grads.keys())
        finite = all(torch.isfinite(torch.tensor(v)) for v in grads.values())
        print(f'[step {step}] loss={float(loss.detach()):.4f} n_grad={len(grads)} finite={finite}',
              flush=True)
        for n in gnames[:4]:
            print(f'    {n}: |grad|max={grads[n]:.3e}', flush=True)
        gates = [float(torch.tanh(ca.gate.detach())) for ca in hm.cross_attns.values()]
        print(f'    gates={gates}', flush=True)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
    print('DONE')


if __name__ == '__main__':
    main()
