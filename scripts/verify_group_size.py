#!/usr/bin/env python3
"""Isolate the group>1 divergence: group=1 batch should equal serial re-prefill.

If group=1 == serial but group=2 != serial, the bug is in the batch expansion
(image_grid_thw / mm_token_type_ids) for group>1. Greedy decode is deterministic.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('PYTHONUNBUFFERED', '1')

import torch

from outcome.engine_multibox import datasets, load_model, validate_config
from outcome.inputs_multibox import OutcomeMultiboxCollator
from outcome.policy import generate_group_staged
from outcome.staged_decode import run_cached
from rl.grpo import move_batch
from utils.config import load_yaml_config
from utils.common import set_seed


def main():
    cfg = load_yaml_config('configs/qwen35_2b_annos_probe_v2_fsm_rl.yaml')
    validate_config(cfg)
    set_seed(int(cfg['training']['seed']))
    model, processor, prior = load_model(cfg, adapter=None, fresh_lora=False)
    model.eval()
    _, dev, test = datasets(cfg, processor)
    collator = OutcomeMultiboxCollator(processor, prior, cfg)
    device = next(model.parameters()).device
    batch = move_batch(collator([dev[0]]), device)

    kv = run_cached(model, processor, batch, max_stage_tokens=96, max_answer_tokens=192, greedy=True).text

    g1 = generate_group_staged(model, processor, batch, cfg, group=1, sample=False)[0].text
    g2 = generate_group_staged(model, processor, batch, cfg, group=2, sample=False)

    print(f'kv == group=1: {kv == g1}', flush=True)
    print(f'group=1 == group=2[0]: {g1 == g2[0].text}', flush=True)
    print(f'group=2[0] == group=2[1]: {g2[0].text == g2[1].text}', flush=True)
    print(f'kv == group=2[0]: {kv == g2[0].text}', flush=True)

    for a, b, na, nb in ((kv, g1, 'kv', 'g1'), (g1, g2[0].text, 'g1', 'g2[0]'), (kv, g2[0].text, 'kv', 'g2[0]')):
        if a != b:
            n = min(len(a), len(b))
            i = next((k for k in range(n) if a[k] != b[k]), n)
            print(f'--- diff {na} vs {nb} at char {i}:', flush=True)
            print(f'  {na}: ...{a[max(0,i-30):i+30]!r}', flush=True)
            print(f'  {nb}: ...{b[max(0,i-30):i+30]!r}', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
