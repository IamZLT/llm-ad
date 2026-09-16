#!/usr/bin/env python3
"""Benchmark batched FSM rollout (group=8) vs. serial single-trajectory."""
import os
import sys
import time
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

    # warmup
    _ = generate_group_staged(model, processor, batch, cfg, group=4, sample=True)

    # serial: one trajectory (extrapolate to group=8)
    t0 = time.perf_counter()
    _ = run_cached(model, processor, batch, max_stage_tokens=96, max_answer_tokens=192, greedy=False)
    t_serial_1 = time.perf_counter() - t0

    # batched group=8, 3 runs
    for rep in range(3):
        t0 = time.perf_counter()
        comps = generate_group_staged(model, processor, batch, cfg, group=8, sample=True)
        dt = time.perf_counter() - t0
        print(f'batched group=8 rep{rep}: {dt:.2f}s (n={len(comps)})', flush=True)

    print(f'serial single-trajectory: {t_serial_1:.2f}s -> x8 = {t_serial_1 * 8:.2f}s', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
