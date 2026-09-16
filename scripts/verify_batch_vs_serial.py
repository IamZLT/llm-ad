#!/usr/bin/env python3
"""Verify batched FSM rollout == serial FSM rollout (greedy, deterministic).

Right-padding in decoder-only batch generation shifts position ids; if the
batched path corrupts generation, RL gradients are computed from bad rollouts
and the policy degrades (FPR -> 1.0). Greedy decode is deterministic, so a
group=2 batch must yield two identical trajectories equal to two group=1 runs.
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

    # two serial (single-trajectory) greedy runs
    s1 = run_cached(model, processor, batch, max_stage_tokens=96, max_answer_tokens=192, greedy=True)
    s2 = run_cached(model, processor, batch, max_stage_tokens=96, max_answer_tokens=192, greedy=True)
    print(f'serial equal: {s1.new_ids == s2.new_ids}', flush=True)

    # batched group=2 greedy
    comps = generate_group_staged(model, processor, batch, cfg, group=2, sample=False)
    b0 = comps[0].text
    b1 = comps[1].text
    print(f'batched[0]==batched[1]: {b0 == b1}', flush=True)
    print(f'batched[0]==serial1 : {b0 == s1.text}', flush=True)
    print(f'batched[1]==serial1 : {b1 == s1.text}', flush=True)

    # Also show divergence location if any
    if b0 != s1.text:
        n = min(len(b0), len(s1.text))
        i = next((k for k in range(n) if b0[k] != s1.text[k]), n)
        print(f'first diff at char {i}:', flush=True)
        print(f'  batched0 : ...{b0[max(0,i-40):i+40]!r}', flush=True)
        print(f'  serial1  : ...{s1.text[max(0,i-40):i+40]!r}', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
