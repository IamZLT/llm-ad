#!/usr/bin/env python3
"""Verify generate_group_shared_prefill (B=1 prefill + batched decode).

Checks:
  1. (HARD) group=1 shared-prefill greedy == HF generate() greedy. This gates the
     manual decode loop (cache continuation + compute_3d_position_ids) against the
     reference implementation. Must be exact.
  2. (INFO) group=G batch-repeat rows are mutually identical — confirms the repeated
     cache is symmetric (each trajectory starts from the same state).
  3. (INFO) group=G row0 vs group=1 — reports the GatedDeltaNet kernel noise. This is
     EXPECTED to be nonzero: a B=1 prefill and a B=G prefill of identical content
     produce numerically different KV (the torch fallback kernel is batch-size
     dependent, same class of noise documented in verify_loop_cache.py).
  4. sampled-mode smoke test.

Run:
    python scripts/verify_shared_prefill.py [config] [--group N] [--num-samples S] [--max-new N]
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('PYTHONUNBUFFERED', '1')

import torch

from rl.grpo import move_batch
from outcome.engine_multibox import datasets, load_model, validate_config
from outcome.inputs_multibox import OutcomeMultiboxCollator
from outcome.policy import generate_group, generate_group_shared_prefill
from utils.config import load_yaml_config
from utils.common import set_seed


def _ids(comp, prompt_len):
    return comp.ids[prompt_len:].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config', nargs='?', default='configs/qwen35_2b_annos_probe_v2_loop4_rl.yaml')
    ap.add_argument('--group', type=int, default=4)
    ap.add_argument('--num-samples', type=int, default=1)
    ap.add_argument('--max-new', type=int, default=40)
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    validate_config(cfg)
    cfg = dict(cfg)
    cfg['grpo'] = dict(cfg.get('grpo') or {})
    cfg['grpo']['max_new_tokens'] = args.max_new

    set_seed(int(cfg['training']['seed']))
    model, processor, prior = load_model(cfg, fresh_lora=False)
    _, dev, _ = datasets(cfg, processor)
    collator = OutcomeMultiboxCollator(processor, prior, cfg)
    device = next(model.parameters()).device

    hard_ok = True
    for i in range(args.num_samples):
        batch = move_batch(collator([dev[i]]), device)
        prompt_len = int(batch['prompt_len'][0])

        # Check 1 (HARD): exact match against HF generate at group=1.
        ref = generate_group(model, processor, batch, cfg, group=1, sample=False)[0]
        ref_ids = _ids(ref, prompt_len)
        sp1 = generate_group_shared_prefill(model, processor, batch, cfg, group=1, sample=False)[0]
        sp1_ids = _ids(sp1, prompt_len)
        ok1 = sp1_ids == ref_ids
        hard_ok &= ok1
        print(f'[{i}] (1 HARD) group=1 shared-prefill == HF generate: '
              f'{"OK" if ok1 else "MISMATCH"} ({len(sp1_ids)} tokens)', flush=True)

        # Check 2 (INFO): batch-repeat symmetric.
        spg = generate_group_shared_prefill(model, processor, batch, cfg, group=args.group, sample=False)
        assert len(spg) == args.group
        spg_ids = [_ids(c, prompt_len) for c in spg]
        rows_identical = all(ids == spg_ids[0] for ids in spg_ids)
        print(f'[{i}] (2 INFO) group={args.group} rows mutually identical: '
              f'{"YES" if rows_identical else "NO"}', flush=True)

        # Check 3 (INFO): kernel noise between batch-repeat and group=1 (expected nonzero).
        first_div = next((k for k in range(min(len(spg_ids[0]), len(sp1_ids)))
                          if spg_ids[0][k] != sp1_ids[k]), min(len(spg_ids[0]), len(sp1_ids)))
        print(f'[{i}] (3 INFO) group={args.group} row0 vs group=1: first greedy divergence '
              f'at token {first_div}/{len(sp1_ids)} (GatedDeltaNet batch-size kernel noise)', flush=True)

        smp = generate_group_shared_prefill(model, processor, batch, cfg, group=args.group, sample=True)
        assert len(smp) == args.group
        print(f'[{i}] (4) sampled mode OK ({args.group} completions)', flush=True)

    print('RESULT:', 'PASS' if hard_ok else 'FAIL',
          '(hard gate: group=1 shared-prefill must exactly match HF generate)', flush=True)
    sys.exit(0 if hard_ok else 1)


if __name__ == '__main__':
    main()
