#!/usr/bin/env python3
"""Verify K=1 looped forward == original Qwen3.5 forward (bit-comparable logits).

The structural fix (decoder layers only, final RMSNorm once) must be numerically
identical to the original stack when K=1. If K=1 != Direct, K=2/K=4 are built on
a broken base and must not be trained.

Approach: load a loop-enabled model (the SFT weights the RL training starts from),
run one full forward through the patched ``looped_forward`` at ``_loop_steps=1``,
then one full forward through the saved original forward (``_loop_original_forward``),
and compare the last-position logits.

Run:
    python scripts/verify_loop_k1.py [config] [--num-samples S] [--atol T]
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('PYTHONUNBUFFERED', '1')

import torch

from models.looped_qwen import get_qwen_text_model, set_loop_steps
from models.qwen35 import unwrap_model
from models.vision_cache import bind_cached_image_features
from rl.grpo import model_inputs, move_batch
from outcome.engine_multibox import datasets, load_model, validate_config
from outcome.inputs_multibox import OutcomeMultiboxCollator
from outcome.policy import _region_bind
from utils.config import load_yaml_config
from utils.common import set_seed


def _forward_logits(model, batch, use_cache=False):
    core = unwrap_model(model)
    with torch.no_grad(), bind_cached_image_features(core, batch['image_embeds']), _region_bind(model, batch):
        out = core(**model_inputs(batch), use_cache=use_cache)
    return out.logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config', nargs='?', default='configs/qwen35_2b_annos_probe_v2_loop4_rl.yaml')
    ap.add_argument('--num-samples', type=int, default=2)
    ap.add_argument('--atol', type=float, default=1e-3)
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    validate_config(cfg)
    set_seed(int(cfg['training']['seed']))
    model, processor, prior = load_model(cfg, fresh_lora=False)
    _, dev, _ = datasets(cfg, processor)
    collator = OutcomeMultiboxCollator(processor, prior, cfg)
    device = next(model.parameters()).device

    core = unwrap_model(model)
    text_model = get_qwen_text_model(model)
    assert hasattr(text_model, '_loop_original_forward'), 'loop patch not installed'

    all_ok = True
    for i in range(args.num_samples):
        batch = move_batch(collator([dev[i]]), device)

        # Patched forward at K=1.
        set_loop_steps(model, 1)
        logits_loop = _forward_logits(model, batch, use_cache=False)

        # Original forward (temporarily restore the saved bound method).
        saved = text_model.forward
        text_model.forward = text_model._loop_original_forward
        try:
            logits_orig = _forward_logits(model, batch, use_cache=False)
        finally:
            text_model.forward = saved

        diff = (logits_loop - logits_orig).abs().max().item()
        argmax_match = bool(torch.equal(logits_loop[:, -1, :].argmax(-1),
                                        logits_orig[:, -1, :].argmax(-1)))
        ok = diff <= args.atol
        all_ok &= ok
        print(f'[{i}] K=1 vs original: max|logit diff|={diff:.6f} (atol={args.atol}) '
              f'last-token argmax match={argmax_match} -> {"OK" if ok else "MISMATCH"}', flush=True)

    print('RESULT:', 'PASS' if all_ok else 'FAIL',
          '(K=1 looped forward must be numerically identical to the original stack)', flush=True)
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
