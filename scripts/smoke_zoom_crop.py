#!/usr/bin/env python3
"""Smoke-test the two-stage zoom-crop rollout (world-model observation step).

Loads the base model + frozen SFT adapter, then runs BOTH the single-pass
(generate_group) and two-stage (generate_group_zoom) rollouts on a few anomaly
samples so we can verify:

1. Stage 1 stops at ``[localize]`` and yields a parseable candidate box.
2. The zoom crop is non-degenerate and the second 3-image pass completes.
3. The concatenated text still parses as one uninterrupted think chain.

Usage:
  python scripts/smoke_zoom_crop.py --config configs/qwen35_2b_world_model_rl.yaml --limit 3
"""
from __future__ import annotations

import argparse
import sys

import torch

sys.path.insert(0, '.')

from utils.common import set_seed
from utils.config import load_yaml_config
from rl.grpo import move_batch
from outcome.engine_multibox import datasets, load_model
from outcome.inputs_multibox import OutcomeMultiboxCollator
from outcome.policy import generate_group, generate_group_zoom
from outcome.protocol_multibox import parse_output_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/qwen35_2b_world_model_rl.yaml')
    ap.add_argument('--limit', type=int, default=3)
    ap.add_argument('--adapter', default=None, help='optional RL LoRA checkpoint')
    ap.add_argument('--sft-adapter', default=None, help='override outcome.sft_adapter')
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    cfg['outcome'].setdefault('zoom', {})['enabled'] = True
    if args.sft_adapter:
        cfg['outcome']['sft_adapter'] = args.sft_adapter
    set_seed(int(cfg['training']['seed']))

    model, processor, prior = load_model(cfg, args.adapter)
    device = next(model.parameters()).device
    _, _, test_set = datasets(cfg, processor)
    collator = OutcomeMultiboxCollator(processor, prior, cfg)
    max_boxes = int(cfg['outcome'].get('max_boxes', 16))

    n_done = 0
    for idx in range(len(test_set)):
        item = test_set[idx]
        if not item.get('is_anomaly'):
            continue
        batch = move_batch(collator([item]), device)
        meta = batch['_meta'][0]

        print(f"\n===== sample {idx} class={meta['class_name']} defect={meta.get('defect_type')} =====", flush=True)

        # Single-pass baseline.
        base = generate_group(model, processor, batch, cfg, group=1, sample=False)[0]
        bparsed = parse_output_cfg(base.text, cfg, max_boxes=max_boxes)
        print(f"[single] candidate={bparsed['candidate_bboxes_2d']} "
              f"imagine={bparsed['imagine_action']} confirm={bparsed['verify_action']} "
              f"answer_boxes={bparsed['bboxes_2d']}", flush=True)

        # Two-stage zoom.
        zoom = generate_group_zoom(model, processor, prior, batch, cfg, group=1, sample=False)[0]
        zparsed = parse_output_cfg(zoom.text, cfg, max_boxes=max_boxes)
        print(f"[zoom  ] candidate={zparsed['candidate_bboxes_2d']} "
              f"imagine={zparsed['imagine_action']} confirm={zparsed['verify_action']} "
              f"answer_boxes={zparsed['bboxes_2d']} stop={zoom.stop_reason}", flush=True)
        print(f"[zoom  ] think_headers={zparsed['think_headers']}", flush=True)
        print(f"[zoom  ] text preview: {zoom.text[:400]}", flush=True)

        n_done += 1
        if n_done >= args.limit:
            break

    print("\n[done]", flush=True)


if __name__ == '__main__':
    main()
