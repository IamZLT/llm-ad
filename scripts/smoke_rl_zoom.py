#!/usr/bin/env python3
"""Smoke-test the scheme-B RL zoom training path (engine_multibox.py).

Loads the base model + zoom SFT adapter, then on a single-box anomalous sample:

1. Builds the 2-image collator batch.
2. Upgrades it via ``_maybe_zoom_train_batch`` to a 3-image (ref + test + crop) batch.
3. Runs ``generate_group(group=8)`` on the 3-image batch and checks all 8 trajectories
   complete and parse.

Usage:
  python scripts/smoke_rl_zoom.py --config configs/qwen35_2b_world_model_rl.yaml
"""
from __future__ import annotations

import argparse
import random
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from utils.config import load_yaml_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/qwen35_2b_world_model_rl.yaml')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    cfg.setdefault('runtime', {})['mode'] = 'train'
    cfg.setdefault('distributed', {})['num_gpu'] = 1
    device = torch.device(args.device)

    from outcome.engine_multibox import load_model, _maybe_zoom_train_batch
    model, processor, prior = load_model(cfg)
    model = model.to(device)

    from data.scan import load_prior_split
    from data.prior_dataset import build_train_ref_pool
    from outcome.inputs_multibox import OutcomeMultiboxCollator, OutcomeMultiboxDataset
    from outcome.policy import generate_group
    from outcome.protocol_multibox import parse_output_cfg
    from rl.grpo import move_batch

    train, _ = load_prior_split(cfg)
    pool = build_train_ref_pool(train)
    ds = OutcomeMultiboxDataset(train, cfg, processor, 'train', pool)
    collator = OutcomeMultiboxCollator(processor, prior, cfg)

    # Find a single-box anomaly.
    anom_single = None
    for i in range(len(ds)):
        it = ds[i]
        if not it.get('is_anomaly'):
            continue
        comps = list(it.get('component_bboxes') or [])
        if not comps and it.get('gt_box_px') is not None:
            comps = [it['gt_box_px']]
        if len(comps) == 1:
            anom_single = it
            break
    assert anom_single is not None, 'no single-box anomaly found'
    print(f'[smoke] sample class={anom_single["class_name"]} '
          f'gt={anom_single.get("gt_box_px")} comps={anom_single.get("component_bboxes")}', flush=True)

    batch = move_batch(collator([anom_single]), device)
    print(f'[smoke] base grid={batch["image_grid_thw"].shape} '
          f'image_embeds={batch["image_embeds"].shape}', flush=True)

    # Force the zoom path.
    cfg['outcome']['zoom']['enabled'] = True
    cfg['outcome']['zoom']['train_prob'] = 1.0
    rng = random.Random(42)
    zbatch = _maybe_zoom_train_batch(cfg, processor, prior, device, batch, rng)
    print(f'[smoke] zoom grid={zbatch["image_grid_thw"].shape} '
          f'image_embeds={zbatch["image_embeds"].shape} '
          f'image_count={zbatch["_meta"][0].get("image_count")} '
          f'prompt_tokens={zbatch["_meta"][0].get("prompt_tokens")} '
          f'visual_tokens={zbatch["_meta"][0].get("visual_tokens")}', flush=True)

    comps = generate_group(model, processor, zbatch, cfg, group=8, sample=True)
    print(f'[smoke] generated {len(comps)} trajectories', flush=True)
    for k, c in enumerate(comps):
        parsed = parse_output_cfg(c.text, cfg, max_boxes=int(cfg['outcome']['max_boxes']))
        print(f'[smoke]   [{k}] stop={c.stop_reason} task_valid={parsed.get("task_valid")} '
              f'n_tok={len(c.ids)-int(zbatch["prompt_len"][0])} '
              f'imagine={parsed.get("imagine_action")} verify={parsed.get("verify_action")}', flush=True)

    print('[smoke] DONE', flush=True)


if __name__ == '__main__':
    main()
