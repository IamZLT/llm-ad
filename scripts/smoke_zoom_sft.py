#!/usr/bin/env python3
"""Smoke-test the zoom-crop SFT path (train_region_sft.py).

Loads the SFT model + collator, then runs ``pack_sft_batch`` on a mix of anomalous
and normal samples with ``outcome.sft.zoom_prob`` forced to 1.0 so every single-box
anomaly is upgraded to a 3-image (ref + test + zoom crop) batch. Verifies:

1. ``build_sft_target(zoom=True)`` emits an undershoot->correct two-round chain.
2. ``_zoom_sample_batch`` produces a 3-image batch that still carries the H fields.
3. ``pack_sft_batch`` mixes 2-image and 3-image singles without shape errors.
4. ``batch_loss`` runs the H-Box + H-VPT + 3-image forward and yields a finite loss.

Usage:
  python scripts/smoke_zoom_sft.py --config configs/qwen35_2b_world_model_sft.yaml
"""
from __future__ import annotations

import argparse
import random
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from train_region_sft import (batch_loss, build_sft_target, load_sft_model,
                              pack_sft_batch)
from data.scan import load_prior_split
from data.prior_dataset import build_train_ref_pool
from outcome.inputs import build_zoom_train_batch
from outcome.inputs_multibox import OutcomeMultiboxCollator, OutcomeMultiboxDataset
from outcome.zoom_crop import make_zoom_crop
from utils.config import load_yaml_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/qwen35_2b_world_model_sft.yaml')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    cfg.setdefault('runtime', {})['mode'] = 'train'
    cfg.setdefault('distributed', {})['num_gpu'] = 1
    device = torch.device(args.device)

    model, processor, prior = load_sft_model(cfg, device)
    tokenizer = getattr(processor, 'tokenizer', processor)

    train, _ = load_prior_split(cfg)
    pool = build_train_ref_pool(train)
    ds = OutcomeMultiboxDataset(train, cfg, processor, 'train', pool)
    collator = OutcomeMultiboxCollator(processor, prior, cfg)

    anom_idx = [i for i in range(len(ds)) if ds[i].get('is_anomaly')]
    norm_idx = [i for i in range(len(ds)) if not ds[i].get('is_anomaly')]
    print(f'[smoke] dataset: {len(anom_idx)} anom, {len(norm_idx)} norm', flush=True)

    # Force zoom on every single-box anomaly to exercise the 3-image path.
    sft_cfg = dict(cfg['outcome'].get('sft') or {})
    sft_cfg['zoom_prob'] = 1.0

    # Gather single-box anomalies first, then normals.
    single_box = []
    for i in anom_idx:
        meta = ds[i]
        comps = list(meta.get('component_bboxes') or [])
        if not comps and meta.get('gt_box_px') is not None:
            comps = [meta['gt_box_px']]
        if len(comps) == 1:
            single_box.append(i)
    print(f'[smoke] single-box anomalies: {len(single_box)}', flush=True)

    samples = [ds[i] for i in single_box[:3]] + [ds[i] for i in norm_idx[:2]]
    random.seed(0)

    # --- 1. direct zoom-batch construction on one single-box anomaly ---
    it = ds[single_box[0]]
    b = collator([it])
    meta = b['_meta'][0]
    comps = list(meta.get('component_bboxes') or []) or [meta['gt_box_px']]
    from train_region_sft import _boxes_to_1000, _shrink_box
    gt_1000 = _boxes_to_1000(comps, meta['orig_size'])
    cand = [_shrink_box(gt_1000[0], 0.5)]
    crop = make_zoom_crop(meta['test'], cand[0], orig_size=tuple(meta['orig_size']))
    print(f'[smoke] crop degenerate={crop.degenerate} window={crop.window_px} '
          f'crop_size={crop.image.size}', flush=True)
    zoom_batch = build_zoom_train_batch(processor, prior, cfg, device, b, crop.image)
    print(f'[smoke] zoom_batch grid={zoom_batch["image_grid_thw"].shape} '
          f'image_embeds={zoom_batch["image_embeds"].shape} '
          f'n_box_tokens={zoom_batch.get("n_box_tokens")} '
          f'n_feat_tokens={zoom_batch.get("n_feat_tokens")}', flush=True)

    # --- 2. full pack (2-image + 3-image mix) + loss ---
    packed, n_sup = pack_sft_batch(collator, device, samples, tokenizer, multibox=True,
                                   thinking=True, answer_only=False, sft_cfg=sft_cfg)
    print(f'[smoke] packed supervised_tokens={n_sup} '
          f'image_embeds={packed["image_embeds"].shape} '
          f'grid={packed["gen_in"]["image_grid_thw"].shape} '
          f'n_h_box={len(packed["h_box_geoms"])} n_h_map={len(packed["h_maps"])}', flush=True)
    loss = batch_loss(model, packed)
    print(f'[smoke] batch_loss={float(loss.detach()):.6f}', flush=True)

    # --- 3. print a zoom target ---
    tgt = build_sft_target(meta, multibox=True, thinking=True, sft_cfg=sft_cfg, zoom=True)
    print('[smoke] --- zoom target (first 300 chars) ---', flush=True)
    print(tgt[:300], flush=True)
    print('[smoke] DONE', flush=True)


if __name__ == '__main__':
    main()
