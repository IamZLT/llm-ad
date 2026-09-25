#!/usr/bin/env python3
"""Comprehensive zoom-crop effectiveness evaluation (no fine-tuning).

Compares the single-pass (2-image global) rollout against the two-stage zoom-crop
(3-image) rollout on the SAME samples, using the post-refactor SFT adapter. The
question is narrow and concrete: does giving the model a zoomed, outline-drawn crop
of its candidate box let it correct extent errors (the known "box too small"
failure) that the global view cannot?

Per-sample metrics (baseline vs zoom):
  mask_iou / union_iou   — final box quality vs GT
  delta_refine           — final vs first-candidate localization improvement
  imagine_correct        — [imagine] box-quality rehearsal vs objective verdict
  discrim_correct        — [confirm] refine-loop verdict vs sign(delta_refine)
  candidate -> final IoU delta — did the box move toward GT

Usage:
  python scripts/eval_zoom_effectiveness.py \
      --config configs/qwen35_2b_world_model_rl.yaml \
      --sft-adapter outputs/train/world_model_sft_2b_hvpt_768 \
      --n-anomaly 45 --n-normal 15
"""
from __future__ import annotations

import argparse
import copy
import random
import sys
from collections import defaultdict

import torch

sys.path.insert(0, '.')

from rl.grpo import move_batch
from outcome.engine_multibox import datasets, load_model
from outcome.inputs_multibox import OutcomeMultiboxCollator
from outcome.policy import generate_group, generate_group_zoom
from outcome.protocol import to_pixels
from outcome.protocol_multibox import parse_output_cfg, score_output
from utils.common import set_seed
from utils.config import load_yaml_config


def _size_bin(meta, anomaly):
    if not anomaly:
        return 'normal'
    frac = meta.get('mask_area_fraction')
    if frac is None:
        gt = meta.get('gt_box_px')
        frac = (gt[2]-gt[0])*(gt[3]-gt[1])/(meta['orig_size'][0]*meta['orig_size'][1]) if gt else 0.0
    return 'small' if frac < .02 else 'medium' if frac < .1 else 'large'


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def _best_iou(boxes, comps):
    if not boxes or not comps:
        return 0.0
    return max(max(_iou(b, g) for b in boxes) for g in comps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/qwen35_2b_world_model_rl.yaml')
    ap.add_argument('--sft-adapter', default='outputs/train/world_model_sft_2b_hvpt_768')
    ap.add_argument('--n-anomaly', type=int, default=45)
    ap.add_argument('--n-normal', type=int, default=15)
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    cfg['outcome']['sft_adapter'] = args.sft_adapter
    set_seed(int(cfg['training']['seed']))

    model, processor, prior = load_model(cfg, adapter=None, fresh_lora=False)
    device = next(model.parameters()).device
    _, _, test_set = datasets(cfg, processor)
    collator = OutcomeMultiboxCollator(processor, prior, cfg)
    max_boxes = int(cfg['outcome'].get('max_boxes', 16))
    loc = {**cfg['outcome'].get('localization', {}), **cfg['outcome'].get('reward', {})}

    # Stratified sampling: anomaly split by size_bin (so we don't miss the small-defect
    # extent case), plus a balanced normal set.
    anom_by_bin = defaultdict(list)
    normal = []
    for i in range(len(test_set)):
        it = test_set[i]
        if it.get('is_anomaly'):
            anom_by_bin[_size_bin(it, True)].append(i)
        else:
            normal.append(i)
    rng = random.Random(123)
    per_bin = max(1, args.n_anomaly // max(1, len(anom_by_bin)))
    picks = []
    for bin_name in ('small', 'medium', 'large'):
        idxs = rng.sample(anom_by_bin.get(bin_name, []), min(per_bin, len(anom_by_bin.get(bin_name, []))))
        picks.extend(idxs)
    picks.extend(rng.sample(normal, min(args.n_normal, len(normal))))

    zoom_cfg = copy.deepcopy(cfg)
    zoom_cfg['outcome'].setdefault('zoom', {})['enabled'] = True

    rows = []
    for idx in picks:
        item = test_set[idx]
        batch = move_batch(collator([item]), device)
        meta = batch['_meta'][0]
        anomaly = bool(meta['is_anomaly'])

        base = generate_group(model, processor, batch, cfg, group=1, sample=False)[0]
        zoom = generate_group_zoom(model, processor, prior, batch, zoom_cfg, group=1, sample=False)[0]

        bp = parse_output_cfg(base.text, cfg, max_boxes=max_boxes)
        zp = parse_output_cfg(zoom.text, cfg, max_boxes=max_boxes)
        bs = score_output(bp, meta, float(cfg['outcome']['protocol_weight']), loc, max_boxes=max_boxes)
        zs = score_output(zp, meta, float(cfg['outcome']['protocol_weight']), loc, max_boxes=max_boxes)

        comps = meta.get('component_bboxes') or ([meta['gt_box_px']] if meta.get('gt_box_px') else [])
        comps = [to_pixels(c, meta['orig_size']) for c in comps] if anomaly else []
        b_cand = [to_pixels(c, meta['orig_size']) for c in bp['candidate_bboxes_2d']]
        b_final = [to_pixels(c, meta['orig_size']) for c in bp['bboxes_2d']]
        z_cand = [to_pixels(c, meta['orig_size']) for c in zp['candidate_bboxes_2d']]
        z_final = [to_pixels(c, meta['orig_size']) for c in zp['bboxes_2d']]

        rows.append(dict(
            class_name=meta['class_name'], defect=meta.get('defect_type'), anomaly=anomaly,
            size_bin=_size_bin(meta, anomaly),
            b_mask_iou=bs['mask_iou'], z_mask_iou=zs['mask_iou'],
            b_union_iou=bs['union_iou'], z_union_iou=zs['union_iou'],
            b_delta=bs['delta_refine'], z_delta=zs['delta_refine'],
            b_imagine=bs['imagine_correct'], z_imagine=zs['imagine_correct'],
            b_discrim=bs['discrim_correct'], z_discrim=zs['discrim_correct'],
            b_cand_iou=_best_iou(b_cand, comps) if anomaly else None,
            b_final_iou=_best_iou(b_final, comps) if anomaly else None,
            z_cand_iou=_best_iou(z_cand, comps) if anomaly else None,
            z_final_iou=_best_iou(z_final, comps) if anomaly else None,
            b_pred=bool(bp['is_anomaly']), z_pred=bool(zp['is_anomaly']),
            b_headers=bp['think_headers'], z_headers=zp['think_headers'],
            z_stop=zoom.stop_reason,
        ))

    # ---- Summarize ----
    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else float('nan')

    anom = [r for r in rows if r['anomaly']]
    norm = [r for r in rows if not r['anomaly']]

    print('\n==================== ZOOM EFFECTIVENESS ====================')
    print(f"samples: {len(anom)} anomaly / {len(norm)} normal")
    print(f"\n{'metric':<22}{'baseline':>12}{'zoom':>12}{'delta':>12}")
    for name, bkey, zkey in [
        ('mask_iou (anom)', 'b_mask_iou', 'z_mask_iou'),
        ('union_iou (anom)', 'b_union_iou', 'z_union_iou'),
        ('final best-iou (anom)', 'b_final_iou', 'z_final_iou'),
        ('candidate best-iou', 'b_cand_iou', 'z_cand_iou'),
        ('delta_refine', 'b_delta', 'z_delta'),
        ('imagine_correct', 'b_imagine', 'z_imagine'),
        ('discrim_correct', 'b_discrim', 'z_discrim'),
    ]:
        b = mean(r[bkey] for r in anom)
        z = mean(r[zkey] for r in anom)
        print(f'{name:<22}{b:>12.4f}{z:>12.4f}{z-b:>+12.4f}')

    print(f"\n--- normal samples (TNR = fraction judged normal) ---")
    print(f"baseline TNR={mean(r['b_pred'] is False for r in norm):.3f}  "
          f"zoom TNR={mean(r['z_pred'] is False for r in norm):.3f}")

    print(f"\n--- mask_iou by size bin (baseline -> zoom) ---")
    for bin_name in ('small', 'medium', 'large'):
        sub = [r for r in anom if r['size_bin'] == bin_name]
        if not sub:
            continue
        b = mean(r['b_mask_iou'] for r in sub)
        z = mean(r['z_mask_iou'] for r in sub)
        print(f"  {bin_name:<8} n={len(sub):<3} {b:.4f} -> {z:.4f}  ({z-b:+.4f})")

    # Extent-correction hit rate: fraction of anomaly samples where the final box
    # moved toward GT (vs candidate) by a meaningful margin, and where zoom beat baseline.
    improved = sum(1 for r in anom if r['z_final_iou'] > r['b_final_iou'] + 1e-3)
    degraded = sum(1 for r in anom if r['z_final_iou'] < r['b_final_iou'] - 1e-3)
    print(f"\n--- zoom vs baseline final box (anomaly) ---")
    print(f"  zoom better: {improved}/{len(anom)}   zoom worse: {degraded}/{len(anom)}   "
          f"tie: {len(anom)-improved-degraded}/{len(anom)}")

    # Per-sample detail dump for the small defects (the extent bottleneck).
    print(f"\n--- small-defect detail (baseline -> zoom final best-iou) ---")
    for r in sorted([r for r in anom if r['size_bin'] == 'small'], key=lambda r: -(r['z_final_iou']-r['b_final_iou'])):
        print(f"  {r['class_name']:<16} {str(r['defect']):<22} "
              f"cand={r['b_cand_iou']:.3f} final={r['b_final_iou']:.3f} -> {r['z_final_iou']:.3f} "
              f"({r['z_final_iou']-r['b_final_iou']:+.3f})  hdr={r['z_headers']}")

    print('\n==================== END ====================')


if __name__ == '__main__':
    main()
