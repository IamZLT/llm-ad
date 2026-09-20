#!/usr/bin/env python3
"""Layer-1 (training-free) representation-resolution diagnostic.

Quantifies, on SMALL defects only, how much spatial localization information
survives in each candidate representation. This answers the question that gates
the cross-attention decision:

    does the spatial merger lose localization?
    is dense pre-merger >> fused H (what FSM currently uses)?

Representations compared (all computed from the SAME frozen vision tower, H@768):

    block12 / block16 / block20 / block24 : per pre-merger layer nearest-reference
                                            distance map S_l = 1 - cos_sim(f_t, f_r)
    fused_H                              : softmax-weighted fusion of the 4 maps
                                            (what FSM currently feeds downstream)
    merged                               : nearest-reference distance recomputed on
                                            the post-merger vision features (coarse grid)

Per representation we run the SAME ``region_proposals`` (relative-threshold CC ->
top-K boxes) and measure, against GT component boxes (small anomalies only):

    1. component recall @ IoU 0.1 / 0.3   (hit rate)
    2. best proposal IoU                   (mean, how tight the best proposal is)
    3. peak-in-GT rate                     (does the top-1 peak land on the defect)
    4. boundary coverage                   (GT perimeter cells inside proposal union)
    5. best achievable CC IoU              (ceiling over a threshold scan)

A faithful-enough pipeline: we reuse the production collator's align+tokenize for
H@768 and ``AnomalyPrior._encode_one``/``_nn_map`` (the same 2-pass code path as
``AnomalyPrior.forward``). Minor BF16 differences vs the joint ``encode_pair_canonical``
path are irrelevant for this relative comparison.

Run:
    python scripts/diagnose_rep_resolution.py --limit 3          # smoke
    python scripts/diagnose_rep_resolution.py                    # all small
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from utils.config import load_yaml_config
from utils.common import set_seed
from models.anomaly_prior import softmax_fuse_maps
from outcome.inputs import pixel_budget, region_proposals
from outcome.engine_multibox import datasets, load_model
from outcome.protocol import iou, to_pixels

SMALL_AREA_FRAC = 0.02  # matches evaluate_multibox._size_bin 'small'


# --------------------------------------------------------------------------- #
# representation extraction
# --------------------------------------------------------------------------- #
def extract_all(prior, ref_pv, ref_grid, test_pv, test_grid):
    """Return per-layer distance maps, fused H, and merged-resolution distance map."""
    f_r, hw_r, v_r = prior._encode_one(ref_pv, ref_grid)
    f_t, hw_t, v_t = prior._encode_one(test_pv, test_grid)
    maps = [prior._nn_map(ft, fr, hw_t, hw_r, prior.neighborhood_radius)
            for ft, fr in zip(f_t, f_r)]
    stack = torch.stack(maps)  # [L, Ht, Wt]
    fused = softmax_fuse_maps(stack, prior.temperature) if len(maps) > 1 else stack[0]

    m = int(prior.spatial_merge_size)
    mhw_t = (int(hw_t[0]) // m, int(hw_t[1]) // m)
    mhw_r = (int(hw_r[0]) // m, int(hw_r[1]) // m)
    if int(v_t.shape[0]) != mhw_t[0] * mhw_t[1]:
        raise RuntimeError(
            f'merged test count {v_t.shape[0]} != {mhw_t} (hw_t={hw_t}, merge={m})')
    s_merged = prior._nn_map(v_t, v_r, mhw_t, mhw_r, prior.neighborhood_radius)

    return dict(maps=maps, fused=fused, s_merged=s_merged,
                hw_t=hw_t, hw_r=hw_r, mhw_t=mhw_t,
                block_indices=list(prior.block_indices))


# --------------------------------------------------------------------------- #
# per-representation localization metrics (vs GT component boxes in px)
# --------------------------------------------------------------------------- #
def _cc_boxes_1000(hmap, prior_cfg, max_candidates):
    """All CC boxes of ``hmap`` (thresholded) as [0,1000] boxes, uncapped by top-K."""
    cfg = dict(prior_cfg)
    cfg['max_candidates'] = int(max_candidates)
    proposals, _, _ = region_proposals(hmap, cfg)
    return [p['bbox_2d'] for p in proposals], [p for p in proposals]


def _best_iou_px(pred_px, comps):
    vals = [iou(p, g) for p in pred_px for g in comps]
    return max(vals) if vals else 0.0


def _recall_at(pred_px, comps, t):
    if not comps:
        return 0.0
    return sum(1 for g in comps if any(iou(p, g) >= t for p in pred_px)) / len(comps)


def _peak_in_gt(proposals, comps, orig):
    if not proposals or not comps:
        return False
    peak = proposals[0].get('peak_2d')
    if not peak or len(peak) != 2:
        return False
    px = float(peak[0]) / 1000.0 * orig[0]
    py = float(peak[1]) / 1000.0 * orig[1]
    return any(g[0] <= px <= g[2] and g[1] <= py <= g[3] for g in comps)


def _boundary_coverage(proposals, comps, orig, hw):
    """Fraction of GT (union) perimeter cells covered by the proposal-mask union.

    GT comps (px) -> union box -> grid cells -> perimeter; proposal masks are the
    CC masks on the same grid (region_proposals returns masks via proposal['_mask']
    only when we request them -- we recompute the union mask from boxes instead).
    """
    if not proposals or not comps:
        return 0.0
    Ht, Wt = int(hw[0]), int(hw[1])
    # GT union box in grid cell space
    gx0 = min(g[0] for g in comps) / orig[0] * Wt
    gy0 = min(g[1] for g in comps) / orig[1] * Ht
    gx1 = max(g[2] for g in comps) / orig[0] * Wt
    gy1 = max(g[3] for g in comps) / orig[1] * Ht
    gx0, gx1 = int(np.floor(gx0)), int(np.ceil(gx1))
    gy0, gy1 = int(np.floor(gy0)), int(np.ceil(gy1))
    gx0 = max(0, min(Wt - 1, gx0)); gx1 = max(0, min(Wt, gx1))
    gy0 = max(0, min(Ht - 1, gy0)); gy1 = max(0, min(Ht, gy1))
    if gx1 <= gx0 or gy1 <= gy0:
        return 0.0
    # proposal union mask on grid
    union = np.zeros((Ht, Wt), dtype=bool)
    for p in proposals:
        b = p['bbox_2d']  # [0,1000]
        bx0 = int(np.floor(b[0] / 1000.0 * Wt)); bx1 = int(np.ceil(b[2] / 1000.0 * Wt))
        by0 = int(np.floor(b[1] / 1000.0 * Ht)); by1 = int(np.ceil(b[3] / 1000.0 * Ht))
        bx0 = max(0, min(Wt, bx0)); bx1 = max(0, min(Wt, bx1))
        by0 = max(0, min(Ht, by0)); by1 = max(0, min(Ht, by1))
        union[by0:by1, bx0:bx1] = True
    # GT perimeter cells (boundary of the filled GT box)
    gt_mask = np.zeros((Ht, Wt), dtype=bool)
    gt_mask[gy0:gy1, gx0:gx1] = True
    inner = np.zeros((Ht, Wt), dtype=bool)
    inner[gy0 + 1:gy1 - 1, gx0 + 1:gx1 - 1] = True
    boundary = gt_mask & ~inner
    if boundary.sum() == 0:
        return 0.0
    return float((boundary & union).sum()) / float(boundary.sum())


def _best_achievable_cc_iou(hmap, prior_cfg, comps, orig):
    """Ceiling IoU over a threshold scan (fraction of the map's own range)."""
    best = 0.0
    for frac in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        cfg = dict(prior_cfg)
        cfg['relative_threshold'] = frac
        boxes_1000, _ = _cc_boxes_1000(hmap, cfg, 999)
        pred_px = [to_pixels(b, orig) for b in boxes_1000]
        best = max(best, _best_iou_px(pred_px, comps))
    return best


def metrics_for_map(hmap, prior_cfg, comps, orig, hw):
    boxes_1000, proposals = _cc_boxes_1000(hmap, prior_cfg, prior_cfg.get('max_candidates', 3))
    pred_px = [to_pixels(b, orig) for b in boxes_1000]
    return dict(
        n_prop=len(boxes_1000),
        recall_01=_recall_at(pred_px, comps, 0.1),
        recall_03=_recall_at(pred_px, comps, 0.3),
        best_iou=_best_iou_px(pred_px, comps),
        peak_in_gt=1.0 if _peak_in_gt(proposals, comps, orig) else 0.0,
        boundary_cov=_boundary_coverage(proposals, comps, orig, hw),
        best_cc_iou=_best_achievable_cc_iou(hmap, prior_cfg, comps, orig),
    )


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/qwen35_2b_annos_probe_v2_fsm_rl.yaml')
    ap.add_argument('--sft-adapter', default='outputs/train/region_sft_annos_think_fsm')
    ap.add_argument('--split', choices=['dev', 'test'], default='test')
    ap.add_argument('--limit', type=int, default=0, help='max small samples (0=all)')
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    cfg.setdefault('outcome', {})['sft_adapter'] = args.sft_adapter
    oc = cfg['outcome']
    oc.setdefault('prior', {})['h_image_size'] = 768
    oc.setdefault('zoom', {})['enabled'] = False

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    set_seed(int(cfg['training']['seed']))

    model, processor, prior = load_model(cfg, None, fresh_lora=False)
    model.eval()

    from outcome.inputs_multibox import OutcomeMultiboxCollator
    collator = OutcomeMultiboxCollator(processor, prior, cfg)
    prior_cfg = dict(cfg['outcome'].get('prior', {}))
    h_size = int(prior_cfg.get('h_image_size', 768))

    _, _, test_set = datasets(cfg, processor)

    # collect small anomaly samples
    samples = []
    for i in range(len(test_set)):
        item = test_set[i]
        if not item.get('is_anomaly'):
            continue
        comps = list(item.get('component_bboxes') or [])
        if not comps and item.get('gt_box_px'):
            comps = [item['gt_box_px']]
        if not comps:
            continue
        frac = item.get('mask_area_fraction')
        if frac is None:
            gt = item['gt_box_px']
            w, h = item['orig_size']
            frac = (gt[2]-gt[0])*(gt[3]-gt[1])/(w*h) if gt else 0.0
        if frac < SMALL_AREA_FRAC:
            samples.append((i, item))
    samples.sort(key=lambda x: x[1]['image_path'])
    if args.limit > 0:
        samples = samples[:args.limit]
    print(f'[diag] small anomaly samples: {len(samples)} (limit={args.limit})', flush=True)

    rep_names = ['block12', 'block16', 'block20', 'block24', 'fused_H', 'merged']
    agg = {r: dict(n_prop=[], recall_01=[], recall_03=[], best_iou=[],
                   peak_in_gt=[], boundary_cov=[], best_cc_iou=[]) for r in rep_names}

    for si, (idx, item) in enumerate(samples):
        with pixel_budget(processor, h_size):
            ref_h, test_h = collator._align_pair_at(item['ref'], item['test'], h_size)
            h_in = collator._concat_image_tensors(ref_h, test_h)
        pv = h_in['pixel_values'].to(device)
        grid = h_in['image_grid_thw'].to(device)
        counts = [int(v) for v in grid.prod(-1)]
        ref_pv, test_pv = torch.split(pv, counts, dim=0)
        ref_grid, test_grid = grid[0:1], grid[1:2]

        with torch.no_grad():
            rep = extract_all(prior, ref_pv, ref_grid, test_pv, test_grid)

        if si == 0:
            print(f'[shapes] hw_t={rep["hw_t"]} hw_r={rep["hw_r"]} '
                  f'mhw_t={rep["mhw_t"]} block_indices={rep["block_indices"]} '
                  f'L={len(rep["maps"])} fused={tuple(rep["fused"].shape)} '
                  f'merged={tuple(rep["s_merged"].shape)}', flush=True)

        comps = list(item.get('component_bboxes') or [])
        if not comps and item.get('gt_box_px'):
            comps = [item['gt_box_px']]
        orig = item['orig_size']  # (w, h)

        name_map = {}
        for name, smap in zip(rep_names[:4], rep['maps']):
            name_map[name] = smap
        name_map['fused_H'] = rep['fused']
        name_map['merged'] = rep['s_merged']

        for name in rep_names:
            hw = rep['mhw_t'] if name == 'merged' else rep['hw_t']
            m = metrics_for_map(name_map[name], prior_cfg, comps, orig, hw)
            for k, v in m.items():
                agg[name][k].append(v)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float('nan')

    print('\n' + '=' * 96, flush=True)
    cols = ['recall@.1', 'recall@.3', 'bestIoU', 'peakInGT', 'bndCov', 'bestCC', 'nProp']
    print(f'{"rep":<10}' + ''.join(f'{c:>10}' for c in cols), flush=True)
    print('-' * 96, flush=True)
    for name in rep_names:
        a = agg[name]
        row = (f'{name:<10}'
               f'{mean(a["recall_01"]):>10.3f}'
               f'{mean(a["recall_03"]):>10.3f}'
               f'{mean(a["best_iou"]):>10.3f}'
               f'{mean(a["peak_in_gt"]):>10.3f}'
               f'{mean(a["boundary_cov"]):>10.3f}'
               f'{mean(a["best_cc_iou"]):>10.3f}'
               f'{mean(a["n_prop"]):>10.2f}')
        print(row, flush=True)
    print('=' * 96, flush=True)

    out = {name: {k: mean(v) for k, v in a.items()} for name, a in agg.items()}
    out['n_samples'] = len(samples)
    outpath = Path('outputs/eval/rep_resolution.json')
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(out, indent=2))
    print(f'\nsaved -> {outpath}', flush=True)


if __name__ == '__main__':
    main()
