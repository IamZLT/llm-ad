#!/usr/bin/env python3
"""Layer-2 tiny-probe: which representation can a frozen-vision, tiny head decode
the defect mask from?

Layer-1 measured how well each representation's *distance map* localizes. Layer-2
asks the sharper question: given the RAW features, does a learned shallow head
(1x1 conv, fixed 256-wide trunk) recover the GT localization? This isolates
"representation content" from "distance-map quality".

Inputs (per cell p, plus 2-D positional encoding PE):

    P_H      : [ H(p) ]                                    (scalar H, 48x48)
    P_M      : [ v_t, v_r, v_t-v_r ]                       (merged, 24x24)
    P_F@l    : [ f_t, f_r^match, f_t-f_r^match, H ]        (dense pre-merger layer l)
    P_R@l    : P_F@l but features gated to H-proposal cells (current sparse access)

Head (same capacity for every input): 1x1 conv C->256->GELU->256->GELU->1.

GT = GT component box rasterized to grid cells. Trained with pos-weighted BCE on
train-split anomalies; evaluated on test-split small anomalies with the SAME
small-defect localization metrics as Layer-1 (best IoU / recall / best-CC ceiling),
so the two layers are directly comparable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import load_yaml_config
from utils.common import set_seed
from models.anomaly_prior import softmax_fuse_maps
from outcome.inputs import pixel_budget, region_proposals
from outcome.engine_multibox import datasets, load_model
from outcome.protocol import iou, to_pixels

from scripts.diagnose_rep_resolution import metrics_for_map

SMALL_AREA_FRAC = 0.02
HIDDEN = 256


# --------------------------------------------------------------------------- #
# representation extraction (raw features, not just distance maps)
# --------------------------------------------------------------------------- #
def extract_raw(prior, ref_pv, ref_grid, test_pv, test_grid):
    f_r, hw_r, v_r = prior._encode_one(ref_pv, ref_grid)
    f_t, hw_t, v_t = prior._encode_one(test_pv, test_grid)
    maps, matched_r = [], []
    for ft, fr in zip(f_t, f_r):
        m = prior._nn_match(ft, fr, hw_t, hw_r, prior.neighborhood_radius)
        maps.append(m['distance'])
        matched_r.append(m['matched_ref_features'])
    stack = torch.stack(maps)
    H = softmax_fuse_maps(stack, prior.temperature) if len(maps) > 1 else stack[0]
    m = int(prior.spatial_merge_size)
    mhw_t = (int(hw_t[0]) // m, int(hw_t[1]) // m)
    mhw_r = (int(hw_r[0]) // m, int(hw_r[1]) // m)
    return dict(f_t=f_t, matched_r=matched_r, v_t=v_t, v_r=v_r, H=H,
                hw_t=hw_t, hw_r=hw_r, mhw_t=mhw_t, mhw_r=mhw_r,
                block_indices=list(prior.block_indices))


def _pe2d(H, W, device):
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([gx, gy], dim=-1).float()  # [H, W, 2]


def _gt_mask(comps_px, orig, H, W):
    """GT component boxes (px) -> [H, W] binary cell mask."""
    mask = np.zeros((H, W), dtype=np.float32)
    for g in comps_px:
        gx0 = int(np.floor(g[0] / orig[0] * W)); gx1 = int(np.ceil(g[2] / orig[0] * W))
        gy0 = int(np.floor(g[1] / orig[1] * H)); gy1 = int(np.ceil(g[3] / orig[1] * H))
        gx0 = max(0, min(W, gx0)); gx1 = max(0, min(W, gx1))
        gy0 = max(0, min(H, gy0)); gy1 = max(0, min(H, gy1))
        if gx1 > gx0 and gy1 > gy0:
            mask[gy0:gy1, gx0:gx1] = 1.0
    return mask


def _proposal_gate(Hmap, prior_cfg, H, W):
    """Union mask of the top-K CC proposal cells (the 'H-selected' cells)."""
    gate = np.zeros((H, W), dtype=np.float32)
    try:
        proposals, masks, _ = region_proposals(Hmap, dict(prior_cfg))
    except Exception:
        return gate
    if masks is not None and len(masks):
        gate = masks.astype(np.float32).sum(axis=0).clip(0, 1)
    return gate


# --------------------------------------------------------------------------- #
# build [1, C, H, W] input for each representation
# --------------------------------------------------------------------------- #
def build_inputs(raw, comps_px, orig, prior_cfg, device):
    Ht, Wt = int(raw['hw_t'][0]), int(raw['hw_t'][1])
    pe = _pe2d(Ht, Wt, device)
    Hmap = raw['H'].float()
    Hc = Hmap.reshape(Ht, Wt, 1)
    bi = list(raw['block_indices'])
    out = {}

    # P_H
    out['P_H'] = (torch.cat([Hc, pe], dim=-1).permute(2, 0, 1).unsqueeze(0).to(device),
                  (Ht, Wt))

    # P_F per layer (dense)
    gate = _proposal_gate(Hmap, prior_cfg, Ht, Wt)
    gate_t = torch.from_numpy(gate).to(device)
    for i, idx in enumerate(bi):
        layer_name = f'L{idx + 1}'
        ft = raw['f_t'][i].float().reshape(Ht, Wt, -1)
        mr = raw['matched_r'][i].float().reshape(Ht, Wt, -1)
        dense = torch.cat([ft, mr, ft - mr, Hc], dim=-1)
        out[f'P_F_{layer_name}'] = (torch.cat([dense, pe], dim=-1).permute(2, 0, 1).unsqueeze(0).to(device),
                                    (Ht, Wt))
        # P_R (sparse-gated): gate only the feature channels, keep H + PE
        g = gate_t.reshape(Ht, Wt, 1)
        sparse = torch.cat([ft * g, mr * g, (ft - mr) * g, Hc], dim=-1)
        out[f'P_R_{layer_name}'] = (torch.cat([sparse, pe], dim=-1).permute(2, 0, 1).unsqueeze(0).to(device),
                                    (Ht, Wt))

    # P_M (merged) -- merged grid may be non-square; use mhw_t
    mh, mw = int(raw['mhw_t'][0]), int(raw['mhw_t'][1])
    v_t = raw['v_t'].float().reshape(mh, mw, -1)
    v_r = raw['v_r'].float().reshape(mh, mw, -1)
    pe_m = _pe2d(mh, mw, device)
    merged = torch.cat([v_t, v_r, v_t - v_r, pe_m], dim=-1)
    out['P_M'] = (merged.permute(2, 0, 1).unsqueeze(0).to(device), (mh, mw))

    return out


# --------------------------------------------------------------------------- #
class TinyProbe(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, HIDDEN, 1), nn.GELU(),
            nn.Conv2d(HIDDEN, HIDDEN, 1), nn.GELU(),
            nn.Conv2d(HIDDEN, 1, 1),
        )

    def forward(self, x):
        return self.net(x)  # [B, 1, H, W]


def train_probe(probe, items, epochs, device, seed):
    probe = probe.to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    total_pos = sum(float(m.sum()) for _, m in items)
    total_neg = sum(float(m.size - m.sum()) for _, m in items)
    pos_weight = torch.tensor((total_neg / (total_pos + 1e-6))).clamp(max=50.0).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    torch.manual_seed(seed)
    # per-sample SGD: grids can differ across samples (non-square crops)
    for _ in range(epochs):
        for inp, m in items:
            y = torch.from_numpy(m).unsqueeze(0).unsqueeze(0).to(device)
            opt.zero_grad()
            logits = probe(inp)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
    return probe.eval()


def _cell_auroc(score, mask):
    s = score.ravel().astype(float)
    m = mask.ravel().astype(bool)
    npos = int(m.sum()); nneg = int((~m).sum())
    if npos == 0 or nneg == 0:
        return float('nan')
    order = np.argsort(s, kind='stable')
    m = m[order].astype(int)
    tps = np.cumsum(m)  # cumulative positives (0-indexed sorted)
    # rank-sum AUC: (sum of positive ranks - npos(npos+1)/2) / (npos*nneg)
    return float((tps.sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/qwen35_2b_annos_probe_v2_fsm_rl.yaml')
    ap.add_argument('--sft-adapter', default='outputs/train/region_sft_annos_think_fsm')
    ap.add_argument('--train-limit', type=int, default=120, help='train anomalies for probe fit')
    ap.add_argument('--test-limit', type=int, default=0, help='test small anomalies (0=all)')
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--gpu', type=int, default=1)
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    cfg.setdefault('outcome', {})['sft_adapter'] = args.sft_adapter
    oc = cfg['outcome']
    oc.setdefault('prior', {})['h_image_size'] = 768
    oc.setdefault('zoom', {})['enabled'] = False

    device = torch.device(f'cuda:{args.gpu}')
    # load_model pins the model to LOCAL_RANK; make it agree with --gpu.
    os.environ['LOCAL_RANK'] = str(args.gpu)
    os.environ['RANK'] = str(args.gpu)
    torch.cuda.set_device(device)
    set_seed(int(cfg['training']['seed']))

    model, processor, prior = load_model(cfg, None, fresh_lora=False)
    model.eval()
    from outcome.inputs_multibox import OutcomeMultiboxCollator
    collator = OutcomeMultiboxCollator(processor, prior, cfg)
    prior_cfg = dict(cfg['outcome'].get('prior', {}))
    h_size = int(prior_cfg.get('h_image_size', 768))

    train_set, _, test_set = datasets(cfg, processor)

    # select indices via metadata (no image load)
    train_anom_idx = [i for i, s in enumerate(train_set.samples)
                      if (s.get('metadata') or {}).get('anomaly')]
    test_small_idx = [i for i, s in enumerate(test_set.samples)
                      if (s.get('metadata') or {}).get('anomaly')
                      and (s.get('metadata') or {}).get('mask_area_fraction', 1.0) < SMALL_AREA_FRAC]
    train_anom_idx = train_anom_idx[:args.train_limit]
    if args.test_limit > 0:
        test_small_idx = test_small_idx[:args.test_limit]
    print(f'[probe] train anomalies={len(train_anom_idx)} test small={len(test_small_idx)}', flush=True)

    def _load_and_extract(idx, ds):
        item = ds[idx]
        with pixel_budget(processor, h_size):
            ref_h, test_h = collator._align_pair_at(item['ref'], item['test'], h_size)
            h_in = collator._concat_image_tensors(ref_h, test_h)
        pv = h_in['pixel_values'].to(device); grid = h_in['image_grid_thw'].to(device)
        counts = [int(v) for v in grid.prod(-1)]
        ref_pv, test_pv = torch.split(pv, counts, dim=0)
        with torch.no_grad():
            raw = extract_raw(prior, ref_pv, grid[0:1], test_pv, grid[1:2])
        comps = list(item.get('component_bboxes') or [])
        if not comps and item.get('gt_box_px'):
            comps = [item['gt_box_px']]
        return raw, comps, item['orig_size']

    # ---- train features ----
    train_in = {}
    print('[probe] extracting train features...', flush=True)
    for t, idx in enumerate(train_anom_idx):
        raw, comps, orig = _load_and_extract(idx, train_set)
        inputs = build_inputs(raw, comps, orig, prior_cfg, device)
        for name, (inp, (H, W)) in inputs.items():
            mask = _gt_mask(comps, orig, H, W)
            train_in.setdefault(name, []).append((inp, mask))
        if (t + 1) % 40 == 0:
            print(f'  train {t+1}/{len(train_anom_idx)}', flush=True)

    # ---- train probes ----
    print('[probe] training probes...', flush=True)
    probes = {}
    for name, items in train_in.items():
        in_ch = items[0][0].shape[1]
        probe = TinyProbe(in_ch)
        train_probe(probe, items, args.epochs, device, int(cfg['training']['seed']))
        probes[name] = probe
        print(f'  {name}: in_ch={in_ch} trained', flush=True)

    # ---- evaluate on test small ----
    print('[probe] evaluating on test small...', flush=True)
    agg = {}
    for si, idx in enumerate(test_small_idx):
        raw, comps, orig = _load_and_extract(idx, test_set)
        inputs = build_inputs(raw, comps, orig, prior_cfg, device)
        Ht, Wt = raw['hw_t']
        for name, (inp, (H, W)) in inputs.items():
            probe = probes[name]
            with torch.no_grad():
                logits = probe(inp)
            score = torch.sigmoid(logits[0, 0]).float().cpu().numpy()  # [H, W]
            mask = _gt_mask(comps, orig, H, W)
            m = metrics_for_map(torch.from_numpy(score), prior_cfg, comps, orig, (H, W))
            m['auroc'] = _cell_auroc(score, mask)
            agg.setdefault(name, {}).setdefault('rows', []).append(m)
        if (si + 1) % 100 == 0:
            print(f'  eval {si+1}/{len(test_small_idx)}', flush=True)

    def mean(xs):
        xs = [x for x in xs if x is not None and x == x]
        return sum(xs) / len(xs) if xs else float('nan')

    names = sorted(agg, key=lambda n: (n.split('_')[0], n))
    KEYMAP = {'recall@.1': 'recall_01', 'recall@.3': 'recall_03', 'bestIoU': 'best_iou',
              'peakInGT': 'peak_in_gt', 'bndCov': 'boundary_cov', 'bestCC': 'best_cc_iou'}
    cols = list(KEYMAP) + ['auroc']
    print('\n' + '=' * 100, flush=True)
    print(f'{"rep":<12}' + ''.join(f'{c:>10}' for c in cols), flush=True)
    print('-' * 100, flush=True)
    summary = {}
    for name in names:
        rows = agg[name]['rows']
        a = {c: mean([r[KEYMAP[c]] for r in rows]) for c in KEYMAP}
        a['auroc'] = mean([r['auroc'] for r in rows])
        summary[name] = a
        print(f'{name:<12}'
              f'{a["recall@.1"]:>10.3f}{a["recall@.3"]:>10.3f}{a["bestIoU"]:>10.3f}'
              f'{a["peakInGT"]:>10.3f}{a["bndCov"]:>10.3f}{a["bestCC"]:>10.3f}'
              f'{a["auroc"]:>10.3f}', flush=True)
    print('=' * 100, flush=True)

    outpath = Path('outputs/eval/tiny_probe.json')
    outpath.write_text(json.dumps({'summary': summary, 'n_train': len(train_anom_idx),
                                   'n_test': len(test_small_idx)}, indent=2))
    print(f'\nsaved -> {outpath}', flush=True)


if __name__ == '__main__':
    main()
