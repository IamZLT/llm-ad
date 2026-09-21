#!/usr/bin/env python3
"""Causal ablation for the H-aware Adaptive Evidence Router.

The router checkpoint is evaluated three times under identical model weights, changing
ONLY ``outcome.region.router.condition``:

    real      router H = the actual H map
    shuffled  router H = deterministic spatial permutation of H
    flat      router H = constant (mean) -> no spatial prior

Proposals ALWAYS come from the real H (prior.condition=real), so the only thing that
varies is H's *second-stage routing* role (confidence, context radius, core threshold,
token budget). Success requires:

    real > shuffled ~= flat

on mask_mIoU_small|TP / P(C|small,TP) / H-good degradation / H-poor recovery, and not
just an FPR-only improvement (which would again be "more conservative", not "uses H").
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import load_yaml_config

CONDITIONS = ['real', 'shuffled', 'flat']

TABLE_KEYS = [
    'n', 'n_anomaly', 'n_normal', 'n_small',
    'anomaly_recall_small', 'anomaly_recall',
    'mask_miou_small_given_tp', 'mask_miou_small',
    'matched_miou_small_given_tp',
    'normal_fpr', 'balanced_accuracy', 'task_valid_rate', 'truncation_rate',
]


def build_eval_cfg(base_cfg, sft_adapter, region_feature_index, h_size, cond):
    cfg = copy.deepcopy(base_cfg)
    oc = cfg.setdefault('outcome', {})
    oc['sft_adapter'] = sft_adapter
    p = oc.setdefault('prior', {})
    p['region_feature_index'] = int(region_feature_index)
    p['h_image_size'] = int(h_size)
    oc['region'] = dict(
        intermediate_dim=256, geometry_dim=5, hstat_dim=2, activation='gelu',
        max_cells=15, token_mode='routed', use_hstat=False, num_roles=3,
        router=dict(enabled=True, condition=cond, core_fraction=0.5,
                    context_radius_min=1, context_radius_max=4,
                    min_tokens_per_candidate=3, expose_peak_text=False),
    )
    oc.setdefault('zoom', {})['enabled'] = False
    oc['h_prior_adapter'] = {'enabled': False}
    oc['h_memory'] = {'enabled': False}
    cfg.setdefault('grpo', {})['staged_rollout'] = True
    return cfg


def _worker(rank, gpu, base_cfg, sft_adapter, region_feature_index, h_size,
            cond, split, limit, out_dir, num_total):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    import torch

    from outcome.engine_multibox import datasets, load_model
    from outcome.evaluate_multibox import evaluate, stratified_eval_indices
    from utils.common import set_seed

    cond_dir = Path(out_dir) / cond
    log_path = cond_dir / f'worker{rank}.log'
    cond_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = open(log_path, 'a', buffering=1)
    sys.stderr = sys.stdout

    cfg = build_eval_cfg(base_cfg, sft_adapter, region_feature_index, h_size, cond)
    set_seed(int(cfg['training']['seed']))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'[worker{rank} gpu{gpu} cond={cond}] device={device}', flush=True)
    model, processor, prior = load_model(cfg, None, fresh_lora=False)
    model.eval()

    _, dev_set, test_set = datasets(cfg, processor)
    sel = test_set if split == 'test' else dev_set
    all_idx = stratified_eval_indices(sel, limit, seed=int(cfg['training']['seed']))
    mine = [all_idx[i] for i in range(rank, len(all_idx), num_total)]
    path = cond_dir / f'rows.w{rank}.json'
    print(f'[worker{rank} cond={cond}] {len(mine)} samples', flush=True)
    evaluate(cfg, model, processor, prior, sel, path, indices=mine,
             writer=None, namespace=f'w{rank}')
    print(f'[worker{rank} cond={cond}] done', flush=True)


def _merge(out_dir, cond, num_total):
    from outcome.evaluate_multibox import summarize

    cond_dir = Path(out_dir) / cond
    rows = []
    for rank in range(num_total):
        p = cond_dir / f'rows.w{rank}.jsonl'
        if p.exists():
            rows.extend(json.loads(l) for l in p.read_text().splitlines() if l.strip())
    rows.sort(key=lambda r: r['image_path'])
    (cond_dir / 'rows.jsonl').write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False, default=str) for r in rows) + '\n')
    stats = summarize(rows)
    (cond_dir / 'summary.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats, rows


def mechanism_kpis(rows):
    """H-good/mid/poor bucketed localization deltas + P(C|small,TP)."""
    from PIL import Image
    from outcome.protocol import iou, to_pixels

    DELTA = 0.05
    small_tp = [r for r in rows if r['is_anomaly'] and r['pred'] is True
                and r['size_bin'] == 'small']
    data = []
    for r in small_tp:
        gt = r.get('gt_box_px')
        orig = Image.open(r['image_path']).size
        h_cands = [c.get('bbox_2d') for c in r.get('prior_candidates', [])]
        vals = [iou(to_pixels(b, orig), gt) for b in h_cands if b and len(b) == 4]
        h_iou = max(vals) if vals else None
        f_vals = [iou(to_pixels(b, orig), gt) for b in r.get('bboxes_2d', [])
                  if b and len(b) == 4]
        f_iou = max(f_vals) if f_vals else None
        if h_iou is None or f_iou is None:
            continue
        dh = f_iou - h_iou
        cat = 'C_degraded' if dh < -DELTA else ('A_improved' if dh > DELTA else 'B_unchanged')
        data.append(dict(h=h_iou, f=f_iou, dh=dh, cat=cat))

    n = len(data)
    out = dict(n_small_tp=n)
    if n:
        out['P_C_given_small_tp'] = sum(1 for d in data if d['cat'] == 'C_degraded') / n
        out['mean_final_minus_h'] = sum(d['dh'] for d in data) / n
        good = [d for d in data if d['h'] >= 0.4]
        mid = [d for d in data if 0.2 <= d['h'] < 0.4]
        poor = [d for d in data if d['h'] < 0.2]
        out['n_good'] = len(good)
        out['n_mid'] = len(mid)
        out['n_poor'] = len(poor)
        out['P_degrade_good'] = (sum(1 for d in good if d['cat'] == 'C_degraded') / len(good)
                                 if good else float('nan'))
        out['H_poor_recovery'] = (sum(1 for d in poor if d['dh'] > 0.1) / len(poor)
                                  if poor else float('nan'))
        out['mean_dh_good'] = (sum(d['dh'] for d in good) / len(good)) if good else float('nan')
        out['mean_dh_mid'] = (sum(d['dh'] for d in mid) / len(mid)) if mid else float('nan')
        out['mean_dh_poor'] = (sum(d['dh'] for d in poor) / len(poor)) if poor else float('nan')
    return out


def router_behavior(rows):
    """Aggregate router_meta (confidence/radius/token counts) over all samples."""
    confs, radius, n_core, n_extent, n_context, n_budget = [], [], [], [], [], []
    n_with = 0
    for r in rows:
        for m in (r.get('router_meta') or []):
            confs.append(m['confidence']); radius.append(m['radius'])
            n_core.append(m['n_core']); n_extent.append(m['n_extent'])
            n_context.append(m['n_context']); n_budget.append(m['budget'])
            n_with += 1
    def mean(xs):
        return sum(xs) / len(xs) if xs else float('nan')
    return dict(n_candidates=n_with, mean_confidence=mean(confs),
                mean_radius=mean(radius), mean_n_core=mean(n_core),
                mean_n_extent=mean(n_extent), mean_n_context=mean(n_context),
                mean_budget=mean(n_budget))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/qwen35_2b_annos_probe_v2_fsm_rl.yaml')
    ap.add_argument('--router-checkpoint', required=True)
    ap.add_argument('--region-feature-index', type=int, default=23)
    ap.add_argument('--h-size', type=int, default=768)
    ap.add_argument('--split', choices=['dev', 'test'], default='test')
    ap.add_argument('--limit', type=int, default=200)
    ap.add_argument('--gpus', default='0,1')
    ap.add_argument('--procs-per-gpu', type=int, default=1)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    base = load_yaml_config(args.config)
    gpus = [int(x) for x in args.gpus.split(',') if x.strip()]
    ppg = max(1, int(args.procs_per_gpu))
    num_total = len(gpus) * ppg
    rank_gpu = [(r, gpus[r // ppg]) for r in range(num_total)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'ablation_config.json').write_text(json.dumps(
        build_eval_cfg(base, args.router_checkpoint, args.region_feature_index,
                       args.h_size, 'real'), ensure_ascii=False, indent=2))

    ctx = mp.get_context('spawn')
    results = {}
    for cond in CONDITIONS:
        print(f'\n===== condition = {cond} =====', flush=True)
        procs = []
        for rank, gpu in rank_gpu:
            p = ctx.Process(target=_worker,
                            args=(rank, gpu, base, args.router_checkpoint,
                                  args.region_feature_index, args.h_size, cond,
                                  args.split, args.limit, str(out_dir), num_total))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        stats, rows = _merge(out_dir, cond, num_total)
        kpis = mechanism_kpis(rows)
        results[cond] = dict(stats=stats, kpis=kpis, behavior=router_behavior(rows))

    # ---- comparison table ----
    print('\n' + '=' * 110)
    print('ROUTER CAUSAL ABLATION')
    print('=' * 110)
    hdr = f'{"metric":32s}' + ''.join(f'{c:>22s}' for c in CONDITIONS)
    print(hdr)
    print('-' * 110)
    for k in TABLE_KEYS:
        row = f'{k:32s}'
        for c in CONDITIONS:
            v = results[c]['stats'].get(k)
            row += f'{("--" if v is None else round(v, 4)):>22}'
        print(row)
    print('-' * 110)
    for k in ['P_C_given_small_tp', 'P_degrade_good', 'H_poor_recovery',
              'mean_final_minus_h', 'mean_dh_good', 'mean_dh_mid', 'mean_dh_poor',
              'n_small_tp', 'n_good', 'n_mid', 'n_poor']:
        row = f'[kpi] {k:28s}'
        for c in CONDITIONS:
            v = results[c]['kpis'].get(k)
            row += f'{("--" if v is None else round(v, 4)):>22}'
        print(row)
    print('-' * 110)
    for k in ['n_candidates', 'mean_confidence', 'mean_radius', 'mean_budget',
              'mean_n_core', 'mean_n_extent', 'mean_n_context']:
        row = f'[router] {k:27s}'
        for c in CONDITIONS:
            v = results[c]['behavior'].get(k)
            row += f'{("--" if v is None else round(v, 4)):>22}'
        print(row)
    print('=' * 110)

    # ---- causal verdict ----
    real = results['real']['stats']; shuf = results['shuffled']['stats']
    flat = results['flat']['stats']
    m = results['real']['kpis'].get('P_C_given_small_tp')
    real_miou = real.get('mask_miou_small_given_tp')
    shuf_miou = shuf.get('mask_miou_small_given_tp')
    flat_miou = flat.get('mask_miou_small_given_tp')
    real_fpr = real.get('normal_fpr'); shuf_fpr = shuf.get('normal_fpr'); flat_fpr = flat.get('normal_fpr')
    print('\nCAUSAL VERDICT', flush=True)
    if real_miou is not None:
        mIoU_gap = real_miou - max(shuf_miou or 0.0, flat_miou or 0.0)
        print(f'  mask_mIoU_small|TP : real={real_miou:.4f} shuffled={shuf_miou:.4f} '
              f'flat={flat_miou:.4f}  (real - max(shuffled,flat) = {mIoU_gap:+.4f})')
    if m is not None:
        print(f'  P(C|small,TP)      : real={m:.4f}  shuffled={results["shuffled"]["kpis"].get("P_C_given_small_tp")}  flat={results["flat"]["kpis"].get("P_C_given_small_tp")}')
    if real_fpr is not None:
        print(f'  normal_fpr         : real={real_fpr:.4f} shuffled={shuf_fpr:.4f} flat={flat_fpr:.4f}')
    # Verdict rule: real must beat shuffled & flat on localization (mIoU|TP) by a margin,
    # and shuffled ~= flat (H spatial structure is actually consumed).
    verdict = 'INCONCLUSIVE'
    if real_miou is not None and shuf_miou is not None and flat_miou is not None:
        uses_structure = (real_miou - max(shuf_miou, flat_miou)) > 0.02
        shuffled_flat_close = abs((shuf_miou or 0.0) - (flat_miou or 0.0)) < 0.03
        if uses_structure and shuffled_flat_close:
            verdict = 'PASS: real > shuffled ~= flat -> router consumes H spatial structure'
        elif uses_structure and not shuffled_flat_close:
            verdict = 'PARTIAL: real > shuffled/flat but shuffled != flat (unexpected)'
        else:
            verdict = 'FAIL: router does not consume H spatial structure (real ~ shuffled ~ flat)'
    print(f'  VERDICT            : {verdict}')
    print('=' * 110)
    print(f'\nsaved under {out_dir}/{{real,shuffled,flat}}/ (rows.jsonl + summary.json)')


if __name__ == '__main__':
    main()
