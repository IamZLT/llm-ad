#!/usr/bin/env python3
"""Evaluate the Zoom-SFT v1 checkpoint (H@768, top-1 paired same-coordinate crop).

Loads the Zoom-v1 LoRA+region adapter and runs the FSM staged decoder with zoom
enabled, reporting the four core metrics for the small-defect decision:
    Recall_small, mIoU_small|TP, FPR, BalancedAcc
against the frozen H@768 baseline (region_sft_annos_think_fsm, zoom off) already
recorded in outputs/eval/zoom_ablation/summary_table.json (experiment B_h768).

Parallelization matches scripts/run_zoom_ablation.py: shard samples round-robin
across --gpus x --procs-per-gpu worker processes, merge the jsonl rows, summarize.
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import load_yaml_config

# Zoom-v1 input: must match the training config exactly (H@768 + top-1 paired
# same-coordinate crop). reference_mode='same' (not matched) is the v1 variable.
ZOOM_V1 = {
    'prior': {'h_image_size': 768},
    'zoom': {'enabled': True, 'source': 'peaks', 'max_crops': 1, 'nms_radius': 2,
             'crop_fraction': 0.20, 'include_reference': True, 'reference_mode': 'same'},
}

CORE_KEYS = [
    'n', 'n_anomaly', 'n_normal', 'n_small',
    'anomaly_recall_small', 'anomaly_recall',
    'mask_miou_small_given_tp', 'mask_miou_small', 'matched_miou_small_given_tp',
    'normal_fpr', 'balanced_accuracy', 'task_valid_rate', 'truncation_rate',
]


def apply_overrides(cfg, overrides):
    cfg = copy.deepcopy(cfg)
    oc = cfg.setdefault('outcome', {})
    if 'prior' in overrides:
        for k, v in overrides['prior'].items():
            oc.setdefault('prior', {})[k] = v
    if 'zoom' in overrides:
        for k, v in overrides['zoom'].items():
            oc.setdefault('zoom', {})[k] = v
    return cfg


def _worker(rank, gpu, base_cfg, split, limit, out_dir, num_total):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    import torch

    from outcome.engine_multibox import datasets, load_model
    from outcome.evaluate_multibox import evaluate, stratified_eval_indices
    from utils.common import set_seed

    out_dir = Path(out_dir)
    sys.stdout = open(out_dir / f'worker{rank}.log', 'a', buffering=1)
    sys.stderr = sys.stdout

    set_seed(int(base_cfg['training']['seed']))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'[worker{rank} gpu{gpu}] device={device}', flush=True)
    model, processor, prior = load_model(base_cfg, None, fresh_lora=False)
    model.eval()

    _, _, test_set = datasets(base_cfg, processor)
    all_idx = stratified_eval_indices(test_set, limit, seed=int(base_cfg['training']['seed']))
    mine = [all_idx[i] for i in range(rank, len(all_idx), num_total)]

    cfg = apply_overrides(base_cfg, ZOOM_V1)
    path = out_dir / f'zoom_v1.w{rank}.json'
    t0 = time.time()
    print(f'[worker{rank} gpu{gpu}] zoom_v1 ({len(mine)} samples)', flush=True)
    evaluate(cfg, model, processor, prior, test_set, path,
             indices=mine, writer=None, namespace=f'zoom_v1.w{rank}')
    print(f'[worker{rank} gpu{gpu}] done in {(time.time()-t0)/60:.1f}m', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/qwen35_2b_annos_probe_v2_fsm_rl.yaml')
    ap.add_argument('--sft-adapter', default='outputs/train/region_sft_annos_think_fsm_zoom_v1')
    ap.add_argument('--split', choices=['dev', 'test'], default='test')
    ap.add_argument('--limit', type=int, default=200)
    ap.add_argument('--gpus', default='0,1')
    ap.add_argument('--procs-per-gpu', type=int, default=2)
    ap.add_argument('--out-dir', default='outputs/eval/zoom_sft_v1')
    args = ap.parse_args()

    base = load_yaml_config(args.config)
    base.setdefault('outcome', {})['sft_adapter'] = args.sft_adapter

    gpus = [int(x) for x in args.gpus.split(',') if x.strip()]
    ppg = max(1, int(args.procs_per_gpu))
    num_total = len(gpus) * ppg
    rank_gpu = [(r, gpus[r // ppg]) for r in range(num_total)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'base_config.json').write_text(json.dumps(base, ensure_ascii=False, indent=2))

    ctx = mp.get_context('spawn')
    procs = []
    for rank, gpu in rank_gpu:
        p = ctx.Process(target=_worker, args=(rank, gpu, base, args.split, args.limit, str(out_dir), num_total))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    from outcome.evaluate_multibox import summarize

    rows = []
    for rank in range(num_total):
        p = out_dir / f'zoom_v1.w{rank}.jsonl'
        if p.exists():
            rows.extend(json.loads(l) for l in p.read_text().splitlines() if l.strip())
    rows.sort(key=lambda r: r['image_path'])
    (out_dir / 'zoom_v1.jsonl').write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False, default=str) for r in rows) + '\n')
    stats = summarize(rows)
    (out_dir / 'zoom_v1.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2))

    baseline = json.loads(Path('outputs/eval/zoom_ablation/summary_table.json').read_text())['B_h768']
    print('\n' + '=' * 78, flush=True)
    print(f'{"metric":<28}{"B_h768 (base)":>16}{"Zoom-v1":>16}{"delta":>12}', flush=True)
    print('-' * 78, flush=True)
    for k in CORE_KEYS:
        b = baseline.get(k)
        z = stats.get(k)
        if isinstance(b, (int, float)) and isinstance(z, (int, float)):
            d = z - b
            print(f'{k:<28}{b:>16.4f}{z:>16.4f}{d:>+12.4f}', flush=True)
        else:
            print(f'{k:<28}{str(b):>16}{str(z):>16}{"":>12}', flush=True)
    print('=' * 78, flush=True)
    print(f'saved {out_dir}/zoom_v1.json  and  {out_dir}/zoom_v1.jsonl', flush=True)


if __name__ == '__main__':
    main()
