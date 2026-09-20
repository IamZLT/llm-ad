#!/usr/bin/env python3
"""No-training zoom ablation: locate the small-defect bottleneck.

Runs the frozen FSM checkpoint under five input conditions and reports the
small-defect diagnostic chain:

    H recall -> FSM recall -> mIoU given TP -> normal FPR

Experiments (H resolution is decoupled from crop zoom via `prior.h_image_size`):
    A  H@448, no crop                     (current baseline)
    B  H@768, no crop                     (isolate H proposal resolution)
    C  H@768, peak test crop              (local visual magnification)
    D  H@768, matched ref + peak crop     (local comparison)
    E  H@768, GT oracle crop              (theoretical upper bound)

Parallelization: each experiment's stratified sample list is sharded round-robin
across ``--gpus``. Each worker process binds one GPU (`CUDA_VISIBLE_DEVICES`),
loads its own model copy, and evaluates its shard. The parent merges the per-shard
jsonl rows and re-runs ``summarize`` once, so the final numbers are identical to a
single-process run. Greedy decode is memory-bound (low single-GPU util), so the
wall-clock win here comes from splitting the independent samples across GPUs.
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

EXPERIMENTS = {
    'A_baseline_h448': {
        'prior': {'h_image_size': 448},
        'zoom': {'enabled': False},
    },
    'B_h768': {
        'prior': {'h_image_size': 768},
        'zoom': {'enabled': False},
    },
    'C_h768_peakcrop': {
        'prior': {'h_image_size': 768},
        'zoom': {'enabled': True, 'source': 'peaks', 'nms_radius': 2,
                 'crop_fraction': 0.20, 'include_reference': False},
    },
    'D_h768_pairedpeak': {
        'prior': {'h_image_size': 768},
        'zoom': {'enabled': True, 'source': 'peaks', 'nms_radius': 2,
                 'crop_fraction': 0.20, 'include_reference': True},
    },
    'E_h768_oraclecrop': {
        'prior': {'h_image_size': 768},
        'zoom': {'enabled': True, 'source': 'cc', 'include_reference': False, 'oracle': True},
    },
}

TABLE_KEYS = [
    'n', 'n_anomaly', 'n_normal', 'n_small',
    'prior_recall_at_01_small', 'prior_recall_at_03_small', 'mean_iou_h_bestk_small',
    'anomaly_recall_small', 'anomaly_recall',
    'mask_miou_small_given_tp', 'mask_miou_small',
    'matched_miou_small_given_tp',
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
    log_path = out_dir / f'worker{rank}.log'
    sys.stdout = open(log_path, 'a', buffering=1)
    sys.stderr = sys.stdout

    set_seed(int(base_cfg['training']['seed']))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'[worker{rank} gpu{gpu}] device={device}', flush=True)
    model, processor, prior = load_model(base_cfg, None, fresh_lora=False)
    model.eval()

    _, _, test_set = datasets(base_cfg, processor)
    all_idx = stratified_eval_indices(test_set, limit, seed=int(base_cfg['training']['seed']))

    for name in EXPERIMENTS:
        cfg = apply_overrides(base_cfg, EXPERIMENTS[name])
        mine = [all_idx[i] for i in range(rank, len(all_idx), num_total)]
        path = out_dir / f'{name}.w{rank}.json'
        t0 = time.time()
        print(f'\n[worker{rank} gpu{gpu}] === {name} ({len(mine)} samples) ===', flush=True)
        evaluate(cfg, model, processor, prior, test_set, path,
                 indices=mine, writer=None, namespace=f'{name}.w{rank}')
        print(f'[worker{rank} gpu{gpu}] {name} done in {(time.time()-t0)/60:.1f}m', flush=True)


def _merge_summaries(out_dir, num_total):
    from outcome.evaluate_multibox import summarize

    out_dir = Path(out_dir)
    results = {}
    for name in EXPERIMENTS:
        rows = []
        for rank in range(num_total):
            p = out_dir / f'{name}.w{rank}.jsonl'
            if p.exists():
                rows.extend(json.loads(l) for l in p.read_text().splitlines() if l.strip())
        rows.sort(key=lambda r: r['image_path'])
        (out_dir / f'{name}.jsonl').write_text(
            '\n'.join(json.dumps(r, ensure_ascii=False, default=str) for r in rows) + '\n')
        stats = summarize(rows)
        results[name] = stats
        (out_dir / f'{name}.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/qwen35_2b_annos_probe_v2_fsm_rl.yaml')
    ap.add_argument('--sft-adapter', default=None, help='override outcome.sft_adapter')
    ap.add_argument('--split', choices=['dev', 'test'], default='test')
    ap.add_argument('--limit', type=int, default=200, help='stratified eval sample count')
    ap.add_argument('--gpus', default='0,1', help='comma-separated physical GPU ids')
    ap.add_argument('--procs-per-gpu', type=int, default=2,
                    help='worker processes per GPU (decode is memory-bound; '
                         'overlap independent decodes to fill the memory pipe)')
    ap.add_argument('--out-dir', default='outputs/eval/zoom_ablation')
    args = ap.parse_args()

    base = load_yaml_config(args.config)
    if args.sft_adapter:
        base.setdefault('outcome', {})['sft_adapter'] = args.sft_adapter

    gpus = [int(x) for x in args.gpus.split(',') if x.strip()]
    ppg = max(1, int(args.procs_per_gpu))
    num_total = len(gpus) * ppg
    # rank -> gpu: consecutive ranks share a gpu (rank//ppg), then round-robin shard.
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

    results = _merge_summaries(out_dir, num_total)
    table = {name: {k: results[name].get(k) for k in TABLE_KEYS} for name in EXPERIMENTS}
    (out_dir / 'summary_table.json').write_text(json.dumps(table, ensure_ascii=False, indent=2))

    header = ['metric'] + list(EXPERIMENTS)
    rows = [[k] + [f'{table[n].get(k)}' for n in EXPERIMENTS] for k in TABLE_KEYS]
    print('\n' + '=' * 110, flush=True)
    widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    fmt = '  '.join(f'{{:<{w}}}' for w in widths)
    print(fmt.format(*header), flush=True)
    for r in rows:
        print(fmt.format(*[str(x) for x in r]), flush=True)
    print('=' * 110, flush=True)
    print(f'saved {out_dir}/summary_table.json', flush=True)


if __name__ == '__main__':
    main()
