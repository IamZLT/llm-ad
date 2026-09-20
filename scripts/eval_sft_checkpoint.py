#!/usr/bin/env python3
"""Evaluate an SFT checkpoint (FSM staged decode, no RL adapter) at H@768.

This is the single-variable eval used for the region-evidence experiment tree
(A0 block24 / A1 block16 / A2 block12, then B0/B1 localize_target). It matches
the protocol that produced the A0 baseline `outputs/eval/zoom_ablation/B_h768.jsonl`:

    * base config = the FSM RL config (grpo.staged_rollout=true -> FSM decode)
    * load_model(adapter=None, fresh_lora=False) -> SFT LoRA merged, no RL
    * prior.h_image_size = 768 (decoupled from the 448 global images)
    * zoom off, n=200 stratified (seed 42), multi-GPU sharded

The three knobs that must move together with the checkpoint are passed explicitly
so a single script serves the whole tree:

    --sft-adapter        outcome.sft_adapter (the checkpoint dir)
    --region-feature-index   prior.region_feature_index (must match training)
    --h-size             prior.h_image_size (default 768)
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import load_yaml_config

TABLE_KEYS = [
    'n', 'n_anomaly', 'n_normal', 'n_small',
    'prior_recall_at_01_small', 'prior_recall_at_03_small', 'mean_iou_h_bestk_small',
    'anomaly_recall_small', 'anomaly_recall',
    'mask_miou_small_given_tp', 'mask_miou_small',
    'matched_miou_small_given_tp',
    'normal_fpr', 'balanced_accuracy', 'task_valid_rate', 'truncation_rate',
]


def build_eval_cfg(base_cfg, sft_adapter, region_feature_index, h_size):
    cfg = copy.deepcopy(base_cfg)
    oc = cfg.setdefault('outcome', {})
    oc['sft_adapter'] = sft_adapter
    p = oc.setdefault('prior', {})
    p['region_feature_index'] = int(region_feature_index)
    p['h_image_size'] = int(h_size)
    oc.setdefault('zoom', {})['enabled'] = False
    return cfg


def _worker(rank, gpu, base_cfg, sft_adapter, region_feature_index, h_size,
            split, limit, out_dir, num_total):
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

    cfg = build_eval_cfg(base_cfg, sft_adapter, region_feature_index, h_size)
    set_seed(int(cfg['training']['seed']))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'[worker{rank} gpu{gpu}] device={device}', flush=True)
    # load_model must see the overridden cfg so it merges the right sft_adapter and
    # builds AnomalyPrior with the matching region_feature_index.
    model, processor, prior = load_model(cfg, None, fresh_lora=False)
    model.eval()

    _, dev_set, test_set = datasets(cfg, processor)
    sel = test_set if split == 'test' else dev_set
    all_idx = stratified_eval_indices(sel, limit, seed=int(cfg['training']['seed']))
    mine = [all_idx[i] for i in range(rank, len(all_idx), num_total)]
    # evaluate() streams rows to path.with_suffix('.jsonl') and writes the summary
    # to path itself, so pass a '.json' path here (rows land in rows.w{rank}.jsonl).
    path = out_dir / f'rows.w{rank}.json'
    print(f'[worker{rank} gpu{gpu}] {len(mine)} samples', flush=True)
    evaluate(cfg, model, processor, prior, sel, path, indices=mine,
             writer=None, namespace=f'w{rank}')
    print(f'[worker{rank} gpu{gpu}] done', flush=True)


def _merge(out_dir, num_total):
    from outcome.evaluate_multibox import summarize

    out_dir = Path(out_dir)
    rows = []
    for rank in range(num_total):
        p = out_dir / f'rows.w{rank}.jsonl'
        if p.exists():
            rows.extend(json.loads(l) for l in p.read_text().splitlines() if l.strip())
    rows.sort(key=lambda r: r['image_path'])
    (out_dir / 'rows.jsonl').write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False, default=str) for r in rows) + '\n')
    stats = summarize(rows)
    (out_dir / 'summary.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/qwen35_2b_annos_probe_v2_fsm_rl.yaml')
    ap.add_argument('--sft-adapter', required=True)
    ap.add_argument('--region-feature-index', type=int, required=True)
    ap.add_argument('--h-size', type=int, default=768)
    ap.add_argument('--split', choices=['dev', 'test'], default='test')
    ap.add_argument('--limit', type=int, default=200)
    ap.add_argument('--gpus', default='0,1', help='comma-separated physical GPU ids')
    ap.add_argument('--procs-per-gpu', type=int, default=2)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    base = load_yaml_config(args.config)
    if 'grpo' not in base:
        base['grpo'] = {}
    base['grpo']['staged_rollout'] = True  # ensure FSM staged decode matches A0 baseline

    gpus = [int(x) for x in args.gpus.split(',') if x.strip()]
    ppg = max(1, int(args.procs_per_gpu))
    num_total = len(gpus) * ppg
    rank_gpu = [(r, gpus[r // ppg]) for r in range(num_total)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'eval_config.json').write_text(
        json.dumps(build_eval_cfg(base, args.sft_adapter, args.region_feature_index,
                                  args.h_size), ensure_ascii=False, indent=2))

    ctx = mp.get_context('spawn')
    procs = []
    for rank, gpu in rank_gpu:
        p = ctx.Process(target=_worker,
                        args=(rank, gpu, base, args.sft_adapter, args.region_feature_index,
                              args.h_size, args.split, args.limit, str(out_dir), num_total))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    stats = _merge(out_dir, num_total)
    print('\n' + '=' * 100, flush=True)
    for k in TABLE_KEYS:
        print(f'{k:32s} {stats.get(k)}', flush=True)
    print('=' * 100, flush=True)
    print(f'saved {out_dir}/rows.jsonl + summary.json', flush=True)


if __name__ == '__main__':
    main()
