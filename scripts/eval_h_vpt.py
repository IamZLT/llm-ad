#!/usr/bin/env python3
"""Evaluate a VPT-style H-injection SFT checkpoint (FSM staged decode).

This is the consolidated single-channel H scheme: H -> HFeatureEncoder (z_H) ->
HControlProjector cross-attention (Q=z_H, K/V=<|h_ctrl|> hidden) -> h_H scattered
into N <|h_feat|> slots. There is no RegionAdapter / HPriorAdapter / HMemory /
zoom / router / H-stat, so the ONLY knob to vary is ``outcome.h_vpt.condition``:

    real      VPT H = the actual H map
    shuffled  VPT H = deterministic spatial permutation of H
    zero      VPT H = all zeros (no spatial prior)

Success requires ``real > shuffled ~= zero`` on mask_mIoU_small|TP (the model must
consume H's spatial structure through the cross-attention), not just a lower FPR.
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

CONDITIONS = ['real', 'shuffled', 'zero']

TABLE_KEYS = [
    'n', 'n_anomaly', 'n_normal', 'n_small',
    'anomaly_recall_small', 'anomaly_recall',
    'mask_miou_small_given_tp', 'mask_miou_small',
    'matched_miou_small_given_tp',
    'mask_miou', 'union_miou',
    'normal_fpr', 'balanced_accuracy', 'task_valid_rate', 'truncation_rate',
]


def build_eval_cfg(base_cfg, sft_adapter, cond, h_size):
    cfg = copy.deepcopy(base_cfg)
    oc = cfg.setdefault('outcome', {})
    oc['sft_adapter'] = sft_adapter
    oc.setdefault('h_vpt', {})['condition'] = cond
    oc.setdefault('h_vpt', {})['enabled'] = True
    oc.setdefault('prior', {})['h_image_size'] = int(h_size)
    cfg.setdefault('grpo', {})['staged_rollout'] = True
    return cfg


def _worker(rank, gpu, base_cfg, sft_adapter, cond, h_size,
            split, limit, out_dir, num_total):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    import torch

    from outcome.engine_multibox import datasets, load_model
    from outcome.evaluate_multibox import evaluate, stratified_eval_indices
    from utils.common import set_seed

    cond_dir = Path(out_dir) / cond
    cond_dir.mkdir(parents=True, exist_ok=True)
    log_path = cond_dir / f'worker{rank}.log'
    sys.stdout = open(log_path, 'a', buffering=1)
    sys.stderr = sys.stdout

    cfg = build_eval_cfg(base_cfg, sft_adapter, cond, h_size)
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
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/qwen35_2b_annos_probe_v2_fsm_h_vpt_sft.yaml')
    ap.add_argument('--sft-adapter', required=True)
    ap.add_argument('--h-size', type=int, default=768)
    ap.add_argument('--split', choices=['dev', 'test'], default='test')
    ap.add_argument('--limit', type=int, default=200)
    ap.add_argument('--gpus', default='0,1')
    ap.add_argument('--procs-per-gpu', type=int, default=1)
    ap.add_argument('--conditions', default='real,shuffled,zero')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    base = load_yaml_config(args.config)
    conds = [c for c in args.conditions.split(',') if c.strip()]
    gpus = [int(x) for x in args.gpus.split(',') if x.strip()]
    ppg = max(1, int(args.procs_per_gpu))
    num_total = len(gpus) * ppg
    rank_gpu = [(r, gpus[r // ppg]) for r in range(num_total)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'eval_config.json').write_text(json.dumps(
        build_eval_cfg(base, args.sft_adapter, 'real', args.h_size),
        ensure_ascii=False, indent=2))

    ctx = mp.get_context('spawn')
    results = {}
    for cond in conds:
        print(f'\n===== condition = {cond} =====', flush=True)
        procs = []
        for rank, gpu in rank_gpu:
            p = ctx.Process(target=_worker,
                            args=(rank, gpu, base, args.sft_adapter, cond, args.h_size,
                                  args.split, args.limit, str(out_dir), num_total))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        results[cond] = _merge(out_dir, cond, num_total)

    print('\n' + '=' * 110)
    print('VPT H CROSS-ATTENTION CAUSAL ABLATION')
    print('=' * 110)
    hdr = f'{"metric":32s}' + ''.join(f'{c:>22s}' for c in conds)
    print(hdr)
    print('-' * 110)
    for k in TABLE_KEYS:
        row = f'{k:32s}'
        for c in conds:
            v = results[c].get(k)
            row += f'{("--" if v is None else round(v, 4)):>22}'
        print(row)
    print('=' * 110)

    if 'real' in results:
        real = results['real']
        others = [c for c in conds if c != 'real']
        real_miou = real.get('mask_miou_small_given_tp')
        print('\nVERDICT', flush=True)
        if real_miou is not None and others:
            best_other = max(results[c].get('mask_miou_small_given_tp') or 0.0 for c in others)
            gap = real_miou - best_other
            print(f'  mask_mIoU_small|TP : real={real_miou:.4f}  max(shuffled,zero)={best_other:.4f}  gap={gap:+.4f}')
            verdict = 'PASS: real > shuffled ~ zero -> VPT consumes H spatial structure' if gap > 0.02 else \
                'FAIL: real ~ shuffled ~ zero -> VPT does not consume H spatial structure'
            print(f'  VERDICT            : {verdict}')
    print(f'\nsaved under {out_dir}/{{{"|".join(conds)}}}/ (rows.jsonl + summary.json)')


if __name__ == '__main__':
    main()
