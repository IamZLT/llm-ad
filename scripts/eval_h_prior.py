#!/usr/bin/env python3
"""Evaluate the HPriorAdapter mechanism probe (H@768, zoom off, h_prior enabled).

Loads the frozen FSM LoRA + frozen RegionAdapter + trained HPriorAdapter (pointed
to by --sft-adapter, whose dir holds region_adapter.pt + h_prior_adapter.pt) and
runs the FSM staged decoder, then reports BOTH the final task metrics (summarize)
and the H-good/mid/poor mechanism KPIs (diagnose_h_prior_kpi).

Parallelization matches scripts/run_zoom_ablation.py: shard test samples
round-robin across --gpus x --procs-per-gpu workers, merge the jsonl, summarize.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import load_yaml_config

CORE_KEYS = [
    'n', 'n_anomaly', 'n_normal', 'n_small',
    'prior_recall_at_01_small', 'prior_recall_at_03_small', 'mean_iou_h_bestk_small',
    'anomaly_recall_small', 'anomaly_recall',
    'mask_miou_small_given_tp', 'mask_miou_small', 'matched_miou_small_given_tp',
    'normal_fpr', 'balanced_accuracy', 'task_valid_rate', 'truncation_rate',
]

# Base config for eval must keep grpo.staged_rollout (FSM decoder); add the
# HPriorAdapter side channel + H@768 (matching the mechanism-probe SFT config).
HPRIOR = {
    'prior': {'h_image_size': 768},
    'h_prior_adapter': {'enabled': True, 'patch_size': 7,
                        'intermediate_dim': 256, 'init_gate': 0.0},
    'zoom': {'enabled': False},
}


def build_eval_cfg(base_cfg):
    import copy
    cfg = copy.deepcopy(base_cfg)
    oc = cfg.setdefault('outcome', {})
    for k, v in HPRIOR.items():
        oc.setdefault(k, {}).update(v)
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
    path = out_dir / f'h_prior.w{rank}.json'
    t0 = time.time()
    print(f'[worker{rank} gpu{gpu}] h_prior ({len(mine)} samples)', flush=True)
    evaluate(base_cfg, model, processor, prior, test_set, path,
             indices=mine, writer=None, namespace=f'h_prior.w{rank}')
    print(f'[worker{rank} gpu{gpu}] done in {(time.time()-t0)/60:.1f}m', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/qwen35_2b_annos_probe_v2_fsm_rl.yaml')
    ap.add_argument('--sft-adapter', default='outputs/train/region_sft_annos_think_fsm_h_prior_v1')
    ap.add_argument('--split', choices=['dev', 'test'], default='test')
    ap.add_argument('--limit', type=int, default=200)
    ap.add_argument('--gpus', default='0,1')
    ap.add_argument('--procs-per-gpu', type=int, default=2)
    ap.add_argument('--out-dir', default='outputs/eval/h_prior_v1')
    args = ap.parse_args()

    base = load_yaml_config(args.config)
    base = build_eval_cfg(base)
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
        p = out_dir / f'h_prior.w{rank}.jsonl'
        if p.exists():
            rows.extend(json.loads(l) for l in p.read_text().splitlines() if l.strip())
    rows.sort(key=lambda r: r['image_path'])
    (out_dir / 'h_prior.jsonl').write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False, default=str) for r in rows) + '\n')
    stats = summarize(rows)
    (out_dir / 'h_prior.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2))

    baseline = json.loads(Path('outputs/eval/zoom_ablation/summary_table.json').read_text())['B_h768']
    print('\n' + '=' * 78, flush=True)
    print(f'{"metric":<28}{"B_h768 (base)":>16}{"H-prior":>16}{"delta":>12}', flush=True)
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
    print(f'saved {out_dir}/h_prior.json  and  {out_dir}/h_prior.jsonl', flush=True)

    # mechanism KPIs (H-good/mid/poor) vs the baseline B_h768
    from scripts.diagnose_h_prior_kpi import analyze
    base_res, _ = analyze('outputs/eval/zoom_ablation/B_h768.jsonl')
    hp_res, _ = analyze(str(out_dir / 'h_prior.jsonl'))
    names = ['B_h768 (base)', 'H-prior']
    print('\n' + '=' * 70, flush=True)
    print('mechanism KPI (small TP):', flush=True)
    print(f'{"metric":<24}' + ''.join(f'{n:>22}' for n in names), flush=True)
    print('-' * 70, flush=True)

    def fmt(v, pct=False):
        if v is None or (isinstance(v, float) and v != v):
            return '   -'
        return f'{v*100:5.1f}%' if pct else f'{v:.3f}'

    keys = [
        ('K1 P_degrade^good', 'k1_degrade_good', True),
        ('K2 E[D_F|H-good]', 'k2_edf_good', False),
        ('K3 H-poor recovery', 'k3_recover_poor', True),
        ('K4 IoU(candidate,H)', 'k4_iou_cand_h', False),
        ('overall P(C|small,TP)', 'overall_p_c_small_tp', True),
    ]
    for label, key, pct in keys:
        row = f'{label:<24}'
        row += f'{fmt(base_res[key], pct=pct):>22}{fmt(hp_res[key], pct=pct):>22}'
        print(row, flush=True)
    print('=' * 70, flush=True)


if __name__ == '__main__':
    main()
