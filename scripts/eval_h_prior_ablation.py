#!/usr/bin/env python3
"""Causal ablation for the HPriorAdapter side channel (H-Region Fusion).

Runs the (joint-trained) model under four input conditions and reports both final
task metrics and the H-good/mid/poor mechanism KPIs, so we can test whether the
side channel actually uses H's *spatial structure* (and not just extra params):

    off      h_prior_adapter.enabled=false   (side channel removed)
    real     enabled, condition=real         (full model)
    shuffled enabled, condition=shuffled     (H patch spatially permuted)
    zero     enabled, condition=zero         (H patch all zeros)

Proposals/geom/hstat always come from real H in every condition (only the H patch
is perturbed), so this isolates the HAdapter side channel from proposal quality.

Parallelization matches scripts/run_zoom_ablation.py.
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

CONDITIONS = ['off', 'real', 'shuffled', 'zero']

CORE_KEYS = [
    'n', 'n_anomaly', 'n_normal', 'n_small',
    'prior_recall_at_01_small', 'prior_recall_at_03_small', 'mean_iou_h_bestk_small',
    'anomaly_recall_small', 'anomaly_recall',
    'mask_miou_small_given_tp', 'mask_miou_small', 'matched_miou_small_given_tp',
    'normal_fpr', 'balanced_accuracy', 'task_valid_rate', 'truncation_rate',
]

BASE = {
    'prior': {'h_image_size': 768},
    'zoom': {'enabled': False},
}


def build_eval_cfg(base_cfg, condition):
    cfg = copy.deepcopy(base_cfg)
    oc = cfg.setdefault('outcome', {})
    for k, v in BASE.items():
        oc.setdefault(k, {}).update(v)
    hpa = oc.setdefault('h_prior_adapter', {})
    if condition == 'off':
        hpa['enabled'] = False
    else:
        hpa['enabled'] = True
        hpa['patch_size'] = 7
        hpa['intermediate_dim'] = 256
        hpa['condition'] = condition
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
    # Load once with the 'real' condition (enabled); the collator reads the condition
    # per-sample from cfg, but the mounted adapter itself does not depend on it.
    model, processor, prior = load_model(build_eval_cfg(base_cfg, 'real'), None, fresh_lora=False)
    model.eval()

    _, _, test_set = datasets(base_cfg, processor)
    all_idx = stratified_eval_indices(test_set, limit, seed=int(base_cfg['training']['seed']))
    mine = [all_idx[i] for i in range(rank, len(all_idx), num_total)]

    # The model is loaded with the adapter mounted; for 'off' we must temporarily
    # unmount it (bind_region_injection reads model.h_prior_adapter).
    from models.qwen35 import unwrap_model
    core = unwrap_model(model)
    saved_adapter = getattr(core, 'h_prior_adapter', None)

    for cond in CONDITIONS:
        cfg = build_eval_cfg(base_cfg, cond)
        if cond == 'off':
            core.h_prior_adapter = None
        else:
            core.h_prior_adapter = saved_adapter
        path = out_dir / f'{cond}.w{rank}.json'
        t0 = time.time()
        print(f'\n[worker{rank} gpu{gpu}] === {cond} ({len(mine)} samples) ===', flush=True)
        evaluate(cfg, model, processor, prior, test_set, path,
                 indices=mine, writer=None, namespace=f'{cond}.w{rank}')
        print(f'[worker{rank} gpu{gpu}] {cond} done in {(time.time()-t0)/60:.1f}m', flush=True)


def _merge(out_dir, num_total):
    from outcome.evaluate_multibox import summarize

    out_dir = Path(out_dir)
    results = {}
    for cond in CONDITIONS:
        rows = []
        for rank in range(num_total):
            p = out_dir / f'{cond}.w{rank}.jsonl'
            if p.exists():
                rows.extend(json.loads(l) for l in p.read_text().splitlines() if l.strip())
        rows.sort(key=lambda r: r['image_path'])
        (out_dir / f'{cond}.jsonl').write_text(
            '\n'.join(json.dumps(r, ensure_ascii=False, default=str) for r in rows) + '\n')
        results[cond] = summarize(rows)
        (out_dir / f'{cond}.json').write_text(json.dumps(results[cond], ensure_ascii=False, indent=2))
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/qwen35_2b_annos_probe_v2_fsm_rl.yaml')
    ap.add_argument('--sft-adapter', default='outputs/train/region_sft_annos_think_fsm_h_prior_joint_v1')
    ap.add_argument('--split', choices=['dev', 'test'], default='test')
    ap.add_argument('--limit', type=int, default=200)
    ap.add_argument('--gpus', default='0,1')
    ap.add_argument('--procs-per-gpu', type=int, default=2)
    ap.add_argument('--out-dir', default='outputs/eval/h_prior_ablation_joint')
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

    results = _merge(out_dir, num_total)

    # ---- final task metrics table ----
    print('\n' + '=' * 100, flush=True)
    header = ['metric'] + CONDITIONS
    rows = [[k] + [f'{results[c].get(k)}' for c in CONDITIONS] for k in CORE_KEYS]
    widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    fmt = '  '.join(f'{{:<{w}}}' for w in widths)
    print(fmt.format(*header), flush=True)
    for r in rows:
        print(fmt.format(*[str(x) for x in r]), flush=True)
    print('=' * 100, flush=True)

    # ---- mechanism KPI table ----
    from scripts.diagnose_h_prior_kpi import analyze
    kpi = {c: analyze(str(out_dir / f'{c}.jsonl'))[0] for c in CONDITIONS}
    print('\n' + '=' * 100, flush=True)
    print('mechanism KPI (small TP):', flush=True)
    print(f'{"metric":<24}' + ''.join(f'{c:>20}' for c in CONDITIONS), flush=True)
    print('-' * 100, flush=True)

    def fmtk(v, pct=False):
        if v is None or (isinstance(v, float) and v != v):
            return '   -'
        return f'{v*100:5.1f}%' if pct else f'{v:.3f}'

    for label, key, pct in [
        ('K1 P_degrade^good', 'k1_degrade_good', True),
        ('K2 E[D_F|H-good]', 'k2_edf_good', False),
        ('K3 H-poor recovery', 'k3_recover_poor', True),
        ('K4 IoU(candidate,H)', 'k4_iou_cand_h', False),
        ('overall P(C|small,TP)', 'overall_p_c_small_tp', True),
    ]:
        row = f'{label:<24}' + ''.join(f'{fmtk(kpi[c][key], pct=pct):>20}' for c in CONDITIONS)
        print(row, flush=True)
    print('=' * 100, flush=True)
    print(f'saved -> {out_dir}/{{off,real,shuffled,zero}}.json + .jsonl', flush=True)


if __name__ == '__main__':
    main()
