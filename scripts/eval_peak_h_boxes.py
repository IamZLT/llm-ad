#!/usr/bin/env python3
"""A/B H-box recipe on the same 200 MVTec eval pairs: full CC bbox vs peak-core.

Does not run the LLM. Encodes each pair once, then proposes both box styles
from the same H so the only variable is the box recipe.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.anomaly_prior import AnomalyPrior, get_qwen_visual
from models.qwen35 import setup_model_and_processor
from outcome.inputs import encode_pair_canonical, region_proposals
from outcome.protocol import iou, to_pixels
from utils.config import load_yaml_config


def pick_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device('cpu')
    free = []
    for i in range(torch.cuda.device_count()):
        info = torch.cuda.mem_get_info(i)
        free.append((info[0], i))
    free_bytes, idx = max(free)
    # Need ~6GB for 2B vision+LLM load, then we drop the LLM.
    if free_bytes < 8 * 1024 ** 3:
        return torch.device('cpu')
    return torch.device(f'cuda:{idx}')


def summarize(rows, key):
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    tag = key.split('_')[0]
    out = dict(
        n=len(rows),
        mean_bestk=mean([r[key] for r in rows]),
        ge01=mean([r[key] >= 0.1 for r in rows]),
        ge03=mean([r[key] >= 0.3 for r in rows]),
        ge05=mean([r[key] >= 0.5 for r in rows]),
        mean_n_cand=mean([r['n_' + tag] for r in rows]),
        mean_area=mean([r.get('area_' + tag) for r in rows]),
        center_hit=mean([r.get('hit_' + tag) for r in rows]),
        box_precision=mean([r.get('prec_' + tag) for r in rows]),
    )
    by = defaultdict(list)
    for r in rows:
        by[r['size_bin']].append(r[key])
    for b, vs in by.items():
        out[f'mean_bestk_{b}'] = mean(vs)
        out[f'ge03_{b}'] = mean([v >= 0.3 for v in vs])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/qwen35_2b_outcome_multibox.yaml')
    ap.add_argument('--jsonl', default='outputs/train/qwen35_2b_outcome_multibox/train_20260913_053232_599794/test_000200.jsonl')
    ap.add_argument('--out', default='outputs/eval/peak_h_boxes_ab.json')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    rows_in = [json.loads(l) for l in Path(args.jsonl).read_text().splitlines() if l.strip()]
    rows_in = [r for r in rows_in if r.get('is_anomaly')]
    if args.limit:
        rows_in = rows_in[: args.limit]
    print(f'anomaly samples: {len(rows_in)}', flush=True)

    cfg = load_yaml_config(args.config)
    cfg.setdefault('outcome', {}).setdefault('zoom', {})['enabled'] = False
    device = pick_device()
    print(f'device={device}', flush=True)
    model, processor = setup_model_and_processor(cfg, for_inference=True, freeze_vision=True)
    model = model.to(device).eval()
    prior = AnomalyPrior.from_qwen(model, cfg)
    prior.visual = get_qwen_visual(model)
    del model

    from data.prior_dataset import PriorCollator
    collator = PriorCollator(processor, prior, cfg)

    pcfg = dict(cfg.get('outcome', {}).get('prior') or {})
    p_full = {**pcfg, 'box_mode': 'full'}
    p_peak = {**pcfg, 'box_mode': 'peak_core'}

    recs = []
    for i, src in enumerate(rows_in):
        test = Image.open(src['image_path']).convert('RGB')
        ref = Image.open(src['ref_path']).convert('RGB')
        item = dict(ref=ref, test=test)
        ref_rs, test_rs = collator._align_pair(item['ref'], item['test'])
        enc = collator._concat_image_tensors(ref_rs, test_rs)
        vis = encode_pair_canonical(
            prior, enc['pixel_values'].to(device), enc['image_grid_thw'].to(device))
        orig = tuple(src.get('orig_size') or test.size)
        if isinstance(orig, list):
            orig = tuple(orig)
        gts = [src['gt_box_px']] if src.get('gt_box_px') else []

        def score(props, tag):
            boxes = [to_pixels(p['bbox_2d'], orig) for p in props]
            ious = [max((iou(b, g) for g in gts), default=0.0) for b in boxes]
            best = max(ious) if ious else 0.0
            areas = [((b[2] - b[0]) * (b[3] - b[1])) / max(orig[0] * orig[1], 1) for b in boxes]
            hits = []
            prec = []
            for b in boxes:
                cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
                hit = any(g[0] <= cx <= g[2] and g[1] <= cy <= g[3] for g in gts)
                hits.append(hit)
                inters = []
                for g in gts:
                    x1, y1 = max(b[0], g[0]), max(b[1], g[1])
                    x2, y2 = min(b[2], g[2]), min(b[3], g[3])
                    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                    pred = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
                    inters.append(inter / pred if pred > 0 else 0.0)
                prec.append(max(inters) if inters else 0.0)
            return {
                f'{tag}_bestk': best,
                f'n_{tag}': len(boxes),
                f'area_{tag}': (sum(areas) / len(areas)) if areas else 0.0,
                f'hit_{tag}': float(any(hits)) if hits else 0.0,
                f'prec_{tag}': max(prec) if prec else 0.0,
            }

        full = region_proposals(vis['patch_map'], p_full)[0]
        peak = region_proposals(vis['patch_map'], p_peak)[0]
        rec = dict(image_path=src['image_path'], class_name=src['class_name'],
                   size_bin=src.get('size_bin', 'small'),
                   baseline_logged=src.get('iou_h_bestk'))
        rec.update(score(full, 'full'))
        rec.update(score(peak, 'peak'))
        recs.append(rec)
        if (i + 1) % 20 == 0 or i + 1 == len(rows_in):
            print(f'[{i + 1}/{len(rows_in)}] full={rec["full_bestk"]:.3f} peak={rec["peak_bestk"]:.3f} '
                  f'area {rec["area_full"]:.3f}->{rec["area_peak"]:.3f} {src["class_name"]}', flush=True)

    out = dict(
        n=len(recs),
        full=summarize(recs, 'full_bestk'),
        peak=summarize(recs, 'peak_bestk'),
        logged_baseline=sum(r['baseline_logged'] or 0 for r in recs) / len(recs),
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(summary=out, rows=recs), indent=2, default=str))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == '__main__':
    main()
