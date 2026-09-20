#!/usr/bin/env python3
"""Small-TP error propagation: H -> [localize] candidate -> final bbox.

No training. Reads the frozen FSM baseline at H@768 with zoom off
(outputs/eval/zoom_ablation/B_h768.jsonl), keeps only small anomalies that the
model detected (pred=True), and for each records:

    H best proposal IoU     (injected H hint, from prior_candidates)
    candidate best IoU      (model's [localize] stage, candidate_bboxes_2d)
    final best bbox IoU     (model's <answer>, bboxes_2d)
    final - H, final - candidate
    class, GT area, H-hit

All three are measured against the SAME GT (gt_box_px, pixel coords) so the
deltas are directly comparable. Categories (vs H, threshold 0.05):
    A improved    final > H + 0.05
    B unchanged   |final - H| <= 0.05
    C degraded    final < H - 0.05

The headline number is P(C | small, TP): how often refinement destroys spatial
evidence that H already had.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from outcome.protocol import iou, to_pixels

HIT_IOU = 0.1
DELTA = 0.05


def best_iou(boxes_1000, gt_px, orig):
    """Best single-box IoU (px) between [0,1000] boxes and the GT union box."""
    if not boxes_1000 or not gt_px:
        return None
    vals = [iou(to_pixels(b, orig), gt_px) for b in boxes_1000 if b and len(b) == 4]
    return max(vals) if vals else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--jsonl', default='outputs/eval/zoom_ablation/B_h768.jsonl',
                    help='eval jsonl to analyze (default: A0 baseline B_h768)')
    ap.add_argument('--out', default='outputs/eval/small_tp_error_diag.json')
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.jsonl).read_text().splitlines() if l.strip()]
    small_tp = [r for r in rows
                if r['is_anomaly'] and r['pred'] is True and r['size_bin'] == 'small']
    small_tp.sort(key=lambda r: r['image_path'])
    print(f'small anomalies: {sum(1 for r in rows if r["is_anomaly"] and r["size_bin"]=="small")}')
    print(f'small TP (pred=anomaly): {len(small_tp)}\n')

    data = []
    for r in small_tp:
        gt = r.get('gt_box_px')
        orig = Image.open(r['image_path']).size  # (w, h)
        h_cands = [c.get('bbox_2d') for c in r.get('prior_candidates', [])]
        h_iou = best_iou(h_cands, gt, orig)
        cand_iou = best_iou(r.get('candidate_bboxes_2d'), gt, orig)
        final_iou = best_iou(r.get('bboxes_2d'), gt, orig)
        gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1]) if gt else 0.0
        gt_area_frac = gt_area / (orig[0] * orig[1]) if orig[0] * orig[1] else 0.0
        dh = (final_iou - h_iou) if (final_iou is not None and h_iou is not None) else None
        dc = (final_iou - cand_iou) if (final_iou is not None and cand_iou is not None) else None
        cat = 'C_degraded' if dh < -DELTA else ('A_improved' if dh > DELTA else 'B_unchanged')
        data.append(dict(
            sample=Path(r['image_path']).parent.name + '/' + Path(r['image_path']).name,
            cls=r['class_name'], h=h_iou, c=cand_iou, f=final_iou, dh=dh, dc=dc,
            cat=cat, gt_area=gt_area, gt_area_frac=gt_area_frac,
            h_hit=(h_iou or 0.0) >= HIT_IOU, mask_iou=r.get('mask_iou'),
            verify=r.get('verify_action'), iou_h_bestk_rec=r.get('iou_h_bestk')))

    # ---- per-sample table ----
    def fmt(v):
        return '  -' if v is None else f'{v:.3f}'

    hdr = (f'{"sample":<34}{"class":<11}{"H":>7}{"cand":>7}{"final":>7}'
           f'{"dH":>8}{"dC":>8}{"GTarea":>9}{"GT%":>7}{"Hhit":>5}  {"category":<12}')
    print(hdr)
    print('-' * len(hdr))
    for d in data:
        print(f'{d["sample"]:<34}{d["cls"]:<11}{fmt(d["h"]):>7}{fmt(d["c"]):>7}{fmt(d["f"]):>7}'
              f'{fmt(d["dh"]):>8}{fmt(d["dc"]):>8}'
              f'{d["gt_area"]:>9.0f}{d["gt_area_frac"]:>7.3f}{"Y" if d["h_hit"] else "N":>5}  {d["cat"]:<12}')

    # ---- summary ----
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float('nan')

    n = len(data)
    a = sum(1 for d in data if d['cat'] == 'A_improved')
    b = sum(1 for d in data if d['cat'] == 'B_unchanged')
    c = sum(1 for d in data if d['cat'] == 'C_degraded')
    print('\n' + '=' * 74)
    print(f'small TP 分类 (vs H, Δ={DELTA}):')
    print(f'  A improved   : {a:3d}  ({a/n*100:5.1f}%)')
    print(f'  B unchanged  : {b:3d}  ({b/n*100:5.1f}%)')
    print(f'  C degraded   : {c:3d}  ({c/n*100:5.1f}%)   <-- P(C | small,TP)')
    print('-' * 74)
    print(f'  mean H IoU         = {mean(d["h"] for d in data):.3f}')
    print(f'  mean candidate IoU = {mean(d["c"] for d in data):.3f}')
    print(f'  mean final IoU     = {mean(d["f"] for d in data):.3f}')
    print(f'  mean (final - H)        = {mean(d["dh"] for d in data):+.3f}')
    print(f'  mean (final - cand)     = {mean(d["dc"] for d in data):+.3f}')
    print(f'  mean mask_iou (bbox-vs-mask) = {mean(d["mask_iou"] for d in data):.3f}')
    print(f'  H-hit rate (H IoU >= {HIT_IOU}) = {sum(d["h_hit"] for d in data)/n*100:.1f}%')
    print('-' * 74)
    # cross-check: my H IoU (vs union gt_box_px) vs record iou_h_bestk (vs components)
    print('  sanity: mean |myH - recH| = '
          f'{mean(abs(d["h"] - d["iou_h_bestk_rec"]) for d in data if d["iou_h_bestk_rec"] is not None):.4f}')
    # candidate->final sub-chain (does [localize]->[confirm] degrade?)
    c_deg = sum(1 for d in data if d['dc'] is not None and d['dc'] < -DELTA)
    c_imp = sum(1 for d in data if d['dc'] is not None and d['dc'] > DELTA)
    c_same = n - c_deg - c_imp
    print(f'  candidate->final: improved {c_imp} ({c_imp/n*100:.1f}%)  '
          f'unchanged {c_same} ({c_same/n*100:.1f}%)  degraded {c_deg} ({c_deg/n*100:.1f}%)')
    # verify-action distribution among degraded
    print('\n  verify_action among C_degraded:')
    from collections import Counter
    print('   ', dict(Counter(d['verify'] for d in data if d['cat'] == 'C_degraded')))
    print('=' * 74)

    out = Path(args.out)
    out.write_text(json.dumps(data, indent=2, default=str))
    print(f'\nsaved per-sample dump -> {out}')


if __name__ == '__main__':
    main()
