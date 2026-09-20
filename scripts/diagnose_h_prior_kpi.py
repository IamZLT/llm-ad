#!/usr/bin/env python3
"""H-Region Fusion mechanism KPI: bucket small TPs by IoU(H, GT) and measure the
H -> [localize] candidate -> final bbox propagation.

Consumes eval jsonl rows (same schema as scripts/diagnose_small_tp_error.py) and
reports the four mechanism KPIs from the H-Region Fusion plan:

    H-good: IoU(H, GT) >= 0.4
    H-mid : 0.2 <= IoU(H, GT) < 0.4
    H-poor: IoU(H, GT) < 0.2

    D_L = IoU(candidate, GT) - IoU(H, GT)
    D_F = IoU(final, GT)     - IoU(H, GT)

    K1  P_degrade^good = P(IoU_final < IoU_H - 0.05 | H-good)   (most important)
    K2  E[D_F | H-good]
    K3  H-poor recovery = P(IoU_final > IoU_H + 0.10 | H-poor)
    K4  IoU(candidate, H)   (diagnostic only)

Run with one jsonl for a single model, or several to compare side by side
(e.g. baseline B_h768 vs the HPriorAdapter mechanism probe).
"""
import argparse
import json
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from outcome.protocol import iou, to_pixels

DELTA = 0.05
RECOVERY = 0.10
GOOD = 0.40
MID = 0.20


def best_iou(boxes_1000, gt_px, orig):
    """Best single-box IoU (px) between [0,1000] boxes and the GT union box."""
    if not boxes_1000 or not gt_px:
        return None
    vals = [iou(to_pixels(b, orig), gt_px) for b in boxes_1000 if b and len(b) == 4]
    return max(vals) if vals else None


def iou_between(a_1000, b_1000, orig):
    """Best IoU between two [0,1000] box sets (px)."""
    if not a_1000 or not b_1000:
        return None
    ap = [to_pixels(b, orig) for b in a_1000 if b and len(b) == 4]
    bp = [to_pixels(b, orig) for b in b_1000 if b and len(b) == 4]
    if not ap or not bp:
        return None
    return max(iou(x, y) for x in ap for y in bp)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float('nan')


def bucket(h_iou):
    if h_iou is None:
        return None
    if h_iou >= GOOD:
        return 'H-good'
    if h_iou >= MID:
        return 'H-mid'
    return 'H-poor'


def analyze(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    small_tp = [r for r in rows
                if r.get('is_anomaly') and r.get('pred') is True and r.get('size_bin') == 'small']
    small_tp.sort(key=lambda r: r.get('image_path', ''))

    data = []
    for r in small_tp:
        gt = r.get('gt_box_px')
        try:
            orig = Image.open(r['image_path']).size
        except Exception:
            orig = None
        h_cands = [c.get('bbox_2d') for c in r.get('prior_candidates', [])]
        h_iou = best_iou(h_cands, gt, orig)
        cand_iou = best_iou(r.get('candidate_bboxes_2d'), gt, orig)
        final_iou = best_iou(r.get('bboxes_2d'), gt, orig)
        cand_h = iou_between(r.get('candidate_bboxes_2d'), h_cands, orig)
        dl = (cand_iou - h_iou) if (cand_iou is not None and h_iou is not None) else None
        df = (final_iou - h_iou) if (final_iou is not None and h_iou is not None) else None
        data.append(dict(h=h_iou, c=cand_iou, f=final_iou, dl=dl, df=df, ch=cand_h,
                         b=bucket(h_iou)))

    out = {}
    n = len(data)
    out['n_small_tp'] = n
    out['n_small_anom'] = sum(1 for r in rows if r.get('is_anomaly') and r.get('size_bin') == 'small')

    # bucket counts and D_L / D_F means
    buckets = {}
    for name in ('H-good', 'H-mid', 'H-poor'):
        grp = [d for d in data if d['b'] == name]
        buckets[name] = dict(
            n=len(grp),
            mean_h=mean(d['h'] for d in grp),
            mean_c=mean(d['c'] for d in grp),
            mean_f=mean(d['f'] for d in grp),
            mean_dl=mean(d['dl'] for d in grp),
            mean_df=mean(d['df'] for d in grp),
            degrade=sum(1 for d in grp if d['df'] is not None and d['df'] < -DELTA) / len(grp) if grp else float('nan'),
            improve=sum(1 for d in grp if d['df'] is not None and d['df'] > DELTA) / len(grp) if grp else float('nan'),
        )
    out['buckets'] = buckets

    good = buckets['H-good']
    poor = buckets['H-poor']

    # K1: P(IoU_final < IoU_H - 0.05 | H-good)
    out['k1_degrade_good'] = good['degrade']
    # K2: E[D_F | H-good]
    out['k2_edf_good'] = good['mean_df']
    # K3: H-poor recovery = P(final > H + 0.10 | H-poor)
    out['k3_recover_poor'] = (
        sum(1 for d in data if d['b'] == 'H-poor' and d['df'] is not None and d['df'] > RECOVERY)
        / poor['n'] if poor['n'] else float('nan')
    )
    # K4: IoU(candidate, H) diagnostic
    out['k4_iou_cand_h'] = mean(d['ch'] for d in data)

    # overall continuity with the old small-TP report
    out['overall_p_c_small_tp'] = (
        sum(1 for d in data if d['df'] is not None and d['df'] < -DELTA) / n if n else float('nan')
    )
    out['overall_mean_h'] = mean(d['h'] for d in data)
    out['overall_mean_c'] = mean(d['c'] for d in data)
    out['overall_mean_f'] = mean(d['f'] for d in data)
    return out, data


def fmt(v, pct=False):
    if v is None or (isinstance(v, float) and v != v):
        return '   -'
    return f'{v*100:5.1f}%' if pct else f'{v:.3f}'


def print_one(name, res):
    print(f'\n=== {name}  (small TP n={res["n_small_tp"]}, small anom n={res["n_small_anom"]}) ===')
    print(f'{"bucket":<8}{"n":>5}{"meanH":>8}{"meanC":>8}{"meanF":>8}{"D_L":>8}{"D_F":>8}'
          f'{"degr%":>7}{"impr%":>7}')
    for b, v in res['buckets'].items():
        print(f'{b:<8}{v["n"]:>5}{v["mean_h"]:>8.3f}{v["mean_c"]:>8.3f}{v["mean_f"]:>8.3f}'
              f'{v["mean_dl"]:>+8.3f}{v["mean_df"]:>+8.3f}'
              f'{fmt(v["degrade"], pct=True):>7}{fmt(v["improve"], pct=True):>7}')
    print('-' * 62)
    print(f'  K1 P_degrade^good      = {fmt(res["k1_degrade_good"], pct=True)}')
    print(f'  K2 E[D_F | H-good]     = {res["k2_edf_good"]:+.3f}')
    print(f'  K3 H-poor recovery     = {fmt(res["k3_recover_poor"], pct=True)}')
    print(f'  K4 IoU(candidate,H)    = {res["k4_iou_cand_h"]:.3f}')
    print(f'  overall P(C|small,TP)  = {fmt(res["overall_p_c_small_tp"], pct=True)}')
    print(f'  overall mean H/C/F     = {res["overall_mean_h"]:.3f} / '
          f'{res["overall_mean_c"]:.3f} / {res["overall_mean_f"]:.3f}')
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--jsonl', nargs='+', required=True,
                    help='one or more eval jsonl paths to analyze')
    ap.add_argument('--out', default=None, help='optional json path to dump per-model KPIs')
    args = ap.parse_args()

    results = {}
    for p in args.jsonl:
        name = Path(p).parent.name + '/' + Path(p).name
        res, _ = analyze(p)
        results[name] = res
        print_one(name, res)

    if len(results) > 1:
        names = list(results)
        print('\n' + '=' * 70)
        print('side-by-side mechanism KPI (small TP):')
        print(f'{"metric":<24}' + ''.join(f'{n:>22}' for n in names))
        print('-' * 70)
        keys = [
            ('K1 P_degrade^good', 'k1_degrade_good', True),
            ('K2 E[D_F|H-good]', 'k2_edf_good', False),
            ('K3 H-poor recovery', 'k3_recover_poor', True),
            ('K4 IoU(candidate,H)', 'k4_iou_cand_h', False),
            ('overall P(C|small,TP)', 'overall_p_c_small_tp', True),
        ]
        for label, key, pct in keys:
            row = f'{label:<24}'
            for n in names:
                v = results[n][key]
                row += f'{fmt(v, pct=pct):>22}'
            print(row)
        print('=' * 70)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f'\nsaved -> {args.out}')


if __name__ == '__main__':
    main()
