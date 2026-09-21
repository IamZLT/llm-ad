"""Decompose *why* a localized-but-low-IoU box misses: center vs width vs height.

For every anomalous sample with a non-empty prediction, compute the DCLR
per-axis terms (s_center / s_w / s_h) and the area ratio, then bucket the
low-IoU cases by which axis dominates the miss.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from PIL import Image


def _to_px(box, wh):
    return [box[0] * wh[0] / 1000.0, box[1] * wh[1] / 1000.0,
            box[2] * wh[0] / 1000.0, box[3] * wh[1] / 1000.0]


def _iou(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    aa = (a[2]-a[0])*(a[3]-a[1]); bb = (b[2]-b[0])*(b[3]-b[1])
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def main():
    d = Path(sys.argv[1])
    files = sorted(d.glob('test_*.jsonl'),
                   key=lambda p: int(p.stem.split('_')[-1]))
    f = files[-1]
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    anom = [r for r in rows if r.get('is_anomaly')]
    localized = [r for r in anom if r.get('pred') is True and r.get('bboxes_2d')]

    print(f"=== {f.name}: {len(localized)} localized anomaly samples ===\n")
    print("bucket by dominant miss (lowest of s_center/s_w/s_h among IoU<0.5):\n")

    center_miss = w_miss = h_miss = 0
    too_small = too_big = ok_scale = 0
    low = []
    for r in localized:
        try:
            W, H = Image.open(r['image_path']).size
        except Exception:
            continue
        gt = r['gt_box_px']
        if not gt:
            continue
        px = [_to_px(b, (W, H)) for b in r['bboxes_2d']]
        x0 = min(b[0] for b in px); y0 = min(b[1] for b in px)
        x1 = max(b[2] for b in px); y1 = max(b[3] for b in px)
        pred = [x0, y0, x1, y1]
        iou = _iou(pred, gt)
        pcx = (x0+x1)/2; pcy = (y0+y1)/2
        gcx = (gt[0]+gt[2])/2; gcy = (gt[1]+gt[3])/2
        dist = math.hypot(pcx-gcx, pcy-gcy)
        img_diag = math.hypot(W, H)
        s_center = 1 - min(1, dist/img_diag)
        pw = x1-x0; ph = y1-y0
        gw = gt[2]-gt[0]; gh = gt[3]-gt[1]
        s_w = min(pw, gw)/max(pw, gw) if pw > 0 and gw > 0 else 0
        s_h = min(ph, gh)/max(ph, gh) if ph > 0 and gh > 0 else 0
        area_ratio = (pw*ph)/(gw*gh) if gw*gh > 0 else 0
        if iou < 0.5:
            low.append((r, s_center, s_w, s_h, area_ratio, dist, gw, gh, iou))
            # dominant miss
            worst = min(s_center, s_w, s_h)
            if worst == s_center:
                center_miss += 1
            elif worst == s_w:
                w_miss += 1
            else:
                h_miss += 1
        if area_ratio < 0.5:
            too_small += 1
        elif area_ratio > 2.0:
            too_big += 1
        else:
            ok_scale += 1

    n_low = len(low)
    print(f"low-IoU (<0.5) samples: {n_low} / {len(localized)}")
    if n_low:
        print(f"  dominant miss = center : {center_miss} ({center_miss/n_low:.0%})")
        print(f"  dominant miss = width  : {w_miss} ({w_miss/n_low:.0%})")
        print(f"  dominant miss = height : {h_miss} ({h_miss/n_low:.0%})")
    print(f"\narea ratio vs GT (all localized):")
    print(f"  too_small (<0.5x) : {too_small} ({too_small/len(localized):.0%})")
    print(f"  too_big   (>2.0x) : {too_big} ({too_big/len(localized):.0%})")
    print(f"  ok_scale  (0.5-2x): {ok_scale} ({ok_scale/len(localized):.0%})")

    print("\nlow-IoU breakdown (s_center, s_w, s_h, area_ratio, iou):")
    for r, sc, sw, sh, ar, dist, gw, gh, iou in sorted(low, key=lambda t: t[8])[:15]:
        print(f"  c={sc:.2f} w={sw:.2f} h={sh:.2f} area={ar:.2f}x "
              f"iou={iou:.2f} def={int(gw)}x{int(gh)}px  {Path(r['image_path']).name}")


if __name__ == '__main__':
    main()
