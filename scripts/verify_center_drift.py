"""Verify the "detected but mislocated" hypothesis.

Quantifies, over eval jsonl records, how often the model (1) correctly detects an
anomaly but (2) places the box with a center offset that the current DCLR reward
barely penalizes (because ``s_center`` normalizes by the *whole-image diagonal*
rather than the *defect scale*).

Reads the ``test_*.jsonl`` files in a training output dir and, for each anomalous
sample with a non-empty prediction, computes:

  * ``dist``        : pixel center offset between pred box and GT box
  * ``gt_diag``     : the defect's own diagonal (defect scale)
  * ``s_center_img`` : 1 - dist/img_diag  (CURRENT reward normalization)
  * ``s_center_def`` : 1 - dist/gt_diag   (defect-scale normalization)

A sample is "detected but drifted" when ``pred == True`` and
``dist >= 0.5 * gt_diag`` (the box center is off by at least half a defect
width). The report contrasts how invisible that drift is under the current
whole-image normalization vs. a defect-scale one.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image


def _to_px(box, wh):
    return [box[0] * wh[0] / 1000.0, box[1] * wh[1] / 1000.0,
            box[2] * wh[0] / 1000.0, box[3] * wh[1] / 1000.0]


def _center(box):
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def analyze_one_jsonl(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    anom = [r for r in rows if r.get('is_anomaly')]
    # detected = predicted anomaly (regardless of box presence)
    detected = [r for r in anom if r.get('pred') is True]
    # localized = detected AND has at least one box
    localized = [r for r in detected if r.get('bboxes_2d')]

    drift = []       # localized but center drifted >= 0.5 * defect
    s_img, s_def, dist_ratios = [], [], []
    maskiou_low = 0
    for r in localized:
        try:
            W, H = Image.open(r['image_path']).size
        except Exception:
            continue
        gt = r['gt_box_px']
        if not gt:
            continue
        boxes = r['bboxes_2d']
        # union of pred boxes -> one box for center comparison
        px = [_to_px(b, (W, H)) for b in boxes]
        xs0 = min(b[0] for b in px); ys0 = min(b[1] for b in px)
        xs1 = max(b[2] for b in px); ys1 = max(b[3] for b in px)
        pc = ((xs0 + xs1) / 2.0, (ys0 + ys1) / 2.0)
        gc = _center(gt)
        dist = math.hypot(pc[0] - gc[0], pc[1] - gc[1])
        img_diag = math.hypot(W, H)
        gw = gt[2] - gt[0]; gh = gt[3] - gt[1]
        gt_diag = math.hypot(gw, gh)
        if gt_diag <= 0:
            continue
        ratio = dist / gt_diag
        s_img.append(1.0 - min(1.0, dist / img_diag))
        s_def.append(max(0.0, 1.0 - ratio))
        dist_ratios.append(ratio)
        if r.get('mask_iou', 0.0) < 0.3:
            maskiou_low += 1
        if ratio >= 0.5:
            drift.append((r, dist, gt_diag, ratio))

    def _mean(xs):
        return sum(xs) / len(xs) if xs else float('nan')

    return dict(
        n_anom=len(anom),
        n_detected=len(detected),
        n_localized=len(localized),
        n_drift=len(drift),
        drift_frac=len(drift) / len(localized) if localized else float('nan'),
        maskiou_low_frac=maskiou_low / len(localized) if localized else float('nan'),
        mean_dist_ratio=_mean(dist_ratios),
        mean_s_center_img=_mean(s_img),
        mean_s_center_def=_mean(s_def),
        drift_samples=[dict(path=r['image_path'], dist=round(d, 1),
                            gt_diag=round(gd, 1), ratio=round(rt, 2),
                            mask_iou=round(r.get('mask_iou', 0.0), 3))
                       for r, d, gd, rt in drift],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('output_dir', type=str)
    ap.add_argument('--latest', action='store_true',
                    help='analyze only the latest step (largest step number)')
    args = ap.parse_args()
    d = Path(args.output_dir)
    files = sorted(d.glob('test_*.jsonl'),
                   key=lambda p: int(p.stem.split('_')[-1]))
    if not files:
        raise SystemExit(f'no test_*.jsonl under {d}')
    if args.latest:
        files = files[-1:]
    for f in files:
        s = analyze_one_jsonl(f)
        print(f"\n=== {f.name} ===")
        print(f"  anomaly={s['n_anom']} detected={s['n_detected']} localized={s['n_localized']}")
        print(f"  DETECTED-BUT-DRIFTED (dist>=0.5*defect): {s['n_drift']}/{s['n_localized']} "
              f"= {s['drift_frac']:.1%}")
        print(f"  localized-with-maskIoU<0.3: {s['maskiou_low_frac']:.1%}")
        print(f"  mean dist/defect_diag     = {s['mean_dist_ratio']:.2f}  "
              f"(1.0 = off by a full defect)")
        print(f"  mean s_center (img_diag)  = {s['mean_s_center_img']:.3f}   <- current reward")
        print(f"  mean s_center (defect_diag)= {s['mean_s_center_def']:.3f}   <- defect-scale")
        if s['drift_samples']:
            print("  drift examples (dist / defect_diag / mask_iou):")
            for x in s['drift_samples'][:8]:
                print(f"    ratio={x['ratio']:.2f} dist={x['dist']}px gt_diag={x['gt_diag']}px "
                      f"mask_iou={x['mask_iou']}  {Path(x['path']).name}")


if __name__ == '__main__':
    main()
