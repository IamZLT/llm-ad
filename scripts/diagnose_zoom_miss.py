#!/usr/bin/env python3
"""Diagnose the small-defect recall drop: B_h768 (zoom off) vs Zoom-v1 (top-1 crop).

Answers: are the ~8 newly-missed small anomalies caused by (a) top-1 H peak
missing the defect (crop shows normal → model rejects), or (b) the model
rejecting *despite* a good H hit (over-conservative decision boundary)?
"""
import json
import re
import sys
from pathlib import Path

BASE = 'outputs/eval/zoom_ablation/B_h768.jsonl'
ZOOM = 'outputs/eval/zoom_sft_v1/zoom_v1.jsonl'


def load(path):
    return {r['image_path']: r for r in
            (json.loads(l) for l in Path(path).read_text().splitlines() if l.strip())}


def confirm_text(text):
    m = re.search(r'\[confirm\]\s*(.*?)(?=\</think\>|<answer>)', text or '', re.S)
    if not m:
        return ''
    return ' '.join(m.group(1).split())


def main():
    base = load(BASE)
    zoom = load(ZOOM)
    keys = sorted(set(base) & set(zoom))
    print(f'common samples: {len(keys)}')

    small = [k for k in keys if base[k]['is_anomaly'] and base[k]['size_bin'] == 'small']
    print(f'small anomalies: {len(small)}')

    regressed = []   # baseline caught, zoom missed
    recovered = []   # baseline missed, zoom caught
    both_miss = []
    for k in small:
        b_hit = base[k]['pred']
        z_hit = zoom[k]['pred']
        if b_hit and not z_hit:
            regressed.append(k)
        elif not b_hit and z_hit:
            recovered.append(k)
        elif not b_hit and not z_hit:
            both_miss.append(k)

    print(f'\nregressed (baseline TP -> zoom FN): {len(regressed)}')
    print(f'recovered (baseline FN -> zoom TP): {len(recovered)}')
    print(f'both miss:                          {len(both_miss)}')

    print('\n=== REGRESSED small anomalies ===')
    for k in regressed:
        b, z = base[k], zoom[k]
        defect = Path(k).parent.name + '/' + Path(k).name
        print(f'\n  {defect}')
        print(f'    H top1 IoU={z["iou_h_top1"]:.3f}  H bestk IoU={z["iou_h_bestk"]:.3f}  '
              f'prior_any_top1={z["prior_any_top1_iou"]:.3f}')
        print(f'    base: pred={b["pred"]} verify={b["verify_action"]} mask_iou={b["mask_iou"]:.3f}')
        print(f'    zoom: pred={z["pred"]} verify={z["verify_action"]} mask_iou={z["mask_iou"]:.3f}')
        c = confirm_text(z['text'])
        if c:
            print(f'    zoom [confirm]: {c[:240]}')

    # distribution of iou_h_top1 among regressed vs caught
    def top1_dist(ks, src):
        vals = [src[k]['iou_h_top1'] for k in ks]
        return vals

    caught = [k for k in small if zoom[k]['pred']]
    print('\n=== iou_h_top1 distribution ===')
    print(f'  zoom caught small (n={len(caught)}): mean={sum(top1_dist(caught, zoom))/max(1,len(caught)):.3f}')
    print(f'  regressed small    (n={len(regressed)}): mean={sum(top1_dist(regressed, zoom))/max(1,len(regressed)):.3f}')

    # how many regressed have a top-1 H hit that clearly MISSES the defect (<0.1 IoU)
    low_hit = [k for k in regressed if zoom[k]['iou_h_top1'] < 0.1]
    good_hit = [k for k in regressed if zoom[k]['iou_h_top1'] >= 0.1]
    print(f'\n  regressed with H-top1 IoU < 0.1 (peak misses defect): {len(low_hit)}')
    print(f'  regressed with H-top1 IoU >= 0.1 (peak hits, model still rejects): {len(good_hit)}')
    print('\n  regressed w/ good hit details:')
    for k in good_hit:
        z = zoom[k]
        c = confirm_text(z['text'])
        print(f'    {Path(k).parent.name}/{Path(k).name}: top1={z["iou_h_top1"]:.3f} '
              f'verify={z["verify_action"]} | {c[:160]}')


if __name__ == '__main__':
    main()
