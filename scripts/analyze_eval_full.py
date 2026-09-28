#!/usr/bin/env python3
"""Cross-analysis for full MVTec eval: reads the per-sample .jsonl + summary .json
and emits a richer analysis.json (self-calibration, error distributions, etc.).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def mean(vals):
    vals = list(vals)
    return sum(vals) / len(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', required=True)
    ap.add_argument('--summary', default=None)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.jsonl).read_text().splitlines() if l.strip()]
    abnormal = [r for r in rows if r['is_anomaly']]
    normal = [r for r in rows if not r['is_anomaly']]

    def pct_cond(cond, sub):
        s = [r for r in sub if cond(r)]
        return len(s) / len(sub) if sub else None

    out = {}

    # --- Self-calibration (world-model) ---
    out['imagine_correct'] = {
        'all': mean(r.get('imagine_correct') for r in rows if r.get('imagine_correct') is not None),
        'abnormal': mean(r.get('imagine_correct') for r in abnormal if r.get('imagine_correct') is not None),
    }
    out['discrim_correct'] = {
        'all': mean(r.get('discrim_correct') for r in rows if r.get('discrim_correct') is not None),
        'abnormal': mean(r.get('discrim_correct') for r in abnormal if r.get('discrim_correct') is not None),
    }

    # --- Refine verdict distribution (only where zoom executed on anomalous) ---
    zoomed_anom = [r for r in abnormal if r.get('zoom_executed')]
    out['refine_verdict_dist'] = dict(Counter(r.get('refine_verdict') for r in zoomed_anom))
    out['n_zoom_executed_anomaly'] = len(zoomed_anom)

    # --- Zoom skip reasons by defect size bin ---
    skip_by_size = defaultdict(Counter)
    for r in abnormal:
        if r.get('zoom_skip_reason'):
            skip_by_size[r['size_bin']][r['zoom_skip_reason']] += 1
    out['zoom_skip_reason_by_size'] = {k: dict(v) for k, v in skip_by_size.items()}

    # --- False positives (normal predicted anomaly) by class ---
    fp = [r for r in normal if r['pred'] is True]
    out['false_positive_by_class'] = dict(Counter(r['class_name'] for r in fp))
    out['n_false_positive'] = len(fp)

    # --- False negatives (anomaly missed) by class + size ---
    fn = [r for r in abnormal if r['pred'] is not True]
    out['false_negative_by_class'] = dict(Counter(r['class_name'] for r in fn))
    out['false_negative_by_size'] = dict(Counter(r['size_bin'] for r in fn))
    out['n_false_negative'] = len(fn)

    # --- Invalid decisions (pred None) ---
    out['invalid_by_class'] = dict(Counter(r['class_name'] for r in rows if r['pred'] is None))
    out['n_invalid'] = sum(1 for r in rows if r['pred'] is None)

    # --- Per-class full table ---
    by_class = defaultdict(list)
    for r in rows:
        by_class[r['class_name']].append(r)
    per_class = {}
    for c, rs in sorted(by_class.items()):
        a = [r for r in rs if r['is_anomaly']]
        n = [r for r in rs if not r['is_anomaly']]
        per_class[c] = {
            'n': len(rs), 'n_anomaly': len(a), 'n_normal': len(n),
            'recall': mean(r['pred'] is True for r in a),
            'tnr': mean(r['pred'] is False for r in n),
            'mask_miou': mean(r['mask_iou'] for r in a),
            'union_miou': mean(r['union_iou'] for r in a),
            'matched_miou': mean(r['matched_miou'] for r in a if r.get('matched_miou') is not None),
            'det_f1_at_50': mean(r['det_f1_at_50'] for r in a if r.get('det_f1_at_50') is not None),
            'det_f1_at_75': mean(r['det_f1_at_75'] for r in a if r.get('det_f1_at_75') is not None),
            'count_error': mean(r['count_error'] for r in a if r.get('count_error') is not None),
        }
    out['per_class'] = per_class

    # --- Count-error (component count) breakdown ---
    out['count_error_dist'] = dict(Counter(round(r['count_error'], 3) for r in abnormal if r.get('count_error') is not None))
    out['num_components_dist'] = dict(Counter(r['num_components'] for r in abnormal))
    out['num_boxes_dist'] = dict(Counter(r['num_boxes'] for r in abnormal))

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
