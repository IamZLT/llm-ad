#!/usr/bin/env python3
"""Collect real inspection traces on the TRAIN split with a fixed SFT checkpoint.

Phase 1 of the world-model plan, step "collect real candidates before judging
whether correction helps". The frozen SFT model generates its *own* initial boxes
B0 via the unified two-stage flow (``generate_inspection_group``), the selected
candidate is cropped and observed, and the final boxes B1 are recorded alongside
GT-derived supervision (candidate quality Q0, final quality Q1, and per-action
realized-quality labels for the action-outcome predictor).

Two sources are written, so error-type coverage does not depend on the model's
(possibly narrow) real mistakes alone:

* ``source='real'``       — the model's actual B0/B1 for a sample;
* ``source='gt_perturb'`` — GT-perturbed candidates (undershoot / overshoot /
  shift / keep) plus a false-alarm candidate, each with per-action labels.

Normal images are included; a normal sample where the model still proposed
candidate boxes contributes a real false-positive trace.

Output is one JSONL per line (``--output``), plus a ``<output>.summary.json``.

Usage:
    python scripts/collect_inspection_traces.py \
        --config configs/qwen35_2b_world_model_rl.yaml \
        --sft-adapter outputs/train/world_model_sft_2b_hvpt_768 \
        --output outputs/traces/inspection_traces.jsonl \
        [--max-samples 2000] [--n-perturb 4] [--seed 42]
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, '.')

from models.box_outcome_model import ALL_ACTIONS, action_outcome_quality, apply_selected_action
from outcome.engine_multibox import datasets, load_model
from outcome.inputs_multibox import OutcomeMultiboxCollator
from outcome.policy import generate_inspection_group
from outcome.protocol_multibox import parse_output_cfg, score_output
from outcome.zoom_crop import crop_window_from_box, make_zoom_crop
from rl.grpo import move_batch
from train_region_sft import _boxes_to_1000, _expand_box, _shift_box, _shrink_box
from utils.common import set_seed
from utils.config import load_yaml_config

_PERTURB_NAMES = ('keep', 'under', 'over', 'shift')


def _gt_components(meta) -> list:
    comps = list(meta.get('component_bboxes') or [])
    if not comps and meta.get('gt_box_px') is not None:
        comps = [meta['gt_box_px']]
    return comps


def _quality_cfg(cfg):
    return {**cfg['outcome'].get('localization', {}), **cfg['outcome'].get('reward', {})}


def _action_labels(boxes_1000, selected_index, meta, quality_cfg) -> dict:
    """{action_name: realized quality} for the selected candidate, plus the argmax."""
    if not 0 <= selected_index < len(boxes_1000):
        return {}
    iou_threshold = float(quality_cfg.get('iou_threshold', 0.30))
    geometry_weight = float(quality_cfg.get('geometry_weight', 0.30))
    labels = {}
    for ai, action in enumerate(ALL_ACTIONS):
        new_boxes = apply_selected_action(boxes_1000, selected_index, ai)
        labels[action] = action_outcome_quality(
            new_boxes, meta, iou_threshold=iou_threshold, geometry_weight=geometry_weight)
    best = max(labels, key=labels.get)
    return dict(labels=labels, argmax_action=best, argmax_quality=labels[best])


def _perturbed_candidates(meta, quality_cfg, n_perturb):
    """GT-perturbed candidate sets for one anomalous sample.

    Returns a list of dicts ``{name, boxes_1000}``: undershoot / overshoot /
    shift / keep, plus a false-alarm candidate (an H prior candidate that does not
    mark any GT component) when available. The perturbed candidates cover extent
    and offset errors the model's real B0 may not exhibit.
    """
    comps = _gt_components(meta)
    if not comps:
        return []
    gt_1000 = _boxes_to_1000(comps, meta['orig_size'])
    out = []
    for name in _PERTURB_NAMES[:max(0, n_perturb)]:
        if name == 'keep':
            boxes = [list(b) for b in gt_1000]
        elif name == 'under':
            boxes = [_shrink_box(b, 0.5) for b in gt_1000]
        elif name == 'over':
            boxes = [_expand_box(b, 1.3) for b in gt_1000]
        else:  # shift
            boxes = [_shift_box(b, 0.3) for b in gt_1000]
        out.append(dict(name=name, boxes_1000=boxes))
    # False alarm: an H candidate that misses every GT component.
    fp = _pick_fp_candidate(meta, gt_1000)
    if fp is not None:
        out.append(dict(name='false_alarm', boxes_1000=[[float(v) for v in fp]]))
    return out


def _pick_fp_candidate(meta, gt_boxes_1000, iou_threshold=0.10):
    """An H prior candidate that does not mark any GT component (false alarm)."""
    cands = meta.get('prior_candidates') or []
    from outcome.protocol_multibox import candidate_hits_comp
    for c in cands:
        box = c.get('bbox_2d')
        if box is None:
            continue
        if not any(candidate_hits_comp(box, g, iou_threshold) for g in gt_boxes_1000):
            return box
    return None


def _stratified_indices(dataset, max_samples, seed):
    """Class+normal/anomaly balanced index list over the train split."""
    rng = random.Random(seed)
    anom = []
    normal = []
    for i in range(len(dataset)):
        s = dataset.samples[i]
        if s.get('is_anomaly'):
            anom.append(i)
        else:
            normal.append(i)
    rng.shuffle(anom)
    rng.shuffle(normal)
    if max_samples is None:
        return anom + normal
    half = max_samples // 2
    picked = anom[:half] + normal[:max_samples - min(half, len(anom))]
    # If few anomalies, backfill with more normals (and vice versa).
    if len(picked) < max_samples:
        remaining = [i for i in (anom + normal) if i not in set(picked)]
        rng.shuffle(remaining)
        picked += remaining[:max_samples - len(picked)]
    rng.shuffle(picked)
    return picked


def _base_fields(meta):
    return dict(
        image_path=meta.get('image_path'), ref_path=meta.get('ref_path'),
        class_name=meta.get('class_name'), defect_type=meta.get('defect_type'),
        is_anomaly=bool(meta.get('is_anomaly')),
        orig_size=list(meta['orig_size']) if meta.get('orig_size') else None,
        gt_box_px=meta.get('gt_box_px'), component_bboxes=meta.get('component_bboxes'),
    )


def _real_trace(meta, completion, trace, parsed, score, quality_cfg):
    rec = _base_fields(meta)
    rec.update(source='real')
    rec.update(trace.to_record())
    rec['stage1_text'] = (trace.segments[0].completion.text if trace.segments else '')
    rec['stage2_text'] = (trace.segments[1].completion.text if len(trace.segments) > 1 else '')
    rec['full_text'] = completion.text
    rec['stop_reason'] = completion.stop_reason
    rec['task_valid'] = parsed['task_valid']
    rec['is_anomaly_pred'] = parsed['is_anomaly']
    rec['q0'] = score.get('q0')
    rec['q1'] = score.get('q1')
    rec['delta_refine'] = score['delta_refine']
    rec['objective_verdict'] = score.get('objective_verdict')
    rec['refine_verdict'] = score.get('refine_verdict')
    idx = int(trace.selected_box_index)
    if 0 <= idx < len(trace.initial_boxes):
        al = _action_labels(trace.initial_boxes, idx, meta, quality_cfg)
        rec.update(action_labels=al.get('labels'), argmax_action=al.get('argmax_action'),
                   argmax_quality=al.get('argmax_quality'))
    else:
        rec.update(action_labels=None, argmax_action=None, argmax_quality=None)
    return rec


def _perturb_trace(meta, p, quality_cfg):
    rec = _base_fields(meta)
    rec.update(source='gt_perturb', perturb_name=p['name'])
    rec['initial_boxes'] = p['boxes_1000']
    rec['selected_box_index'] = 0
    rec['final_boxes'] = p['boxes_1000']
    # Crop window for the perturbed candidate (deterministic, same pad convention).
    rec['zoom_executed'] = True
    rec['zoom_skip_reason'] = None
    try:
        rec['crop_window_px'] = list(crop_window_from_box(p['boxes_1000'][0], tuple(meta['orig_size'])))
    except Exception:
        rec['crop_window_px'] = None
        rec['zoom_executed'] = False
    al = _action_labels(p['boxes_1000'], 0, meta, quality_cfg)
    rec.update(action_labels=al.get('labels'), argmax_action=al.get('argmax_action'),
               argmax_quality=al.get('argmax_quality'))
    return rec


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--config', default='configs/qwen35_2b_world_model_rl.yaml')
    ap.add_argument('--sft-adapter', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--max-samples', type=int, default=None)
    ap.add_argument('--n-perturb', type=int, default=4,
                    help='number of GT-perturbed candidate variants per anomaly (0 disables)')
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    cfg['outcome']['sft_adapter'] = args.sft_adapter
    seed = args.seed if args.seed is not None else int(cfg['training']['seed'])
    set_seed(seed)

    model, processor, prior = load_model(cfg, adapter=None, fresh_lora=False)
    device = next(model.parameters()).device
    train_set, _dev_set, _test_set = datasets(cfg, processor)
    collator = OutcomeMultiboxCollator(processor, prior, cfg)
    max_boxes = int(cfg['outcome'].get('max_boxes', 16))
    quality_cfg = _quality_cfg(cfg)
    protocol_weight = float(cfg['outcome'].get('protocol_weight', 0.01))

    zoom_cfg = copy.deepcopy(cfg)
    zoom_cfg['outcome'].setdefault('zoom', {})['enabled'] = True

    indices = _stratified_indices(train_set, args.max_samples, seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_real = n_perturb = n_skipped = 0
    with out_path.open('w') as stream:
        for idx in indices:
            item = train_set[idx]
            batch = move_batch(collator([item]), device)
            meta = batch['_meta'][0]

            # Real two-stage trajectory from the frozen SFT model.
            completion, trace = generate_inspection_group(model, processor, prior, batch, zoom_cfg)
            parsed = parse_output_cfg(completion.text, zoom_cfg, max_boxes=max_boxes)
            score = score_output(parsed, meta, protocol_weight, quality_cfg, max_boxes=max_boxes)
            if not trace.initial_boxes and trace.selected_box_index < 0:
                n_skipped += 1
            rec = _real_trace(meta, completion, trace, parsed, score, quality_cfg)
            stream.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
            n_real += 1

            # GT-perturbed supplement for anomaly samples.
            if bool(meta.get('is_anomaly')) and args.n_perturb > 0:
                for p in _perturbed_candidates(meta, quality_cfg, args.n_perturb):
                    stream.write(json.dumps(_perturb_trace(meta, p, quality_cfg),
                                            ensure_ascii=False, default=str) + '\n')
                    n_perturb += 1

            if n_real % 50 == 0:
                print(f'[traces] {n_real} real / {n_perturb} perturbed', flush=True)

    summary = dict(
        output=str(out_path), n_real=n_real, n_perturb=n_perturb,
        n_no_candidate=n_skipped, seed=seed, sft_adapter=args.sft_adapter,
        actions=list(ALL_ACTIONS),
    )
    Path(str(out_path) + '.summary.json').write_text(json.dumps(summary, indent=2))
    print(f'[traces] done: {n_real} real / {n_perturb} perturbed '
          f'({n_skipped} no-candidate) -> {out_path}', flush=True)


if __name__ == '__main__':
    main()
