#!/usr/bin/env python3
"""H prior candidate statistics on the VisA train split (CPU-only, no LLM forward).

Validates the proposed SFT ground-stage redesign in which the <ground> target is
taught as "read out the H region hints" (candidate_bboxes_2d = prior_candidates):

  normals  : H fire rate and candidate-count distribution — feasibility of the
             localize-then-reject target on normal samples;
  anomalies: GT-component coverage by H candidates at several IoU thresholds,
             spurious-candidate rate, matched candidate-GT IoU (is there real
             refine headroom, or are H boxes already == GT?), and the resulting
             verify-verb distribution under the proposed target rules.

GPUs are busy with RL training, so this runs the frozen vision tower on CPU.
The full model is loaded only because setup_model_and_processor returns it;
the LLM part is dropped immediately and only the vision tower is kept.

Usage:
  python scripts/prior_candidate_stats.py --config configs/qwen35_4b_outcome_multibox.yaml
  python scripts/prior_candidate_stats.py --max-per-class 5 --workers 2   # smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from utils.config import load_yaml_config

COVER_THRESHOLDS = (0.05, 0.10, 0.20, 0.30)
PRIMARY_T = 0.10

_G: Dict[str, Any] = {}


def _iou1000(a: List[float], b: List[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + ab - inter
    return float(inter / union) if union > 0 else 0.0


def _comp_to_1000(box: List[float], orig_size) -> List[float]:
    w, h = float(orig_size[0]), float(orig_size[1])
    return [box[0] / w * 1000.0, box[1] / h * 1000.0, box[2] / w * 1000.0, box[3] / h * 1000.0]


def _worker_init(cfg: dict, threads: int) -> None:
    torch.set_num_threads(int(threads))
    from models.anomaly_prior import AnomalyPrior
    from models.qwen35 import setup_model_and_processor

    model, processor = setup_model_and_processor(cfg, for_inference=True, freeze_vision=True)
    prior = AnomalyPrior.from_qwen(model, cfg)
    del model  # free the LLM; the frozen vision tower stays alive via prior.visual
    import gc

    gc.collect()
    _G["prior"] = prior
    _G["processor"] = processor
    _G["cfg"] = cfg


def _run_shard(samples: List[dict]) -> List[dict]:
    from data.prior_dataset import build_train_ref_pool
    from outcome.inputs import (
        OutcomeCollator,
        OutcomeDataset,
        encode_pair_canonical,
        region_proposals,
    )

    cfg, prior, processor = _G["cfg"], _G["prior"], _G["processor"]
    # Shards contain whole classes, so the per-class ref pool built from the
    # shard is identical to the one built from the full train split.
    ref_pool = build_train_ref_pool(samples)
    ds = OutcomeDataset(samples, cfg, processor, mode="eval", ref_pool=ref_pool)
    collator = OutcomeCollator(processor, prior, cfg)
    pcfg = (cfg.get("outcome") or {}).get("prior") or {}

    records: List[dict] = []
    for i in range(len(ds)):
        item = ds[i]
        ref_rs, test_rs = collator._align_pair(item["ref"], item["test"])
        enc = collator._concat_image_tensors(ref_rs, test_rs)
        vis = encode_pair_canonical(prior, enc["pixel_values"], enc["image_grid_thw"])
        proposals, _, _ = region_proposals(vis["patch_map"], pcfg)

        comps = [_comp_to_1000(b, item["orig_size"]) for b in (item.get("component_bboxes") or [])]
        cand_boxes = [list(map(float, p["bbox_2d"])) for p in proposals]
        comp_max_iou = [max((_iou1000(c, g) for c in cand_boxes), default=0.0) for g in comps]
        cand_max_iou = [max((_iou1000(c, g) for g in comps), default=0.0) for c in cand_boxes]
        records.append(
            dict(
                id=item.get("id"),
                cls=item["class_name"],
                anom=bool(item["is_anomaly"]),
                n_comps=len(comps),
                n_cand=len(cand_boxes),
                cand_areas=[round((c[2] - c[0]) * (c[3] - c[1]) / 1e4, 3) for c in cand_boxes],  # % of image
                cand_peaks=[float(p.get("raw_peak", 0.0)) for p in proposals],
                comp_max_iou=[round(v, 4) for v in comp_max_iou],
                cand_max_iou=[round(v, 4) for v in cand_max_iou],
                cand_boxes=[[round(v, 2) for v in c] for c in cand_boxes],
                comp_boxes=[[round(v, 2) for v in g] for g in comps],
            )
        )
    return records


def _verb(rec: dict, t: float = PRIMARY_T) -> str:
    """Verify verb under the SFT target rules (center-rule hit, tight bar 0.5)."""
    from outcome.protocol_multibox import candidate_hits_comp

    if not rec["anom"]:
        return "none" if rec["n_cand"] == 0 else "reject"
    cands = rec.get("cand_boxes") or []
    comps = rec.get("comp_boxes") or []
    if not cands:
        return "discover_scratch"
    covered = [any(candidate_hits_comp(c, g, t) for c in cands) for g in comps]
    n_missed = sum(1 for ok in covered if not ok)
    n_spurious = sum(1 for c in cands if not any(candidate_hits_comp(c, g, t) for g in comps))
    if n_missed > 0:
        return "discover+refine"
    if n_spurious > 0:
        return "refine+prune"
    if all(v >= 0.5 for v in rec["comp_max_iou"]):
        return "keep"
    return "refine"


def _mean(xs) -> float:
    xs = [float(x) for x in xs]
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _quantiles(xs) -> Dict[str, float]:
    xs = np.asarray([float(x) for x in xs], dtype=float)
    if xs.size == 0:
        return {}
    return {f"p{q}": round(float(np.percentile(xs, q)), 3) for q in (10, 50, 90, 99)}


def aggregate(records: List[dict]) -> dict:
    normals = [r for r in records if not r["anom"]]
    anoms = [r for r in records if r["anom"]]

    def fire_rate(rs) -> float:
        return _mean(r["n_cand"] > 0 for r in rs)

    classes = sorted({r["cls"] for r in records})
    per_class = {}
    for cls in classes:
        rn = [r for r in normals if r["cls"] == cls]
        ra = [r for r in anoms if r["cls"] == cls]
        per_class[cls] = dict(
            n_normal=len(rn),
            normal_fire=fire_rate(rn),
            n_anom=len(ra),
            anom_fire=fire_rate(ra),
            anom_cov_t01=_mean(
                _mean([1.0 if v >= PRIMARY_T else 0.0 for v in r["comp_max_iou"]])
                for r in ra if r["n_comps"]
            ) if ra else float("nan"),
        )

    cov_at = {}
    for t in COVER_THRESHOLDS:
        per_comp = [1.0 if v >= t else 0.0 for r in anoms for v in r["comp_max_iou"]]
        full = [all(v >= t for v in r["comp_max_iou"]) for r in anoms if r["n_comps"]]
        cov_at[f"t={t}"] = dict(per_comp_recall=_mean(per_comp), frac_full_coverage=_mean(full))

    from outcome.protocol_multibox import candidate_hits_comp

    def _center_recall(rs, t=PRIMARY_T):
        vals = []
        for r in rs:
            comps = r.get("comp_boxes") or []
            cands = r.get("cand_boxes") or []
            vals.extend(1.0 if any(candidate_hits_comp(c, g, t) for c in cands) else 0.0
                        for g in comps)
        return _mean(vals)

    cov_at["center_rule_t=0.1"] = dict(
        per_comp_recall=_center_recall(anoms),
        frac_full_coverage=_mean(
            all(any(candidate_hits_comp(c, g, PRIMARY_T) for c in (r.get("cand_boxes") or []))
                for g in (r.get("comp_boxes") or []))
            for r in anoms if r["n_comps"]),
    )

    verbs: Dict[str, int] = {}
    for r in records:
        verbs[_verb(r)] = verbs.get(_verb(r), 0) + 1

    matched_iou = [v for r in anoms for v in r["comp_max_iou"]]
    multi = [r for r in anoms if r["n_comps"] >= 2]
    single = [r for r in anoms if r["n_comps"] == 1]

    return dict(
        n_total=len(records),
        n_normal=len(normals),
        n_anom=len(anoms),
        normal_fire_rate=fire_rate(normals),
        normal_cand_count_hist={str(k): sum(1 for r in normals if r["n_cand"] == k) for k in range(5)},
        anom_fire_rate=fire_rate(anoms),
        anom_cand_count_hist={str(k): sum(1 for r in anoms if r["n_cand"] == k) for k in range(5)},
        coverage=cov_at,
        matched_cand_gt_iou=dict(mean=_mean(matched_iou), **_quantiles(matched_iou)),
        cand_area_pct=_quantiles(a for r in records for a in r["cand_areas"]),
        verb_distribution=dict(sorted(verbs.items(), key=lambda kv: -kv[1])),
        multi_comp=dict(
            n=len(multi),
            per_comp_recall_t01=_mean(
                1.0 if v >= PRIMARY_T else 0.0 for r in multi for v in r["comp_max_iou"]
            ),
            frac_cand_ge_comps=_mean(r["n_cand"] >= r["n_comps"] for r in multi),
        ),
        single_comp=dict(
            n=len(single),
            recall_t01=_mean(
                1.0 if r["comp_max_iou"] and r["comp_max_iou"][0] >= PRIMARY_T else 0.0
                for r in single
            ),
        ),
        per_class=per_class,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="H prior candidate stats on VisA train (CPU)")
    p.add_argument("--config", type=str, default="configs/qwen35_4b_outcome_multibox.yaml")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--threads-per-worker", type=int, default=16)
    p.add_argument("--max-per-class", type=int, default=None, help="smoke-test cap per class")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    cfg = load_yaml_config(args.config)
    cfg.setdefault("runtime", {})["mode"] = "train"
    cfg.setdefault("distributed", {})["num_gpu"] = 1
    cfg["model"]["torch_dtype"] = "float32"  # CPU: fp32 GEMM is faster than bf16 here

    from data.scan import load_prior_split

    t0 = time.time()
    train_samples, _ = load_prior_split(cfg)
    print(f"[stats] scanned {len(train_samples)} train samples in {time.time() - t0:.1f}s", flush=True)

    if args.max_per_class is not None:
        # Stratified by class AND label: a class whose smoke subset ends up with a
        # single normal makes pick_ref_image fall back to the query image itself.
        by_cls: Dict[str, List[dict]] = {}
        for s in train_samples:
            by_cls.setdefault(str((s.get("metadata") or {}).get("class")), []).append(s)
        half = max(2, args.max_per_class // 2)
        train_samples = [
            s
            for cls in sorted(by_cls)
            for ss in (
                [x for x in by_cls[cls] if (x.get("metadata") or {}).get("anomaly")][:half],
                [x for x in by_cls[cls] if not (x.get("metadata") or {}).get("anomaly")][:half],
            )
            for s in ss
        ]
        print(f"[stats] smoke mode: {len(train_samples)} samples", flush=True)

    by_cls = {}
    for s in train_samples:
        by_cls.setdefault(str((s.get("metadata") or {}).get("class")), []).append(s)
    classes = sorted(by_cls)
    shards: List[List[dict]] = [[] for _ in range(min(args.workers, len(classes)))]
    for i, cls in enumerate(classes):
        shards[i % len(shards)].extend(by_cls[cls])

    out_dir = args.out or os.path.join(PROJECT_ROOT, "outputs", "prior_candidate_stats")
    os.makedirs(out_dir, exist_ok=True)
    records_path = os.path.join(out_dir, "visa_train_records.jsonl")

    t0 = time.time()
    records: List[dict] = []
    with ProcessPoolExecutor(
        max_workers=len(shards),
        initializer=_worker_init,
        initargs=(cfg, args.threads_per_worker),
    ) as ex:
        for i, shard_records in enumerate(ex.map(_run_shard, shards)):
            records.extend(shard_records)
            done_cls = {r["cls"] for r in records}
            print(
                f"[stats] shard {i + 1}/{len(shards)} done: {len(records)} records "
                f"({time.time() - t0:.1f}s, classes so far: {len(done_cls)})",
                flush=True,
            )

    with open(records_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = aggregate(records)
    summary["config"] = args.config
    summary["elapsed_s"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, "visa_train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    s = summary
    print("\n" + "=" * 78, flush=True)
    print("PRIOR CANDIDATE STATS (VisA train)", flush=True)
    print("=" * 78, flush=True)
    print(f"samples: {s['n_total']}  (normal {s['n_normal']}, anomaly {s['n_anom']})", flush=True)
    print(f"NORMAL  fire rate       = {s['normal_fire_rate']:.3f}   cand hist {s['normal_cand_count_hist']}", flush=True)
    print(f"ANOMALY fire rate       = {s['anom_fire_rate']:.3f}   cand hist {s['anom_cand_count_hist']}", flush=True)
    for t, d in s["coverage"].items():
        print(f"  coverage {t}: per-comp recall {d['per_comp_recall']:.3f}  full-coverage {d['frac_full_coverage']:.3f}", flush=True)
    mi = s["matched_cand_gt_iou"]
    print(f"  matched cand-GT IoU   = {mi['mean']:.3f}  ({ {k: v for k, v in mi.items() if k != 'mean'} })", flush=True)
    print(f"  candidate area %      = {s['cand_area_pct']}", flush=True)
    print(f"  verb distribution     = {s['verb_distribution']}", flush=True)
    mc = s["multi_comp"]
    print(f"  multi-comp (n={mc['n']}): recall@0.1 {mc['per_comp_recall_t01']:.3f}  P(cand>=comps) {mc['frac_cand_ge_comps']:.3f}", flush=True)
    print("  per-class:", flush=True)
    for cls, d in s["per_class"].items():
        print(
            f"    {cls:<12} normal_fire={d['normal_fire']:.3f} (n={d['n_normal']})  "
            f"anom_fire={d['anom_fire']:.3f} (n={d['n_anom']})  anom_cov@0.1={d['anom_cov_t01']:.3f}",
            flush=True,
        )
    print("=" * 78, flush=True)
    print(f"saved: {records_path} + visa_train_summary.json", flush=True)


if __name__ == "__main__":
    main()
