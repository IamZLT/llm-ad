#!/usr/bin/env python3
"""Offline H-prior resolution ablation: H candidate best-IoU @448 vs @768 (no LLM forward).

The H-prior's localization ceiling is bounded by how finely its patch map samples the
inspection image (patch_size * merge_size = 32 native px per cell). Raising
``data.max_image_size`` shrinks each cell in *original-image* pixels. This script loads
only the frozen vision tower + AnomalyPrior, encodes each anomaly twice (448 / 768), and
reports how well the H candidate boxes bound the ground-truth components.

Usage:
  python scripts/prior_resolution_ablation.py --config configs/qwen35_2b_world_model_rl.yaml
  python scripts/prior_resolution_ablation.py --max-samples 30   # smoke
"""

from __future__ import annotations

import argparse
import gc
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.config import load_yaml_config
from outcome.protocol import to_pixels


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen35_2b_world_model_rl.yaml")
    ap.add_argument("--sizes", default="448,768")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    cfg.setdefault("runtime", {})["mode"] = "train"
    cfg.setdefault("distributed", {})["num_gpu"] = 1

    from models.qwen35 import setup_model_and_processor, apply_processor_geometry
    from models.anomaly_prior import AnomalyPrior

    model, processor = setup_model_and_processor(cfg, for_inference=True, freeze_vision=True)
    device = torch.device(args.device)
    model = model.to(device)
    prior = AnomalyPrior.from_qwen(model, cfg)
    del model
    gc.collect()

    from data.scan import load_prior_split
    from data.prior_dataset import build_train_ref_pool
    from outcome.inputs import (
        OutcomeCollator,
        OutcomeDataset,
        encode_pair_canonical,
        region_proposals,
    )

    _, evals = load_prior_split(cfg)
    pool = build_train_ref_pool(evals)
    ds = OutcomeDataset(evals, cfg, processor, mode="eval", ref_pool=pool)
    collator = OutcomeCollator(processor, prior, cfg)
    pcfg = (cfg.get("outcome") or {}).get("prior") or {}

    anom_items = []
    for i in range(len(ds)):
        it = ds[i]
        if it.get("is_anomaly"):
            anom_items.append(it)
            if args.max_samples and len(anom_items) >= args.max_samples:
                break
    print(f"[ablation] {len(anom_items)} anomaly samples", flush=True)

    for max_size in [int(s) for s in args.sizes.split(",")]:
        cfg["data"]["max_image_size"] = max_size
        apply_processor_geometry(processor, cfg, prior.visual)
        bests, cells, ncands = [], [], []
        for it in anom_items:
            ref_g, test_g = collator._align_pair_at(it["ref"], it["test"], max_size)
            enc = collator._concat_image_tensors(ref_g, test_g)
            vis = encode_pair_canonical(
                prior, enc["pixel_values"].to(device), enc["image_grid_thw"].to(device)
            )
            proposals, _, _ = region_proposals(vis["patch_map"], pcfg)
            comps = it.get("component_bboxes") or []
            if not comps and it.get("gt_box_px"):
                comps = [it["gt_box_px"]]
            if not comps:
                continue
            orig = it["orig_size"]
            cand_px = [to_pixels([float(v) for v in p["bbox_2d"]], orig) for p in proposals]
            if not cand_px:
                continue
            b = max(max(iou(c, g) for c in cand_px) for g in comps)
            h, w = vis["patch_map"].shape
            cell = max(orig[0] / w, orig[1] / h)
            bests.append(b)
            cells.append(cell)
            ncands.append(len(proposals))

        bests = np.asarray(bests)
        print(f"\n=== max_image_size={max_size} ===", flush=True)
        print(
            f"  n={len(bests)}  H best-IoU: mean={bests.mean():.4f} "
            f"p50={np.percentile(bests, 50):.4f} p90={np.percentile(bests, 90):.4f} "
            f"min={bests.min():.4f}",
            flush=True,
        )
        print(
            f"  mean patch-cell (orig px)={np.mean(cells):.1f}  "
            f"mean n_candidates={np.mean(ncands):.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
