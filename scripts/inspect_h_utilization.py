#!/usr/bin/env python3
"""Inspect how H is currently produced and consumed, and verify it is correct.

Pipeline (matches outcome/inputs.py exactly):
    (ref, test) --align@768--> pair -> encode_pair_canonical (hook blocks 12/16/20/24)
        -> per-layer NN distance map -> softmax_fuse -> H map [Ht, Wt]
    H -> region_proposals (relative threshold 0.7 -> 4-connected components)
        -> proposal masks + meta (bbox_2d/peak_2d in [0,1000], cell-index based)

This script does NOT touch the LLM / RegionAdapter / router. It only checks the
frozen H front-end:

    1. H dynamic range + threshold (is the map non-flat?)
    2. For anomaly samples: does H's connected-component proposal hit GT?
       (best proposal IoU, peak-in-GT, H-peak vs GT-center distance)
    3. For normal samples: how many false-alarm proposals does H produce?
    4. Coordinate alignment: draw proposal boxes / peak over the ORIGINAL test image
       to eyeball whether H's patch grid maps back to the image correctly.

Outputs:
    <out-dir>/summary.json          aggregate H-quality numbers
    <out-dir>/<idx>_<class>_<defect>.png  4-panel viz (ref / test / H / overlay+boxes)
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw

from utils.config import load_yaml_config

RED = (220, 40, 40)
GREEN = (30, 180, 60)
BLUE = (40, 80, 220)


def _to_px(box1000, size):
    w, h = size
    return [box1000[0] / 1000.0 * w, box1000[1] / 1000.0 * h,
            box1000[2] / 1000.0 * w, box1000[3] / 1000.0 * h]


def _iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(x1 - x0, 0) * max(y1 - y0, 0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def compute_h(collator, item, h_size):
    """Reproduce the H front-end of OutcomeCollator.__call__ without prompt/tokenize."""
    import torch

    from outcome.inputs import encode_pair_canonical, pixel_budget

    device = collator._device()
    with pixel_budget(collator.processor, h_size):
        ref_h, test_h = collator._align_pair_at(item["ref"], item["test"], h_size)
        h_in = collator._concat_image_tensors(ref_h, test_h)
    vis_h = encode_pair_canonical(
        collator.prior, h_in["pixel_values"].to(device), h_in["image_grid_thw"].to(device)
    )
    return vis_h["patch_map"], test_h.size  # [Ht, Wt], (W, H) of resized test


def draw_boxes(img, boxes_px, color, width=4, label=None):
    d = ImageDraw.Draw(img)
    for b in boxes_px:
        d.rectangle([b[0], b[1], b[2], b[3]], outline=color, width=width)
    return img


def render_panel(item, hmap, proposals, gt_boxes_px, orig_size):
    """4-panel: ref | test | H heatmap | overlay(H+proposals+GT+peak)."""
    from models.anomaly_prior import heatmap_to_pil

    ref = item["ref"].convert("RGB").resize(orig_size, Image.Resampling.BILINEAR)
    test = item["test"].convert("RGB")
    heat = heatmap_to_pil(hmap, orig_size).resize(orig_size, Image.Resampling.BILINEAR)
    overlay = Image.blend(test.copy(), heat, 0.55)
    d = ImageDraw.Draw(overlay)
    for p in proposals:
        b = _to_px(p["bbox_2d"], orig_size)
        d.rectangle([b[0], b[1], b[2], b[3]], outline=RED, width=5)
        pkx = p["peak_2d"][0] / 1000.0 * orig_size[0]
        pky = p["peak_2d"][1] / 1000.0 * orig_size[1]
        r = 7
        d.ellipse([pkx - r, pky - r, pkx + r, pky + r], outline=(0, 0, 0), width=3)
    for g in gt_boxes_px:
        d.rectangle([g[0], g[1], g[2], g[3]], outline=GREEN, width=5)

    panels = [ref, test, heat, overlay]
    w, h = panels[0].size
    pad = 8
    canvas = Image.new("RGB", (2 * w + 3 * pad, 2 * h + 3 * pad), (20, 20, 20))
    labels = ["ref", "test", "H heatmap", "overlay (red=proposal green=GT)"]
    for i, (p, lab) in enumerate(zip(panels, labels)):
        x = pad + (i % 2) * (w + pad)
        y = pad + (i // 2) * (h + pad)
        canvas.paste(p, (x, y))
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/qwen35_2b_annos_probe_v2_fsm_rl.yaml")
    ap.add_argument("--h-size", type=int, default=768)
    ap.add_argument("--region-feature-index", type=int, default=23)
    ap.add_argument("--fusion-mode", default=None, choices=["softmax", "mean", "harmonic"])
    ap.add_argument("--normalize-layers", action="store_true", default=None)
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    ap.add_argument("--num-anomaly", type=int, default=8)
    ap.add_argument("--num-normal", type=int, default=4)
    ap.add_argument("--prefer-small", action="store_true",
                    help="bias anomaly selection toward small defects (mask_area_fraction < 0.02)")
    ap.add_argument("--out-dir", default="outputs/eval/h_inspect")
    args = ap.parse_args()

    cfg = copy.deepcopy(load_yaml_config(args.config))
    oc = cfg.setdefault("outcome", {})
    oc.setdefault("prior", {})["h_image_size"] = int(args.h_size)
    oc["prior"]["region_feature_index"] = int(args.region_feature_index)

    import torch

    from outcome.engine_multibox import datasets, load_model
    from outcome.inputs import OutcomeCollator, region_proposals
    from outcome.inputs_multibox import OutcomeMultiboxCollator
    from utils.common import set_seed

    set_seed(int(cfg["training"]["seed"]))
    model, processor, prior = load_model(cfg, None, fresh_lora=False)
    model.eval()
    if args.fusion_mode is not None:
        prior.fusion_mode = args.fusion_mode
        if hasattr(prior, "normalize_layers"):
            prior.normalize_layers = bool(args.normalize_layers)
    _, dev_set, test_set = datasets(cfg, processor)
    sel = test_set if args.split == "test" else dev_set

    collator_cls = OutcomeMultiboxCollator if (cfg["outcome"].get("version") == "outcome-multibox-v1") else OutcomeCollator
    collator = collator_cls(processor, prior, cfg)
    pcfg = dict(cfg["outcome"]["prior"])

    # ---- select samples ----
    anom_idx, norm_idx = [], []
    for i, s in enumerate(sel.samples):
        meta = s.get("metadata") or {}
        if meta.get("anomaly"):
            anom_idx.append(i)
        else:
            norm_idx.append(i)
    if args.prefer_small:
        def frac_key(i):
            m = (sel.samples[i].get("metadata") or {})
            return m.get("mask_area_fraction") or 1.0
        anom_idx.sort(key=frac_key)
    chosen = anom_idx[: args.num_anomaly] + norm_idx[: args.num_normal]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for n, idx in enumerate(chosen):
        item = sel[idx]
        hmap, test_rs_size = compute_h(collator, item, args.h_size)
        proposals, masks, mode = region_proposals(hmap, pcfg)

        h_arr = hmap.detach().float().cpu().numpy()
        hmin, hmax = float(h_arr.min()), float(h_arr.max())
        thr = hmin + float(pcfg.get("relative_threshold", 0.7)) * (hmax - hmin)

        gt = item.get("gt_box_px")
        comps = list(item.get("component_bboxes") or [])
        if not comps and gt:
            comps = [gt]
        orig = item["orig_size"]

        best_iou = None
        peak_in_gt = False
        if item["is_anomaly"] and comps:
            best_iou = max((_iou(_to_px(p["bbox_2d"], orig), g) for p in proposals for g in comps), default=0.0)
            for p in proposals:
                pkx = p["peak_2d"][0] / 1000.0 * orig[0]
                pky = p["peak_2d"][1] / 1000.0 * orig[1]
                for g in comps:
                    if g[0] <= pkx <= g[2] and g[1] <= pky <= g[3]:
                        peak_in_gt = True

        rec = dict(
            idx=n, class_name=item["class_name"], defect=item["defect_type"],
            is_anomaly=bool(item["is_anomaly"]),
            area_frac=item.get("mask_area_fraction"),
            H_shape=list(h_arr.shape), H_min=hmin, H_max=hmax, H_range=hmax - hmin,
            threshold=thr, n_proposals=len(proposals),
            proposals=[dict(bbox=p["bbox_2d"], peak=p["peak_2d"], area_frac=p["area_fraction"])
                       for p in proposals],
            gt_boxes=comps,
        )
        if item["is_anomaly"]:
            rec["best_proposal_iou"] = best_iou
            rec["peak_in_gt"] = peak_in_gt
        rows.append(rec)

        print(f"[{n}] {item['class_name']}/{item['defect_type']} anom={item['is_anomaly']} "
              f"area={item.get('mask_area_fraction')} H=[{hmin:.3f},{hmax:.3f}] range={hmax-hmin:.4f} "
              f"n_prop={len(proposals)}" + (f" bestIoU={best_iou:.3f} peakInGT={peak_in_gt}" if item["is_anomaly"] else ""),
              flush=True)

        gt_px = comps if item["is_anomaly"] else []
        panel = render_panel(item, hmap, proposals, gt_px, orig)
        panel.save(out_dir / f"{n:02d}_{item['class_name']}_{item['defect_type']}.png")

    # ---- aggregate ----
    anom = [r for r in rows if r["is_anomaly"]]
    norm = [r for r in rows if not r["is_anomaly"]]
    def mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else None

    summary = dict(
        n_anomaly=len(anom), n_normal=len(norm),
        H_mean_range=mean(r["H_range"] for r in rows),
        n_flat_H=sum(1 for r in rows if r["H_range"] < 1e-6),
        anomaly_best_iou_mean=mean(r["best_proposal_iou"] for r in anom),
        anomaly_recall_at_01=mean(r["best_proposal_iou"] >= 0.1 for r in anom),
        anomaly_recall_at_03=mean(r["best_proposal_iou"] >= 0.3 for r in anom),
        anomaly_peak_in_gt_rate=mean(r["peak_in_gt"] for r in anom),
        normal_mean_n_proposals=mean(r["n_proposals"] for r in norm),
        normal_fp_rate=mean(r["n_proposals"] > 0 for r in norm),
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (out_dir / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str))

    print("\n" + "=" * 78)
    print("H INSPECTION SUMMARY")
    print("=" * 78)
    for k, v in summary.items():
        print(f"  {k:28s} {v}")
    print("=" * 78)
    print(f"\npanels -> {out_dir}/*.png")
    print(f"rows   -> {out_dir}/rows.json")
    print(f"summary-> {out_dir}/summary.json")


if __name__ == "__main__":
    main()
