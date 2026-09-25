#!/usr/bin/env python3
"""Controlled extent-probe: isolate the crop visual signal (no fine-tuning).

The end-to-end two-stage rollout confounds several things (the model's own
[localize] quality, [imagine]/[confirm] continuation stability, OOD 3-image
prompt). This probe removes those confounds by FIXING the candidate box and asking
a single narrow question: given a candidate box that undershoots / overshoots the
true defect, does seeing a zoomed, outline-drawn crop of that box let the model
correct the extent better than the global 2-image view alone?

For each anomaly sample we build a perturbed candidate (e.g. GT shrunk to 50%) and
render it two ways:
  A) global-only:  ref + test (2 images) + candidate coords in text
  B) global+crop:  ref + test + crop(3 images, box outline drawn) + same text
The model outputs a corrected box; we measure its IoU against GT and how far it
moved from the (wrong) candidate toward GT.

Usage:
  python scripts/eval_zoom_probe.py --config configs/qwen35_2b_world_model_rl.yaml \
      --sft-adapter outputs/train/world_model_sft_2b_hvpt_768 --n 30
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict

import torch
from PIL import Image

sys.path.insert(0, '.')

from outcome.engine_multibox import datasets, load_model
from outcome.policy import generate_group
from outcome.protocol import to_pixels
from outcome.zoom_crop import make_zoom_crop
from outcome.inputs import build_zoom_batch
from rl.grpo import move_batch
from utils.common import set_seed
from utils.config import load_yaml_config


def _to_1000(box_px, orig):
    w, h = orig
    return [round(box_px[0] / w * 1000, 2), round(box_px[1] / h * 1000, 2),
            round(box_px[2] / w * 1000, 2), round(box_px[3] / h * 1000, 2)]


def _shrink(box_px, scale):
    cx = (box_px[0] + box_px[2]) / 2.0
    cy = (box_px[1] + box_px[3]) / 2.0
    w = (box_px[2] - box_px[0]) * scale
    h = (box_px[3] - box_px[1]) * scale
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


_PROMPT_A = (
    "Image 1 is a defect-free reference. Image 2 is the inspection image. "
    "A candidate anomaly box is given as {cand}. Judge whether this box accurately "
    "bounds the defect. If it undershoots, overshoots, or is shifted, output the "
    "corrected box; otherwise output the same box. Reply with exactly "
    '<answer>{"bboxes_2d": [[x1,y1,x2,y2]]}</answer> using integer [0,1000] coords '
    "on Image 2."
)

_PROMPT_B = (
    "Image 1 is a defect-free reference. Image 2 is the inspection image. Image 3 is "
    "a zoomed crop around a candidate anomaly box; the red rectangle is that candidate "
    "box ({cand}). Look closely at Image 3 to see whether the box undershoots, "
    "overshoots, or is shifted relative to the true defect boundary. Output the "
    "corrected box (or the same box if it is right). Reply with exactly "
    '<answer>{"bboxes_2d": [[x1,y1,x2,y2]]}</answer> using integer [0,1000] coords '
    "on Image 2."
)


def _build_2img_batch(processor, prior, cfg, ref_img, test_img, text, device):
    from outcome.inputs import _smart_resize_image
    from models.qwen35 import qwen_vision_factor
    data = cfg.get('data') or {}
    max_size = int(data.get('max_image_size', 768))
    factor = qwen_vision_factor(processor, getattr(prior, 'visual', None))
    cap = max_size * max_size
    test_rs = _smart_resize_image(test_img, max_size, factor, 256 * 256, cap)
    ref_rs = ref_img.resize(test_rs.size, Image.Resampling.BICUBIC)
    from outcome.inputs import encode_visual_merged, apply_chat_template_safe
    img_proc = getattr(processor, 'image_processor', None)
    enc = img_proc(images=[ref_rs, test_rs], return_tensors='pt')
    vis = encode_visual_merged(prior, enc['pixel_values'].to(device), enc['image_grid_thw'].to(device))
    user = dict(role='user', content=[dict(type='image', image=ref_rs),
                                      dict(type='image', image=test_rs),
                                      dict(type='text', text=text)])
    enable_thinking = bool((cfg.get('prompt') or {}).get('enable_thinking', False))
    rendered = apply_chat_template_safe(processor, [user], True, enable_thinking)
    full = processor(text=[rendered], images=[ref_rs, test_rs], return_tensors='pt', truncation=False)
    full['image_embeds'] = vis
    full['prompt_len'] = torch.tensor([full['input_ids'].shape[-1]])
    full['box_token_id'] = -1
    full['n_box_tokens'] = 0
    full['control_token_id'] = -1
    full['feat_token_id'] = -1
    full['n_feat_tokens'] = 0
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/qwen35_2b_world_model_rl.yaml')
    ap.add_argument('--sft-adapter', default='outputs/train/world_model_sft_2b_hvpt_768')
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--shrink', type=float, default=0.5, help='candidate = GT * shrink')
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    cfg['outcome']['sft_adapter'] = args.sft_adapter
    set_seed(int(cfg['training']['seed']))

    model, processor, prior = load_model(cfg, adapter=None, fresh_lora=False)
    device = next(model.parameters()).device
    _, _, test_set = datasets(cfg, processor)

    # Sample anomaly items, biased to small/medium (the extent bottleneck).
    anom = [test_set[i] for i in range(len(test_set)) if test_set[i].get('is_anomaly')]
    rng = random.Random(7)
    anom = rng.sample(anom, min(args.n, len(anom)))

    rows = []
    for item in anom:
        meta = item
        orig = meta['orig_size']
        gt_px = meta.get('gt_box_px')
        comps = meta.get('component_bboxes') or ([gt_px] if gt_px else [])
        if not comps:
            continue
        gt_px = comps[0] if not gt_px else gt_px
        cand_px = _shrink(gt_px, args.shrink)
        cand = _to_1000(cand_px, orig)

        ref_img = meta['ref']
        test_img = meta['test']
        crop = make_zoom_crop(test_img, cand, orig_size=orig, expand=1.0, min_pad_frac=0.12, max_area_frac=0.6)

        cand_s = str([int(round(c)) for c in cand])
        text_a = _PROMPT_A.replace('{cand}', cand_s)
        text_b = _PROMPT_B.replace('{cand}', cand_s)

        batch_a = _build_2img_batch(processor, prior, cfg, ref_img, test_img, text_a, device)
        batch_a = move_batch(batch_a, device)
        out_a = generate_group(model, processor, batch_a, cfg, group=1, sample=False)[0].text

        if crop.degenerate:
            # A whole-image window is just the global view again; skip crop condition.
            box_b = None
        else:
            batch_b = build_zoom_batch(processor, prior, cfg, ref_img, test_img, crop.image, text_b, device)
            batch_b = move_batch(batch_b, device)
            out_b = generate_group(model, processor, batch_b, cfg, group=1, sample=False)[0].text
            box_b = _parse_box(out_b)

        box_a = _parse_box(out_a)
        iou_a = _iou(to_pixels(box_a, orig), gt_px) if box_a else 0.0
        iou_b = _iou(to_pixels(box_b, orig), gt_px) if box_b else None
        cand_iou = _iou(cand_px, gt_px)

        rows.append(dict(class_name=meta['class_name'], defect=meta.get('defect_type'),
                         cand_iou=cand_iou, iou_a=iou_a, iou_b=iou_b,
                         size=_size(meta)))

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else float('nan')

    def delta_b(r):
        return r['iou_b'] - r['cand_iou'] if r['iou_b'] is not None else None

    print('\n==================== CONTROLLED EXTENT PROBE ====================')
    print(f"candidate = GT x {args.shrink} (simulated undershoot), n={len(rows)}")
    print(f"candidate baseline IoU vs GT: {mean(r['cand_iou'] for r in rows):.4f}")
    print(f"{'condition':<18}{'mean IoU vs GT':>16}{'delta vs cand':>16}")
    print(f"{'global-only (A)':<18}{mean(r['iou_a'] for r in rows):>16.4f}"
          f"{mean(r['iou_a']-r['cand_iou'] for r in rows):>+16.4f}")
    print(f"{'global+crop  (B)':<18}{mean(r['iou_b'] for r in rows):>16.4f}"
          f"{mean(delta_b(r) for r in rows):>+16.4f}")

    better = sum(1 for r in rows if r['iou_b'] is not None and r['iou_b'] > r['iou_a'] + 1e-3)
    worse = sum(1 for r in rows if r['iou_b'] is not None and r['iou_b'] < r['iou_a'] - 1e-3)
    n_comp = sum(1 for r in rows if r['iou_b'] is not None)
    print(f"\ncrop better than global: {better}/{n_comp}   worse: {worse}/{n_comp}")

    print("\n--- per-sample (A -> B) ---")
    for r in sorted(rows, key=lambda r: -(0 if r['iou_b'] is None else r['iou_b'] - r['iou_a'])):
        b = f"{r['iou_b']:.3f}" if r['iou_b'] is not None else '  -- '
        print(f"  {r['class_name']:<16} {str(r['defect']):<22} {str(r['size']):<7} "
              f"cand={r['cand_iou']:.3f}  A={r['iou_a']:.3f}  B={b}")
    print('==================== END ====================')


def _parse_box(text):
    """Parse the first [x1,y1,x2,y2] list following bboxes_2d / candidate_bboxes_2d."""
    import re
    m = re.search(r'(?:bboxes_2d|candidate_bboxes_2d)\s*[=:]\s*(\[\[[^\]]*\]\])', text, re.I)
    if m:
        try:
            import json
            arr = json.loads(m.group(1))
            if arr and len(arr[0]) == 4:
                return [float(v) for v in arr[0]]
        except (ValueError, IndexError, TypeError):
            pass
    # Fallback: first 4-number bracket anywhere.
    for g in re.findall(r'\[\s*([0-9.,\s-]+)\]', text):
        parts = [p.strip() for p in g.split(',')]
        if len(parts) == 4:
            try:
                return [float(p) for p in parts]
            except ValueError:
                continue
    return None


def _size(meta):
    frac = meta.get('mask_area_fraction')
    if frac is None:
        gt = meta.get('gt_box_px')
        frac = (gt[2]-gt[0])*(gt[3]-gt[1])/(meta['orig_size'][0]*meta['orig_size'][1]) if gt else 0.0
    return 'small' if frac < .02 else 'medium' if frac < .1 else 'large'


if __name__ == '__main__':
    main()
