#!/usr/bin/env python3
"""Outcome-multibox SFT (world-model style, single-pass internal thinking).

Trains a language LoRA on the *full* autoregressive thought chain
``[understand]...[confirm]</think><answer>`` with no stage-marker masking, so the
model learns to write its own internal reasoning in one decode (the Qwen-AgentWorld /
RLVR-World recipe: SFT supervises the complete CoT token-by-token).

The H anomaly heatmap is reduced to a static search hint rendered once at the top
of the prompt (connected-component candidate boxes). It is never the training
target: ``[localize]`` is always supervised on GT boxes. The confirm stage is taught
*all five* actions — keep / refine / reject / discover / none — including
reject→relocalize and refine→refine multi-round intermediate states (P0), so the
cold-start policy already knows how to correct a bad localization instead of only
emitting ``keep``.

After this run, point ``outcome.sft_adapter`` at the saved directory in the RL config.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "7200")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from data.prior_dataset import build_train_ref_pool
from data.scan import load_prior_split, split_holdout_by_class
from models.anomaly_prior import AnomalyPrior
from models.h_box_prior import (build_h_box_prior, load_h_box_prior, save_h_box_prior,
                                ensure_box_token, bind_h_box, compute_g_proj)
from models.h_vpt import (HVPT, build_h_vpt, load_h_vpt, save_h_vpt,
                          ensure_control_token, compute_h_H, bind_h_vpt)
from models.lora import apply_lora
from models.qwen35 import (setup_model_and_processor, freeze_vision_encoder, force_vision_eval,
                           unwrap_model, print_trainable_params)
from models.vision_cache import bind_cached_image_features
from outcome.inputs import (OutcomeCollator, OutcomeDataset, build_observation_batch,
                            build_zoom_train_batch)
from outcome.inputs_multibox import OutcomeMultiboxCollator, OutcomeMultiboxDataset
from outcome.observation import observation_prompt, plan_observations
from outcome.policy import generate_group
from outcome.protocol import iou, to_pixels, valid_box
from outcome.protocol_multibox import (objective_verdict, parse_output_cfg, refine_verdict,
                                       score_output, set_localization_reward)
from outcome.thinking import (THINK_CLOSE_PREFIX, selective_sft_encoding,
                              staged_sft_target, thinking_enabled)
from rl.grpo import forward_with_vision, model_inputs, move_batch
from utils.common import is_main_process, set_seed
from utils.config import load_yaml_config
from visualization.tensorboard import start_tensorboard


class _NullWriter:
    def add_scalar(self, *a, **k): pass
    def add_text(self, *a, **k): pass
    def add_image(self, *a, **k): pass
    def add_hparams(self, *a, **k): pass
    def flush(self): pass
    def close(self): pass


def _archive_legacy_events(tb_root: Path) -> None:
    """Move event files from previous runs at the logdir root into run_legacy/."""
    if not tb_root.is_dir():
        return
    legacy = [p for p in tb_root.iterdir() if p.is_file() and p.name.startswith('events.')]
    if not legacy:
        return
    dst = tb_root / 'run_legacy'
    dst.mkdir(exist_ok=True)
    for p in legacy:
        p.rename(dst / p.name)


def _font(size=14):
    for path in (
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _pil_to_tb(image):
    return np.asarray(image.convert('RGB')).transpose(2, 0, 1)


def _scale_box(box, orig_wh, dest_wh):
    ow, oh = float(orig_wh[0]), float(orig_wh[1])
    dw, dh = float(dest_wh[0]), float(dest_wh[1])
    return [box[0] * dw / ow, box[1] * dh / oh, box[2] * dw / ow, box[3] * dh / oh]


def _draw_gt_boxes(test, meta):
    im = test.copy().convert('RGB')
    draw = ImageDraw.Draw(im)
    font = _font(14)
    orig = tuple(meta.get('orig_size') or im.size)
    for box in meta.get('component_bboxes') or []:
        xy = _scale_box(box, orig, im.size)
        draw.rectangle(xy, outline=(0, 220, 0), width=3)
        draw.text((xy[0] + 3, max(0, xy[1] - 16)), 'GT', fill=(0, 220, 0), font=font)
    union = meta.get('gt_box_px')
    if union is not None:
        xy = _scale_box(union, orig, im.size)
        draw.rectangle(xy, outline=(120, 200, 120), width=2)
    return im


def _draw_pred_boxes(test, meta, parsed):
    """GT (green) + model output: final bboxes_2d (red) + [localize] candidate (orange)."""
    im = _draw_gt_boxes(test, meta)
    draw = ImageDraw.Draw(im)
    font = _font(14)
    orig = tuple(meta.get('orig_size') or im.size)
    for box in parsed.get('candidate_bboxes_2d') or []:
        px = to_pixels(box, orig)
        xy = _scale_box(px, orig, im.size)
        draw.rectangle(xy, outline=(255, 170, 0), width=2)
        draw.text((xy[0] + 3, max(0, xy[1] - 16)), 'CAND', fill=(255, 170, 0), font=font)
    for box in parsed.get('bboxes_2d') or []:
        px = to_pixels(box, orig)
        xy = _scale_box(px, orig, im.size)
        draw.rectangle(xy, outline=(255, 60, 60), width=3)
        draw.text((xy[0] + 3, xy[3] + 2), 'PRED', fill=(255, 60, 60), font=font)
    return im


def _mask_overlay(test, mask_path, alpha=0.55):
    if not mask_path or not os.path.exists(mask_path):
        return None
    try:
        mask = Image.open(mask_path).convert('L').resize(test.size, Image.Resampling.NEAREST)
    except Exception:
        return None
    arr = np.array(mask) > 0
    if not arr.any():
        return None
    red = Image.new('RGB', test.size, (220, 30, 30))
    mask_alpha = Image.fromarray((arr * int(255 * alpha)).astype(np.uint8))
    return Image.composite(red, test.convert('RGB'), mask_alpha)


def log_sft_hparams(writer, cfg, args, world):
    hparams = {
        'arch': str(cfg.get('model', {}).get('arch')),
        'lr': float(args.lr),
        'batch_size': int(args.batch_size),
        'num_gpu': int(world),
        'accum': int(args.accum),
        'epochs': int(args.epochs),
        'max_grad_norm': float(args.max_grad_norm),
        'lora_r': int((cfg.get('lora') or {}).get('r', 0)),
        'lora_alpha': int((cfg.get('lora') or {}).get('alpha', 0)),
        'max_length': int(cfg.get('training', {}).get('max_length', 0)),
        'seed': int(cfg.get('training', {}).get('seed', 0)),
    }
    writer.add_text('sft/0_hparams', json.dumps(hparams, indent=2), 0)
    writer.add_text('sft/0_config', json.dumps({
        'model': cfg.get('model'), 'lora': cfg.get('lora'), 'training': cfg.get('training'),
        'grpo': cfg.get('grpo'), 'outcome': cfg.get('outcome'), 'tensorboard': cfg.get('tensorboard'),
    }, ensure_ascii=False, indent=2, default=str), 0)
    for name, value in hparams.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f'hparams/{name}', value, 0)
    writer.flush()


def log_sft_case(writer, step, meta, target, overlay_alpha=0.45):
    ref = meta.get('ref')
    test = meta.get('test')
    heat = meta.get('heatmap')
    if ref is not None and test is not None and heat is not None:
        heat_rgb = heat.convert('RGB').resize(test.size, Image.Resampling.BILINEAR)
        overlay = Image.blend(test.convert('RGB'), heat_rgb, overlay_alpha)
        writer.add_image('train_case/1_heatmap', _pil_to_tb(overlay), step)
        writer.add_image('train_case/2_ref', _pil_to_tb(ref), step)
    if test is not None:
        writer.add_image('train_case/3_gt_box', _pil_to_tb(_draw_gt_boxes(test, meta)), step)
        mask = _mask_overlay(test, meta.get('full_mask_path'))
        if mask is not None:
            writer.add_image('train_case/4_gt_mask', _pil_to_tb(mask), step)
    text = (
        f"step={step}\n"
        f"class={meta.get('class_name')} is_anomaly={meta.get('is_anomaly')} "
        f"defect={meta.get('defect_type')}\n"
        f"image={meta.get('image_path')}\n"
        f"prompt_tokens={meta.get('prompt_tokens')} visual_tokens={meta.get('visual_tokens')}\n\n"
        f"{target}"
    )
    writer.add_text('train_case/5_target', text, step)
    writer.flush()


def _bbox_to_1000(gt_px, orig_size):
    w, h = float(orig_size[0]), float(orig_size[1])
    return [round(gt_px[0] * 1000.0 / w, 3), round(gt_px[1] * 1000.0 / h, 3),
            round(gt_px[2] * 1000.0 / w, 3), round(gt_px[3] * 1000.0 / h, 3)]


def _boxes_to_1000(boxes_px, orig_size):
    return [_bbox_to_1000(b, orig_size) for b in boxes_px]


def _region_count(n: int) -> str:
    return 'one region' if int(n) == 1 else f'{int(n)} regions'


def _plural(n: int, singular: str, plural: str) -> str:
    return singular if int(n) == 1 else plural


def _box_where(box_1000) -> str:
    """Coarse image-frame location from a 0-1000 box. Not a defect-type label."""
    x1, y1, x2, y2 = [float(v) for v in box_1000]
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / 1e6
    if area >= 0.40:
        return 'across a large area'
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    left, right = 1000.0 / 3.0, 2000.0 / 3.0
    if cx < left:
        horiz = 'left'
    elif cx > right:
        horiz = 'right'
    else:
        horiz = None
    if cy < left:
        vert = 'upper'
    elif cy > right:
        vert = 'lower'
    else:
        vert = None
    if vert is None and horiz is None:
        return 'near the center'
    if vert is None:
        return f'on the {horiz}'
    if horiz is None:
        return f'in the {vert} portion'
    return f'in the {vert} {horiz}'


def _where_join(boxes_1000) -> str:
    """Deduped location phrase for one or more boxes, e.g. 'in the upper left'."""
    seen = []
    for box in boxes_1000:
        phrase = _box_where(box)
        if phrase not in seen:
            seen.append(phrase)
    if not seen:
        return 'somewhere on the object'
    if len(seen) == 1:
        return seen[0]
    if len(seen) == 2:
        return f'{seen[0]} and {seen[1]}'
    return ', '.join(seen[:-1]) + f', and {seen[-1]}'


def _ground_multibox(boxes: list = None) -> str:
    """Localize-stage candidate boxes (0-1000) with a coarse location phrase."""
    boxes = boxes or []
    if not boxes:
        return 'candidate_bboxes_2d=[]'
    return f'candidate_bboxes_2d={boxes}; {_where_join(boxes)}'


def _shrink_box(box, scale: float) -> list:
    """Shrink a 0-1000 box about its center (extent-undershoot intermediate state)."""
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    return [round(cx - w / 2, 3), round(cy - h / 2, 3), round(cx + w / 2, 3), round(cy + h / 2, 3)]


def _shift_box(box, frac: float) -> list:
    """Randomly translate a 0-1000 box (center-offset intermediate state)."""
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = x2 - x1, y2 - y1
    dx = (random.random() * 2 - 1) * w * frac
    dy = (random.random() * 2 - 1) * h * frac
    return [round(x1 + dx, 3), round(y1 + dy, 3), round(x2 + dx, 3), round(y2 + dy, 3)]


def _pick_fp_candidate(meta, gt_boxes, iou_threshold: float = 0.10):
    """Pick an H candidate box that does NOT mark any GT component (a false alarm).

    H's false positives are the natural negative source for teaching ``reject``:
    a box the heatmap highlighted but that does not overlap a real defect.
    """
    cands = meta.get('prior_candidates') or []
    order = list(range(len(cands)))
    random.shuffle(order)
    for i in order:
        box = cands[i].get('bbox_2d')
        if box is None:
            continue
        box = [float(v) for v in box]
        if all(iou(box, g) < iou_threshold for g in gt_boxes):
            return box
    return None


def _anomaly_rounds(meta, cls, gt_boxes, sft_cfg):
    """(localize, imagine, confirm) rounds for an anomalous sample.

    ``[imagine]`` rehearses the *candidate box's objective quality*
    (keep / refine / reject), while ``[confirm]`` evaluates the *refine loop's net
    effect* (improved / unchanged / degraded). The two stages answer different
    questions, so SFT can no longer collapse them into the same verdict:
    a refined trajectory ends in ``[imagine]=keep`` but ``[confirm]=improved``
    (the box got better relative to the first candidate).
    """
    keep_imagine = 'keep; this box matches the observed defect and is absent from the reference'
    keep_confirm = ('unchanged; the localized boxes already matched the true defect '
                    'extent, so no refinement was needed')
    final_ground = _ground_multibox(gt_boxes)
    if not gt_boxes:
        return [(final_ground, keep_imagine, keep_confirm)]
    fp = _pick_fp_candidate(meta, gt_boxes)
    p_keep = float(sft_cfg.get('p_keep', 0.5))
    p_refine_small = float(sft_cfg.get('p_refine_small', 0.2))
    p_refine_shift = float(sft_cfg.get('p_refine_shift', 0.2))
    p_reject = float(sft_cfg.get('p_reject', 0.1))
    choices = [('keep', p_keep), ('refine_small', p_refine_small), ('refine_shift', p_refine_shift)]
    if fp is not None:
        choices.append(('reject', p_reject))
    total = sum(w for _, w in choices)
    if total <= 0:
        return [(final_ground, keep_imagine, keep_confirm)]
    r = random.random() * total
    acc = 0.0
    mode = 'keep'
    for name, w in choices:
        acc += w
        if r < acc:
            mode = name
            break
    if mode == 'keep':
        return [(final_ground, keep_imagine, keep_confirm)]
    if mode == 'refine_small':
        scale = max(0.1, min(float(sft_cfg.get('refine_small_scale', 0.5)), 0.95))
        small = [_shrink_box(b, scale) for b in gt_boxes]
        imagine = ('refine; this box undershoots the true defect extent, so it must '
                   'expand to cover the full affected region')
        confirm_r1 = 'improved; expanding the undersized box recovers the true defect extent'
        confirm_r2 = ('improved; the expanded box now covers the full defect, better '
                      'than the initial undersized candidate')
        return [(_ground_multibox(small), imagine, confirm_r1),
                (final_ground, keep_imagine, confirm_r2)]
    if mode == 'refine_shift':
        frac = max(0.05, min(float(sft_cfg.get('refine_shift_frac', 0.3)), 0.6))
        shifted = [_shift_box(b, frac) for b in gt_boxes]
        imagine = ('refine; this box is offset from the true defect center, so it must '
                   'recenter')
        confirm_r1 = 'improved; recentering the offset box recovers the true defect center'
        confirm_r2 = ('improved; the recentered box now matches the true defect, better '
                      'than the initial offset candidate')
        return [(_ground_multibox(shifted), imagine, confirm_r1),
                (final_ground, keep_imagine, confirm_r2)]
    imagine = ('reject; this region is a normal surface feature, not a true defect, '
               'so it must search elsewhere')
    confirm_r1 = ('improved; rejecting this false-alarm region and searching elsewhere '
                  'recovers the true defect')
    confirm_r2 = ('improved; the re-localized box now marks the true defect, better '
                  'than the false-alarm candidate')
    return [(_ground_multibox([fp]), imagine, confirm_r1),
            (final_ground, keep_imagine, confirm_r2)]


def _normal_rounds(meta, cls, sft_cfg):
    """(localize, imagine, confirm) rounds for a normal sample: reject an H false alarm, then none.

    A normal sample has no GT, so its ``delta_refine`` is always 0 and ``[confirm]``
    is always ``unchanged``; ``[imagine]`` still rehearses the candidate quality
    (reject for the false-alarm round, none for the empty round).
    """
    none_imagine = f'none; this candidate region shows no true defect on the {cls}'
    none_confirm = f'unchanged; no true defect on the {cls}, so nothing changed'
    empty_ground = _ground_multibox([])
    cands = meta.get('prior_candidates') or []
    fp = cands[0].get('bbox_2d') if cands else None
    p_reject_none = float(sft_cfg.get('p_normal_reject_none', 0.5))
    if fp is not None and random.random() < p_reject_none:
        imagine = 'reject; this region is a normal surface feature, not a true defect'
        confirm = f'unchanged; there is no true defect on the {cls} to refine toward'
        return [(_ground_multibox([fp]), imagine, confirm),
                (empty_ground, none_imagine, none_confirm)]
    return [(empty_ground, none_imagine, none_confirm)]


def _zoom_anomaly_rounds(meta, cls, gt_boxes, sft_cfg):
    """(localize, imagine, confirm) rounds for a ZOOM-supervised anomalous sample.

    The first round localizes a deliberately undersized candidate (simulating the
    global view's known "box too small" failure), then ``[imagine]``/``[confirm]``
    reference the zoomed crop (Image 3) to judge the extent and correct it to the
    full GT box. This teaches the model the *tool semantics* of the crop: it exists
    to fix extent, not to re-run classification or to reject a valid box.
    """
    scale = max(0.1, min(float(sft_cfg.get('zoom_shrink_scale', 0.5)), 0.95))
    small = [_shrink_box(b, scale) for b in gt_boxes]
    small_ground = _ground_multibox(small)
    final_ground = _ground_multibox(gt_boxes)
    zoom_imagine = ('refine; in Image 3 the red box undershoots the true defect '
                    'extent, so it must expand to cover the full affected region')
    zoom_confirm = ('improved; consulting the Image 3 crop and expanding the '
                    'undersized box recovers the full defect extent')
    keep_imagine = ('keep; in Image 3 the expanded box now aligns with the true '
                    'defect boundary')
    keep_confirm = ('improved; the corrected box covers the full defect, better '
                    'than the initial undersized candidate')
    return [(small_ground, zoom_imagine, zoom_confirm),
            (final_ground, keep_imagine, keep_confirm)]


def _expand_box(box, scale: float) -> list:
    """Expand a 0-1000 box about its center (extent-overshoot intermediate state)."""
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    return [round(max(0.0, cx - w / 2.0), 3), round(max(0.0, cy - h / 2.0), 3),
            round(min(1000.0, cx + w / 2.0), 3), round(min(1000.0, cy + h / 2.0), 3)]


def _zoom_candidate(meta, gt_boxes, sft_cfg):
    """Pick a perturbed B0 candidate + its pre-obs [imagine] and post-obs [confirm].

    Returns ``(candidate_boxes_1000, imagine_text, confirm_text)`` or ``(None, None,
    None)`` when no perturbation is selected. The candidate is a perturbation of the
    *true* defect box (undershoot / overshoot / shift) so the crop observation has a
    concrete, correctable error; ``[imagine]`` is written *before* the crop and only
    predicts the likely problem, while ``[confirm]`` (post-crop) cites Image 3.
    """
    p_keep = float(sft_cfg.get('zoom_p_keep', 0.2))
    p_under = float(sft_cfg.get('zoom_p_undershoot', 0.4))
    p_over = float(sft_cfg.get('zoom_p_overshoot', 0.15))
    p_shift = float(sft_cfg.get('zoom_p_shift', 0.25))
    choices = [('keep', p_keep), ('under', p_under), ('over', p_over), ('shift', p_shift)]
    total = sum(w for _, w in choices)
    if total <= 0:
        return None, None, None
    r = random.random() * total
    acc = 0.0
    mode = 'keep'
    for name, w in choices:
        acc += w
        if r < acc:
            mode = name
            break
    if mode == 'keep':
        return (list(gt_boxes),
                'keep; this candidate already matches the true defect extent, so no correction should be needed',
                'unchanged; the Image 3 crop confirms the candidate already matches the true defect, so no correction was needed')
    if mode == 'under':
        scale = max(0.1, min(float(sft_cfg.get('zoom_shrink_scale', 0.5)), 0.95))
        return ([_shrink_box(b, scale) for b in gt_boxes],
                'refine; this candidate is likely undersized relative to the true defect, so it may need expansion',
                'improved; the Image 3 crop shows the candidate undershoots the defect, and the corrected box now covers the full extent')
    if mode == 'over':
        scale = min(1.5, max(1.05, float(sft_cfg.get('zoom_expand_scale', 1.3))))
        return ([_expand_box(b, scale) for b in gt_boxes],
                'refine; this candidate is likely oversized relative to the true defect, so it may need contraction',
                'improved; the Image 3 crop shows the candidate overshoots the defect, and the corrected box now fits the true boundary')
    frac = max(0.05, min(float(sft_cfg.get('refine_shift_frac', 0.3)), 0.6))
    return ([_shift_box(b, frac) for b in gt_boxes],
            'refine; this candidate is likely offset from the true defect center, so it may need recentering',
            'improved; the Image 3 crop shows the candidate is offset, and the corrected box now matches the true defect center')


def sample_correction_candidates(meta, gt_boxes, sft_cfg):
    """Synthetic first-stage candidate set for the stage-2 correction prefix.

    These candidates are ONLY the conditional stage-2 input prefix (label=-100);
    they are never supervised as a stage-1 target, otherwise the model would be
    taught to emit intentionally incomplete candidate sets.

    Normal samples reuse the prior's suspicious regions (up to 2) as the false
    alarms to reject; anomalous samples return an empty set, a subset of the GT
    components, or a perturbed version, per ``correction_p_empty`` /
    ``correction_p_drop``.
    """
    if not meta['is_anomaly']:
        candidates = []
        for item in meta.get('prior_candidates') or []:
            box = item.get('bbox_2d')
            if box is not None and valid_box(box):
                candidates.append(list(box))
        return candidates[:2]

    p_empty = float(sft_cfg.get('correction_p_empty', 0.15))
    p_drop = float(sft_cfg.get('correction_p_drop', 0.35))
    u = random.random()

    if u < p_empty:
        return []

    if len(gt_boxes) > 1 and u < p_empty + p_drop:
        n_keep = random.randint(1, len(gt_boxes) - 1)
        indices = sorted(random.sample(range(len(gt_boxes)), n_keep))
        return [list(gt_boxes[i]) for i in indices]

    boxes, _, _ = _zoom_candidate(meta, gt_boxes, sft_cfg)
    boxes = list(gt_boxes) if boxes is None else boxes

    # Drop out-of-range perturbations so invalid boxes don't become routine
    # correction trajectories.
    return [
        list(box) for box in boxes
        if valid_box(box)
    ]


def _verdict_imagine(verdict, cls):
    """Objective-verdict -> [imagine] rehearsal text (candidate quality)."""
    if verdict == 'keep':
        return 'keep; this candidate set matches the observed defect and is absent from the reference'
    if verdict == 'refine':
        return 'refine; this candidate set marks a defect but its extent or coverage needs correction'
    if verdict == 'reject':
        return 'reject; this candidate is not supported by a real defect and must be discarded'
    if verdict == 'discover':
        return 'discover; no candidate was proposed but a defect may still be present, so keep searching'
    return f'none; this candidate region shows no true defect on the {cls}'


def _effect_confirm(effect):
    """Refine-effect -> [confirm] text (did the correction help)."""
    if effect == 'improved':
        return 'improved; correcting the candidate set recovers the complete defect set'
    if effect == 'degraded':
        return 'degraded; the correction worsened the set relative to the original candidate'
    return 'unchanged; the candidate set already matched, so no correction was needed'


def build_correction_staged_targets(meta, collator, sft_cfg):
    """Two-stage correction targets driven by real set-level scoring.

    Returns ``(pre_text, post_text, cand_boxes)``. ``pre_text`` is the stage-1
    chain carrying an *artificial wrong* candidate set (used only as the stage-2
    condition, label=-100). ``post_text`` is ``[confirm]</think><answer>`` whose
    ``[confirm]`` comes from the true set-localization effect (not hardcoded
    'improved') and whose answer is the GT box set (or empty for normal samples).
    """
    cls = str(meta.get('class_name') or 'object').replace('_', ' ')
    is_anom = bool(meta['is_anomaly'])
    comps = list(meta.get('component_bboxes') or [])
    if not comps and meta.get('gt_box_px') is not None:
        comps = [meta['gt_box_px']]
    gt_boxes = _boxes_to_1000(comps, meta['orig_size']) if is_anom else []

    cand_boxes = sample_correction_candidates(meta, gt_boxes, sft_cfg)

    understand = (
        f'Image 1 is a defect-free {cls} and sets the normal baseline: treat its '
        'material, structure, texture, print, and lighting as expected appearance. '
        f'Image 2 shows the same {cls} under inspection and should match that '
        'baseline aside from a true defect. Weigh the anomaly heatmap as a fallible '
        'search hint; do not decide anomaly or coordinates yet'
    )
    # Deliberately non-leaking: never state how many regions the GT has.
    compare = (
        'Compare the inspection image against the normal reference. '
        'The current candidate set is provisional; both its validity '
        'and its coverage require verification.'
    )
    localize = _ground_multibox(cand_boxes)

    loc_cfg = {
        **((collator.cfg.get('outcome') or {}).get('localization') or {}),
        **((collator.cfg.get('outcome') or {}).get('reward') or {}),
    }
    loc_args = {
        'iou_threshold': float(loc_cfg.get('iou_threshold', 0.30)),
        'geometry_weight': float(loc_cfg.get('geometry_weight', 0.30)),
    }
    q0 = set_localization_reward(cand_boxes, comps, meta['orig_size'], **loc_args)['reward']
    q1 = set_localization_reward(gt_boxes, comps, meta['orig_size'], **loc_args)['reward']
    effect = refine_verdict(q1 - q0, float(loc_cfg.get('refine_eps', 0.05)))
    cand_px = [to_pixels(b, meta['orig_size']) for b in cand_boxes]
    verdict = objective_verdict(
        cand_px, comps,
        float(loc_cfg.get('cand_iou_threshold', 0.10)),
        float(loc_cfg.get('keep_iou_threshold', 0.50)),
    )

    imagine = _verdict_imagine(verdict, cls)
    confirm = _effect_confirm(effect)

    if is_anom:
        n = len(gt_boxes)
        description = (
            f'A localized defect is present on the {cls}: {_region_count(n)} '
            f'{_plural(n, "differs", "differ")} from the Image 1 baseline in a way '
            f'material or appearance variation cannot explain.'
        )
        answer = json.dumps({'is_anomaly': True, 'bboxes_2d': gt_boxes, 'description': description})
    else:
        answer = json.dumps({
            'is_anomaly': False,
            'bboxes_2d': [],
            'description': 'No defect is identified after reviewing the image.',
        })

    pre = f'[understand]\n{understand}\n[compare]\n{compare}\n[localize]\n{localize}\n[imagine]\n{imagine}\n'
    post = f'[confirm]\n{confirm}\n{THINK_CLOSE_PREFIX}<answer>\n{answer}\n</answer>'
    return pre, post, cand_boxes


def build_sft_target(meta: dict, *, multibox: bool = False, thinking: bool = False,
                     answer_only: bool = False, sft_cfg: dict = None, zoom: bool = False) -> str:
    """Full thought-chain target; category/boxes come from GT.

    ``[localize]`` is supervised on GT boxes (H is never the training target — it only
    conditions the prompt). ``[imagine]`` rehearses the candidate box's objective
    quality (keep/refine/reject/none); ``[confirm]`` evaluates the refine loop's net
    effect (improved/unchanged/degraded). Multi-round reject→relocalize and
    refine→refine trajectories teach the model to correct its own localization
    inside a single pass (world-model style).
    """
    is_anom = bool(meta['is_anomaly'])
    cls = str(meta.get('class_name') or 'object').replace('_', ' ')
    sft_cfg = sft_cfg or {}
    understand = (
        f'Image 1 is a defect-free {cls} and sets the normal baseline: treat its '
        'material, structure, texture, print, and lighting as expected appearance. '
        f'Image 2 shows the same {cls} under inspection and should match that '
        'baseline aside from a true defect. Weigh the anomaly heatmap as a fallible '
        'search hint; do not decide anomaly or coordinates yet'
    )
    if multibox:
        gt_boxes = []
        if is_anom:
            comps = list(meta.get('component_bboxes') or [])
            if not comps and meta.get('gt_box_px') is not None:
                comps = [meta['gt_box_px']]
            gt_boxes = _boxes_to_1000(comps, meta['orig_size'])
        if is_anom:
            n = len(gt_boxes)
            n_txt = _region_count(n)
            where = _where_join(gt_boxes)
            compare = (
                f'against that Image 1 baseline, {n_txt} of the {cls} '
                f'{_plural(n, "differs", "differ")} locally in a way material or '
                'appearance variation on the reference cannot explain, so this is a '
                'true defect rather than normal variation'
            )
            description = (
                f'A localized defect is present on the {cls} {where}: {n_txt} '
                f'{_plural(n, "differs", "differ")} from the Image 1 baseline in a way '
                f'material or appearance variation cannot explain.'
            )
            answer = json.dumps({'is_anomaly': True, 'bboxes_2d': gt_boxes, 'description': description})
            rounds = (_zoom_anomaly_rounds(meta, cls, gt_boxes, sft_cfg) if zoom
                      else _anomaly_rounds(meta, cls, gt_boxes, sft_cfg))
        else:
            compare = (
                f'against that Image 1 baseline, Image 2 matches the reference {cls}; '
                'any apparent change is material or appearance variation rather than '
                'a true defect'
            )
            description = (
                f'The {cls} inspection image is consistent with the Image 1 baseline; '
                'apparent changes are material or appearance variation rather than a true defect.'
            )
            answer = json.dumps({'is_anomaly': False, 'bboxes_2d': [], 'description': description})
            rounds = _normal_rounds(meta, cls, sft_cfg)
        if answer_only:
            return f'<answer>\n{answer}\n</answer>'
        ground, imagine, verify = rounds[0]
        if thinking:
            return staged_sft_target(understand, compare, ground, imagine, verify, answer,
                                     extra_rounds=rounds[1:])
        return (
            f'<understand>\n{understand}\n</understand>\n'
            f'<compare>\n{compare}\n</compare>\n'
            f'<ground>\n{ground}\n</ground>\n'
            f'<verify>\n{verify}\n</verify>\n'
            f'<answer>\n{answer}\n</answer>'
        )
    if is_anom:
        box = _bbox_to_1000(meta['gt_box_px'], meta['orig_size'])
        where = _box_where(box)
        compare = (
            f'against that Image 1 baseline, one region of the {cls} differs locally '
            'in a way material or appearance variation on the reference cannot explain, '
            'so this is a true defect rather than normal variation'
        )
        imagine = 'keep; this region matches the observed difference and is absent from the reference'
        verify = 'unchanged; the grounded region already matched the true defect, so no refinement was needed'
        description = (
            f'A localized defect is present on the {cls} {where}: the region differs '
            'from the Image 1 baseline in a way material or appearance variation cannot explain.'
        )
        answer = json.dumps({'is_anomaly': True, 'bbox_2d': box, 'description': description})
        ground = f'candidate_bbox_2d={box}; {where}'
    else:
        compare = (
            f'against that Image 1 baseline, Image 2 matches the reference {cls}; '
            'any apparent change is material or appearance variation rather than '
            'a true defect'
        )
        imagine = f'none; this region shows no true defect on the {cls}'
        verify = f'unchanged; no true defect on the {cls}, so nothing changed'
        description = (
            f'The {cls} inspection image is consistent with the Image 1 baseline; '
            'apparent changes are material or appearance variation rather than a true defect.'
        )
        answer = json.dumps({'is_anomaly': False, 'bbox_2d': None, 'description': description})
        ground = 'candidate_bbox_2d=null'
    if answer_only:
        return f'<answer>\n{answer}\n</answer>'
    if thinking:
        return staged_sft_target(understand, compare, ground, imagine, verify, answer)
    return (
        f'<understand>\n{understand}\n</understand>\n'
        f'<compare>\n{compare}\n</compare>\n'
        f'<ground>\n{ground}\n</ground>\n'
        f'<verify>\n{verify}\n</verify>\n'
        f'<answer>\n{answer}\n</answer>'
    )


def build_labels(prompt_ids, target_ids):
    """Teacher-forced labels: mask the prompt, supervise the whole target.

    The full target (stage headers, bodies, ``</think>``, JSON, closing tags) is
    supervised so the model learns to write the complete chain in one pass.
    """
    return [-100] * len(prompt_ids) + list(target_ids)


def default_batch_size(arch: str) -> int:
    """Per-GPU batch for ~80GB cards."""
    name = str(arch or '')
    if '9B' in name:
        return 2
    if '4B' in name:
        return 4
    return 8


def shard_indices(n, epoch, seed, rank, world, batch_size, device):
    """Rank shard of a shuffled epoch, truncated so every rank has the same step count."""
    rng = random.Random(int(seed) + int(epoch))
    order = list(range(int(n)))
    rng.shuffle(order)
    shard = order[int(rank)::int(world)]
    batch_size = max(int(batch_size), 1)
    n_batches = len(shard) // batch_size
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        t = torch.tensor([n_batches], device=device, dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        n_batches = int(t.item())
    return shard[: n_batches * batch_size]


def _stack_gen_in(singles, max_len):
    """Merge per-sample multimodal kwargs into a padded language batch."""
    keys = set()
    for batch in singles:
        keys.update(model_inputs(batch).keys())
    out = {}
    for key in keys:
        if key in ('input_ids', 'attention_mask', 'labels'):
            continue
        vs = [batch[key] for batch in singles if key in batch and torch.is_tensor(batch[key])]
        if not vs:
            continue
        if key in ('pixel_values', 'pixel_values_videos'):
            out[key] = torch.cat(vs, dim=0)
        elif key in ('image_grid_thw', 'video_grid_thw'):
            out[key] = torch.cat([v.reshape(-1, int(v.shape[-1])) for v in vs], dim=0)
        elif key == 'mm_token_type_ids':
            padded = []
            for v in vs:
                if v.ndim == 1:
                    v = v.unsqueeze(0)
                cur = int(v.shape[-1])
                if cur < max_len:
                    v = F.pad(v, (0, max_len - cur), value=0)
                elif cur > max_len:
                    v = v[..., :max_len]
                padded.append(v)
            out[key] = torch.cat(padded, dim=0)
        else:
            out[key] = torch.cat(vs, dim=0)
    return out


def _pack_group(singles, seqs, labels_list, targets, tokenizer, device):
    """Pad one H-consistent group of (batch, target) pairs into a packed dict."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    max_len = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), max_len), int(pad_id), device=device, dtype=torch.long)
    attention_mask = torch.zeros((len(seqs), max_len), device=device, dtype=torch.long)
    labels_t = torch.full((len(seqs), max_len), -100, device=device, dtype=torch.long)
    for i, (ids, lab) in enumerate(zip(seqs, labels_list)):
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids, device=device, dtype=torch.long)
        attention_mask[i, :n] = 1
        labels_t[i, :n] = torch.tensor(lab, device=device, dtype=torch.long)
    packed = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels_t,
        gen_in=_stack_gen_in(singles, max_len),
        image_embeds=torch.cat([b['image_embeds'] for b in singles], dim=0),
        box_token_id=int(singles[0].get('box_token_id', -1)),
        control_token_id=int(singles[0].get('control_token_id', -1)),
        feat_token_id=int(singles[0].get('feat_token_id', -1)),
        h_box_geoms=[b['h_box_geom'] for b in singles if b.get('h_box_geom') is not None],
        h_maps=[b['h_map'] for b in singles if b.get('h_map') is not None],
        metas=[b['_meta'][0] for b in singles],
        targets=targets,
        seq_lens=[len(s) for s in seqs],
    )
    return packed, int((labels_t != -100).sum())


def pack_sft_batch(collator, device, samples, tokenizer, multibox, thinking=False,
                   answer_only=False, sft_cfg=None):
    """Collate one-at-a-time, then return a LIST of H-consistent packed batches.

    Every sample always keeps its normal single-pass SFT target (fully supervised).
    With probability ``zoom_prob`` a sample *additionally* produces one correction
    continuation sample:

    * normal target: 2-image (ref + test + H) prompt, target = full single-pass
      ``[understand]...[confirm]</think><answer>`` (fully supervised);
    * correction continuation: H-free multi-observation prompt + an *artificial
      wrong* stage-1 prefix (empty / dropped / perturbed candidates), target =
      ``[confirm]</think><answer>`` only. The replayed wrong prefix and the prompt
      are label=-100; only the stage-2 chain is supervised, so the model is never
      taught to emit incomplete candidates as a first-stage answer.

    Normal targets share the H channels and pack together; the H-free correction
    continuation samples pack in a separate batch so the H-Box / H-VPT injection
    stays consistent within a batch.
    """
    sft_cfg = sft_cfg or {}
    zoom_prob = float(sft_cfg.get('zoom_prob', 0.0))
    zcfg = (collator.cfg.get('outcome') or {}).get('zoom') or {}
    crop_min_pixels = zcfg.get('crop_min_pixels')
    h_singles, h_seqs, h_labels, h_targets = [], [], [], []
    post_singles, post_seqs, post_labels, post_targets = [], [], [], []
    for sample in samples:
        batch = move_batch(collator([sample]), device)
        meta = batch['_meta'][0]

        # Every sample always keeps its normal (single-pass) SFT target.
        prompt_ids = batch['input_ids'][0].tolist()
        target = build_sft_target(meta, multibox=multibox, thinking=thinking,
                                  answer_only=answer_only, sft_cfg=sft_cfg, zoom=False)
        target_ids, target_labels = selective_sft_encoding(
            tokenizer, target
        )
        h_seqs.append(prompt_ids + target_ids)
        h_labels.append([-100] * len(prompt_ids) + target_labels)
        h_targets.append(target)
        h_singles.append(batch)

        # Optional stage-2 correction continuation (H-free, multi-observation).
        do_correction = (
            zoom_prob > 0
            and multibox
            and thinking
            and not answer_only
            and random.random() < zoom_prob
        )
        if not do_correction:
            continue

        pre_text, post_text, cand_boxes = build_correction_staged_targets(meta, collator, sft_cfg)

        observations = plan_observations(
            meta['test'], cand_boxes, tuple(meta['orig_size']), collator.cfg
        )
        cont = observation_prompt(
            str(meta.get('class_name', 'object')), observations, tuple(meta['orig_size'])
        )
        zbatch = build_observation_batch(
            collator.processor, collator.prior, collator.cfg,
            meta['ref'], meta['test'],
            [obs.image for obs in observations],
            cont, device,
            crop_min_pixels=crop_min_pixels,
            prefill_text=pre_text,
        )
        # build_observation_batch returns CPU tensors (only image_embeds is on device);
        # move the whole batch to device like the collator path does, then re-attach the
        # collator's meta (H-free builder has no _meta) so _pack_group can build metas.
        zbatch = move_batch(zbatch, device)
        zbatch['_meta'] = [meta]
        zprompt_ids = zbatch['input_ids'][0].tolist()
        post_ids, post_target_labels = selective_sft_encoding(
            tokenizer, post_text
        )
        post_seqs.append(zprompt_ids + post_ids)
        post_labels.append([-100] * len(zprompt_ids) + post_target_labels)
        post_targets.append(post_text)
        post_singles.append(zbatch)
    packed_list, n_sup = [], 0
    if h_singles:
        packed, ns = _pack_group(h_singles, h_seqs, h_labels, h_targets, tokenizer, device)
        packed_list.append(packed)
        n_sup += ns
    if post_singles:
        packed, ns = _pack_group(post_singles, post_seqs, post_labels, post_targets, tokenizer, device)
        packed_list.append(packed)
        n_sup += ns
    return packed_list, n_sup


def load_sft_model(cfg, device, init_sft=None):
    """Load the base model + (optional) LoRA + H-Box projector for SFT.

    ``init_sft`` starts from an existing SFT checkpoint (a merged LoRA), so branches
    can share a common start point. The H-Box projector is built fresh (or reloaded
    from ``init_sft/h_box_prior.pt``) and trained alongside the LoRA.
    """
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    model, processor = setup_model_and_processor(cfg, for_inference=False, freeze_vision=True)
    h_box_cfg = (cfg.get('outcome') or {}).get('h_box_prior') or {}
    h_vpt_cfg = (cfg.get('outcome') or {}).get('h_vpt') or {}
    if bool(h_box_cfg.get('enabled', False)):
        ensure_box_token(processor, model)
    if bool(h_vpt_cfg.get('enabled', False)):
        ensure_control_token(processor, model)
    if init_sft:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, init_sft, is_trainable=True)
    else:
        model = apply_lora(model, cfg)
    freeze_vision_encoder(model)
    model.to(device)
    if cfg['training'].get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    force_vision_eval(model)
    prior = AnomalyPrior.from_qwen(model, cfg)
    hidden_size = int(model.config.text_config.hidden_size)
    dtype = next(model.parameters()).dtype
    model.h_box_prior = None
    if bool(h_box_cfg.get('enabled', False)):
        if init_sft:
            b_path = Path(init_sft) / 'h_box_prior.pt'
            model.h_box_prior = (load_h_box_prior(b_path, hidden_size) if b_path.exists()
                                 else build_h_box_prior(cfg, hidden_size))
        else:
            model.h_box_prior = build_h_box_prior(cfg, hidden_size)
        model.h_box_prior.to(device=device, dtype=dtype)
    model.h_vpt = None
    if bool(h_vpt_cfg.get('enabled', False)):
        if init_sft:
            h_path = Path(init_sft) / 'h_vpt.pt'
            model.h_vpt = (load_h_vpt(HVPT, h_path, hidden_size) if h_path.exists()
                           else build_h_vpt(cfg, hidden_size))
        else:
            model.h_vpt = build_h_vpt(cfg, hidden_size)
        model.h_vpt.to(device=device, dtype=dtype)
    note = 'LoRA'
    if model.h_box_prior is not None:
        note += ' + H-Box'
    if model.h_vpt is not None:
        note += ' + H-VPT'
    note += '（视觉塔已冻结）'
    print_trainable_params(model, note=note)
    return model, processor, prior


def _h_vpt_probe_train(model, h_vpt, packed, g_proj_flat=None):
    """Two-pass VPT step 1 (training): forward prompt→control token, return h_H.

    The prompt carries ``<|h_ctrl|>`` + N ``<|h_feat|>`` placeholders. A truncated
    forward up to (and including) the control token reads its last-layer hidden
    (detached, mirroring VPT's two-turn formulation), then projects H into h_H.
    Returns [B, N, hidden].

    ``g_proj_flat`` scatters the H-Box geometry tokens during the probe so ``h_ctrl``
    is the model's hidden state *after* seeing the geometry prior.
    """
    ctrl_id = int(packed['control_token_id'])
    input_ids = packed['input_ids']
    attention_mask = packed['attention_mask']
    pos = (input_ids == ctrl_id).long().argmax(dim=-1)  # [B], one per row
    max_pos = int(pos.max().item())
    ids_trunc = input_ids[:, :max_pos + 1].contiguous()
    attn_trunc = attention_mask[:, :max_pos + 1].contiguous()
    box_id = int(packed.get('box_token_id', -1))
    with torch.no_grad():
        with bind_h_box(model, g_proj_flat, box_id):
            out = forward_with_vision(model, packed['gen_in'], ids_trunc, attn_trunc,
                                      output_hidden_states=True)
        h = out.hidden_states[-1]  # [B, L, hidden]
        n = int(ids_trunc.shape[0])
        h_ctrl = h[torch.arange(n, device=h.device), pos].detach()  # [B, hidden]
    h_maps = packed['h_maps']
    h_Hs = [compute_h_H(h_vpt, hm, h_ctrl[b:b + 1]) for b, hm in enumerate(h_maps)]
    return torch.stack(h_Hs, dim=0)  # [B, N, hidden]


def batch_loss(model, packed, backward_scale=None):
    """Teacher-forced CE with two-pass VPT H injection + static H-Box geometry tokens.

    Backward stays inside the binds. The H-Box geometry tokens are static (linear
    projection, no probe); the H-VPT channel probes the control token's hidden state
    then scatters the modulated h_H into the ``<|h_feat|>`` slots.
    """
    core = unwrap_model(model)
    h_vpt = getattr(core, 'h_vpt', None)
    h_box = getattr(core, 'h_box_prior', None)
    h_maps = packed.get('h_maps')
    h_box_geoms = packed.get('h_box_geoms')
    box_id = int(packed.get('box_token_id', -1))
    with bind_cached_image_features(model, packed['image_embeds']):
        g_proj_flat = None
        if h_box is not None and h_box_geoms:
            g_proj_flat = torch.cat([compute_g_proj(h_box, g) for g in h_box_geoms], dim=0)
        if h_vpt is not None and h_maps:
            h_H = _h_vpt_probe_train(model, h_vpt, packed, g_proj_flat)  # [B, N, hidden]
            h_H_flat = h_H.reshape(-1, h_H.shape[-1])
            with bind_h_box(model, g_proj_flat, box_id), bind_h_vpt(model, h_H_flat, int(packed['feat_token_id'])):
                out = forward_with_vision(model, packed['gen_in'], packed['input_ids'],
                                          packed['attention_mask'])
        else:
            with bind_h_box(model, g_proj_flat, box_id):
                out = forward_with_vision(model, packed['gen_in'], packed['input_ids'],
                                          packed['attention_mask'])
        shift_logits = out.logits[:, :-1, :].contiguous()
        shift_labels = packed['labels'][:, 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                               shift_labels.view(-1), ignore_index=-100)
        # NOTE: no zero-weight "keep H modules in the graph" edge here. The H-free
        # post-observation batch legitimately does not use h_box_prior / h_vpt, and
        # DDP handles that via ``find_unused_parameters=True`` (set in the config).
        # A zero-weight ``sum() * 0.0`` edge made those params appear in BOTH the H
        # and H-free sub-batches of one step, so their DDP hook fired twice and hit
        # "Expected to mark a variable ready only once".
        if backward_scale is not None:
            (loss * float(backward_scale)).backward()
    return loss


def evaluate_dev(model, collator, dev_dataset, device, tokenizer, multibox, limit, seed, epoch,
                 thinking=False, answer_only=False, sft_cfg=None):
    """Mean teacher-forced loss on the holdout split (no update)."""
    raw = unwrap_model(model)
    raw.eval()
    force_vision_eval(raw)
    rng = random.Random(int(seed) * 1000 + int(epoch))
    order = list(range(len(dev_dataset)))
    rng.shuffle(order)
    order = order[:max(1, int(limit))]
    world = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    shard = order[rank::world]
    if world > 1:
        t = torch.tensor([len(shard)], device=device, dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        shard = shard[:int(t.item())]
    total = 0.0
    with torch.no_grad():
        for i in shard:
            packed_list, _ = pack_sft_batch(collator, device, [dev_dataset[i]], tokenizer, multibox,
                                            thinking, answer_only, sft_cfg)
            for packed in packed_list:
                total += float(batch_loss(raw, packed))
    count = len(shard)
    if world > 1:
        t = torch.tensor([total, float(count)], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total, count = float(t[0]), int(t[1])
    model.train()
    force_vision_eval(model)
    return total / max(1, count), count


def quick_rollout_eval(model, processor, prior, collator, dev_dataset, cfg, device, limit, seed, step,
                       writer=None):
    """Lightweight single-pass rollout smoke eval (real generation + scoring)."""
    raw = unwrap_model(model)
    raw.eval()
    force_vision_eval(raw)
    max_boxes = int(cfg['outcome'].get('max_boxes', 16))
    protocol_weight = float(cfg['outcome'].get('protocol_weight', 0.01))
    loc_cfg = {**(cfg['outcome'].get('localization') or {}), **(cfg['outcome'].get('reward') or {})}
    rng = random.Random(int(seed) * 999983 + int(step))
    order = list(range(len(dev_dataset)))
    rng.shuffle(order)
    order = order[:max(1, int(limit))]
    world = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    shard = order[rank::world]
    if world > 1:
        t = torch.tensor([len(shard)], device=device, dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        shard = shard[:int(t.item())]

    n_anom = n_norm = n_correct_anom = n_correct_norm = n_valid = 0
    miou_sum = 0.0
    samples = []
    with torch.no_grad():
        for i in shard:
            batch = move_batch(collator([dev_dataset[i]]), device)
            comp = generate_group(raw, processor, batch, cfg, group=1, sample=False)[0]
            parsed = parse_output_cfg(comp.text, cfg, max_boxes=max_boxes)
            meta = batch['_meta'][0]
            sc = score_output(parsed, meta, protocol_weight, loc_cfg, max_boxes=max_boxes)
            is_anom = bool(meta['is_anomaly'])
            if is_anom:
                n_anom += 1
                miou_sum += float(sc.get('mask_iou') or 0.0)
                n_correct_anom += int(bool(sc.get('correct')))
            else:
                n_norm += 1
                n_correct_norm += int(bool(sc.get('correct')))
            n_valid += int(bool(parsed.get('task_valid')))
            if rank == 0 and len(samples) < 3:
                samples.append((str(meta.get('class_name')), is_anom, comp.text))
                if writer is not None and meta.get('test') is not None:
                    im = _draw_pred_boxes(meta['test'], meta, parsed)
                    writer.add_image(f'smoke_case/{len(samples)}_pred_box', _pil_to_tb(im), step)

    acc = torch.tensor([float(n_anom), float(n_norm), float(n_correct_anom), float(n_correct_norm),
                        float(n_valid), float(miou_sum)], device=device, dtype=torch.float64)
    if world > 1:
        dist.all_reduce(acc, op=dist.ReduceOp.SUM)
    n_anom, n_norm, n_correct_anom, n_correct_norm, n_valid, miou_sum = (float(acc[k]) for k in range(6))
    model.train()
    force_vision_eval(model)
    return dict(
        n=int(n_anom + n_norm),
        recall=(n_correct_anom / n_anom if n_anom else None),
        tnr=(n_correct_norm / n_norm if n_norm else None),
        acc=(n_correct_anom + n_correct_norm) / max(1, int(n_anom + n_norm)),
        task_valid=n_valid / max(1, int(n_anom + n_norm)),
        mask_miou=(miou_sum / n_anom if n_anom else None),
        samples=samples,
    )


def _in_distributed_worker() -> bool:
    return os.environ.get('LOCAL_RANK') is not None


def _maybe_relaunch_multi_gpu(num_gpu: int) -> None:
    if int(num_gpu) <= 1 or _in_distributed_worker():
        return
    script = os.path.abspath(sys.argv[0])
    cmd = [sys.executable, '-m', 'torch.distributed.run', f'--nproc_per_node={int(num_gpu)}',
           script, *sys.argv[1:]]
    print(f'[sft] distributed.num_gpu={num_gpu}，正在启动: {" ".join(cmd)}', flush=True)
    raise SystemExit(subprocess.call(cmd))


def _save_adapter(model, processor, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    core = unwrap_model(model)
    core.save_pretrained(path)
    processor.save_pretrained(path)
    save_h_box_prior(getattr(core, 'h_box_prior', None), path / 'h_box_prior.pt')
    save_h_vpt(getattr(core, 'h_vpt', None), path / 'h_vpt.pt')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=None,
                        help='per-GPU batch. Default: 8/4/2 for 2B/4B/9B on ~80GB cards')
    parser.add_argument('--num-gpu', type=int, default=1)
    parser.add_argument('--accum', type=int, default=1)
    parser.add_argument('--save-steps', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--dev-eval-samples', type=int, default=64,
                        help='holdout samples for the dev loss reported at each epoch end; 0 disables')
    parser.add_argument('--eval-steps', type=int, default=0,
                        help='run a quick single-pass rollout smoke eval every N optimizer steps (0 disables)')
    parser.add_argument('--eval-rollout-samples', type=int, default=12,
                        help='holdout samples per smoke eval (sharded across ranks)')
    parser.add_argument('--init-sft', default=None,
                        help='start from an existing SFT checkpoint (merged LoRA)')
    args = parser.parse_args()
    _maybe_relaunch_multi_gpu(args.num_gpu)

    cfg = load_yaml_config(args.config)
    seed = int(cfg['training']['seed'])
    set_seed(seed)
    if args.batch_size is None:
        args.batch_size = default_batch_size(cfg.get('model', {}).get('arch', ''))

    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', '0')))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', str(local_rank)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend='nccl', timeout=timedelta(hours=2))
    main_proc = is_main_process()

    multibox = (cfg.get('outcome', {}).get('version') == 'outcome-multibox-v1')
    dataset_cls = OutcomeMultiboxDataset if multibox else OutcomeDataset
    collator_cls = OutcomeMultiboxCollator if multibox else OutcomeCollator

    model, processor, prior = load_sft_model(cfg, device, init_sft=args.init_sft)
    if world > 1:
        # Two-stage zoom supervision packs an H-free post-observation batch whose
        # forward omits h_box_prior / h_vpt, so within one optimizer step those
        # parameters receive no grad from that sub-batch. DDP's reducer rejects a
        # forward whose parameter set differs from the prior one unless it is told
        # to track unused parameters.
        find_unused = bool(cfg.get('distributed', {}).get('ddp_find_unused_parameters', True))
        model = DDP(model, device_ids=[local_rank] if device.type == 'cuda' else None,
                    find_unused_parameters=find_unused)
    tokenizer = getattr(processor, 'tokenizer', processor)
    thinking = thinking_enabled(cfg)
    reasoning_mode = str((cfg.get('outcome') or {}).get('reasoning_mode', 'fsm'))
    answer_only = reasoning_mode in ('loop', 'direct')
    sft_cfg = (cfg.get('outcome') or {}).get('sft') or {}

    train, test = load_prior_split(cfg)
    train, dev = split_holdout_by_class(train, float(cfg['data']['holdout_ratio']),
                                        seed=seed)
    pool = build_train_ref_pool(train)
    dataset = dataset_cls(train, cfg, processor, 'train', pool)
    if args.max_samples is not None:
        dataset.samples = dataset.samples[:max(1, int(args.max_samples))]
    dev_dataset = None
    if int(args.dev_eval_samples) > 0 and len(dev):
        dev_dataset = dataset_cls(dev, cfg, processor, 'eval', pool)

    collator = collator_cls(processor, prior, cfg)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=0.0)

    output_dir = Path(args.output_dir)
    tb_cfg = cfg.get('tensorboard') or {}
    vis_every = int(tb_cfg.get('vis_every_n_steps', 20) or 0)
    log_every = int(tb_cfg.get('log_every_n_steps', 10) or 10)
    overlay_alpha = float((cfg.get('prior') or {}).get('overlay_alpha', 0.45))
    if main_proc:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
        tb_root = output_dir / 'tb'
        _archive_legacy_events(tb_root)
        run_dir = tb_root / time.strftime('run_%Y%m%d_%H%M%S')
        start_tensorboard(tb_root, cfg)
        writer = SummaryWriter(str(run_dir))
        log_sft_hparams(writer, cfg, args, world)
        print(f'[sft] arch={cfg.get("model", {}).get("arch")} batch_size={args.batch_size} '
              f'num_gpu={world} accum={args.accum} '
              f'global_batch={args.batch_size * world * max(1, int(args.accum))} '
              f'tb={run_dir} vis_every={vis_every}', flush=True)
    else:
        writer = _NullWriter()

    total_epochs = max(1, int(args.epochs))
    accum = max(1, int(args.accum))
    batch_size = max(1, int(args.batch_size))
    seen = 0
    step = 0
    running_loss = 0.0
    running_supervised = 0
    window_samples = 0
    try:
        for epoch in range(1, total_epochs + 1):
            model.train()
            force_vision_eval(model)
            indices = shard_indices(len(dataset), epoch, seed, rank, world, batch_size, device)
            for start in range(0, len(indices), batch_size):
                started = time.perf_counter()
                samples = [dataset[i] for i in indices[start:start + batch_size]]
                packed_list, n_sup = pack_sft_batch(collator, device, samples, tokenizer, multibox,
                                                    thinking, answer_only, sft_cfg)
                loss = None
                all_metas = []
                all_targets = []
                all_seq_lens = []
                for packed in packed_list:
                    l = batch_loss(model, packed)
                    loss = l if loss is None else loss + l
                    all_metas.extend(packed['metas'])
                    all_targets.extend(packed['targets'])
                    all_seq_lens.extend(packed['seq_lens'])
                running_loss += float(loss.detach())
                running_supervised += n_sup
                window_samples += len(samples)
                seen += 1
                # Single backward per outer batch: the number of packed sub-batches
                # (h_batch + optional H-free post-observation batch) differs across
                # ranks depending on whether a zoom-upgraded sample landed in each
                # rank's shard. One `.backward()` here keeps DDP's reducer in lockstep
                # and avoids an NCCL deadlock from asymmetric backward counts.
                (loss / accum).backward()
                if seen % accum == 0:
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        float(args.max_grad_norm)))
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    anomaly_frac = sum(bool(m.get('is_anomaly')) for m in all_metas) / max(1, len(all_metas))
                    seq_mean = sum(all_seq_lens) / max(1, len(all_seq_lens))
                    seconds = time.perf_counter() - started
                    writer.add_scalar('train/loss', float(loss.detach()), step)
                    writer.add_scalar('train/supervised_tokens', n_sup, step)
                    writer.add_scalar('train/seq_len_mean', seq_mean, step)
                    writer.add_scalar('train/anomaly_frac', anomaly_frac, step)
                    writer.add_scalar('train/lr', float(args.lr), step)
                    writer.add_scalar('train/epoch', epoch, step)
                    writer.add_scalar('optimizer/grad_norm', grad_norm, step)
                    writer.add_scalar('train/seconds_per_step', seconds, step)
                    if device.type == 'cuda':
                        writer.add_scalar('train/gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 1024 ** 3, step)
                    writer.flush()
                    if main_proc and step % log_every == 0:
                        print(f'[sft] step={step} loss={running_loss / max(1, window_samples):.4f} '
                              f'supervised_tok={running_supervised} seq={seq_mean:.0f} '
                              f'anom={anomaly_frac:.2f} gnorm={grad_norm:.2f} '
                              f'seen={seen * batch_size} ({seconds:.1f}s)', flush=True)
                        running_loss = 0.0
                        running_supervised = 0
                        window_samples = 0
                    if main_proc and vis_every > 0 and step % vis_every == 0:
                        log_sft_case(writer, step, all_metas[0], all_targets[0],
                                     overlay_alpha=overlay_alpha)
                    if main_proc and args.save_steps > 0 and step % int(args.save_steps) == 0:
                        _save_adapter(model, processor, output_dir / f'checkpoint-{step}')
                    if world > 1 and dist.is_initialized() and args.save_steps > 0 and step % int(args.save_steps) == 0:
                        dist.barrier()
                    # Quick single-pass rollout smoke eval every N steps. Falls back to
                    # the train split when holdout_ratio=0 (no dev set), so the model's
                    # real sampled boxes are always visible in TB (smoke_case/*_pred_box).
                    if args.eval_steps > 0 and step % int(args.eval_steps) == 0:
                        smoke_src = dev_dataset if dev_dataset is not None else dataset
                        q = quick_rollout_eval(model, processor, prior, collator, smoke_src, cfg, device,
                                               int(args.eval_rollout_samples), seed, step, writer=writer)
                        writer.add_scalar('smoke/recall', q['recall'] if q['recall'] is not None else float('nan'), step)
                        writer.add_scalar('smoke/tnr', q['tnr'] if q['tnr'] is not None else float('nan'), step)
                        writer.add_scalar('smoke/acc', q['acc'], step)
                        writer.add_scalar('smoke/task_valid', q['task_valid'], step)
                        writer.add_scalar('smoke/mask_miou', q['mask_miou'] if q['mask_miou'] is not None else float('nan'), step)
                        writer.flush()
                        if main_proc:
                            def _fmt(v):
                                return '--' if v is None else f'{v:.3f}'
                            print(f'[smoke] step={step} n={q["n"]} rec={_fmt(q["recall"])} tnr={_fmt(q["tnr"])} '
                                  f'acc={q["acc"]:.3f} tv={q["task_valid"]:.3f} mIoU={_fmt(q["mask_miou"])}', flush=True)
                            for cls, anom, text in q['samples']:
                                tag = 'ANOM' if anom else 'NORM'
                                print(f'  [{tag}:{cls}] {text[-240:]}', flush=True)
                    if world > 1 and dist.is_initialized() and args.eval_steps > 0 and step % int(args.eval_steps) == 0:
                        dist.barrier()
            if dev_dataset is not None:
                dev_loss, n_dev = evaluate_dev(model, collator, dev_dataset, device, tokenizer,
                                               multibox, int(args.dev_eval_samples), seed, epoch,
                                               thinking=thinking, answer_only=answer_only,
                                               sft_cfg=sft_cfg)
                writer.add_scalar('dev/loss', dev_loss, step)
                writer.add_scalar('dev/n', n_dev, step)
                writer.flush()
                if main_proc:
                    print(f'[sft] epoch={epoch}/{total_epochs} dev_loss={dev_loss:.4f} (n={n_dev})', flush=True)
            if world > 1 and dist.is_initialized():
                dist.barrier()
        if seen % accum != 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                float(args.max_grad_norm))
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

        if main_proc:
            _save_adapter(model, processor, output_dir)
            print(f'[sft] saved SFT LoRA to {output_dir} (steps={step})', flush=True)
    finally:
        writer.close()
        if world > 1 and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
