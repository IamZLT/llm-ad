#!/usr/bin/env python3
"""Region-module SFT: train the region adapter + language LoRA with real category/bbox.

Freezes the vision encoder and H/matching computation. Supervises the full five-block
target: <ground>/<answer> carry the GT category/boxes, while the process blocks
(<understand>/<compare>/<verify>) are sample-aware templates built from GT meta, and
the answer description summarizes the reasoning chain in one sentence, so the
cold-start RL policy reliably emits the whole scaffold with meaningful text.

After this run, the saved directory is pointed to by ``outcome.sft_adapter`` in the RL
config: ``load_model`` merges the SFT LoRA into the base weights, mounts the frozen
region adapter, and starts a fresh RL LoRA.
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
from models.looped_qwen import enable_looped_qwen
from models.lora import apply_lora
from models.qwen35 import (setup_model_and_processor, freeze_vision_encoder, force_vision_eval,
                           unwrap_model, print_trainable_params)
from models.region_adapter import build_region_adapter
from models.region_injection import bind_region_injection, ensure_region_token, region_raw_from_batch, save_region_adapter
from models.vision_cache import bind_cached_image_features
from outcome.inputs import OutcomeCollator, OutcomeDataset
from outcome.inputs_multibox import OutcomeMultiboxCollator, OutcomeMultiboxDataset
from outcome.protocol_multibox import candidate_hits_comp
from outcome.thinking import thinking_enabled, staged_sft_target, staged_sft_labels
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
        f"h_candidates={len(meta.get('prior_candidates') or [])} "
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
    """Coarse image-frame location from a 0-1000 box. Not a defect-type label.

    Sample-specific spatial words belong in <ground> (next to the candidate
    coordinates) and the answer description, not in <understand>/<compare>.
    """
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


# Candidate-hit rules for the multibox <ground>/<verify> targets, in the 0-1000
# system. H candidates are patch-aligned boxes that usually sit *inside* a GT
# component (matched cand-GT IoU ~0.17 on VisA train), so a hit is IoU>=bar OR
# candidate-center inside the component (shared with the RL coverage reward via
# outcome.protocol_multibox.candidate_hits_comp).
CAND_HIT_IOU = 0.10
# A candidate that already covers a component this tightly needs no refinement.
CAND_TIGHT_IOU = 0.50


def _box_iou_1000(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + ab - inter
    return float(inter / union) if union > 0 else 0.0


def _prior_candidate_boxes(meta: dict) -> list:
    """H region-hint boxes (already 0-1000) in proposal order."""
    out = []
    for p in meta.get('prior_candidates') or []:
        b = p.get('bbox_2d') if isinstance(p, dict) else None
        if b and len(b) == 4:
            out.append([round(float(v), 3) for v in b])
    return out


def _ground_multibox(cands: list) -> str:
    """H-hint boxes plus coarse location; empty list has no location suffix."""
    if not cands:
        return 'candidate_bboxes_2d=[]'
    return f'candidate_bboxes_2d={cands}; {_where_join(cands)}'


def build_sft_target(meta: dict, multibox: bool = False, thinking: bool = False, answer_only: bool = False) -> str:
    """Five-block target; category/boxes come from GT, process blocks summarize the chain.

    The process blocks and the answer ``description`` are sample-aware templates built
    from GT meta (class name, component count, coarse box location): the description
    restates the understand -> compare -> verify conclusion in one sentence instead of
    a bare label, so the cold-start policy learns to summarize its own reasoning.
    When ``thinking=True``, stages go inside native think as
    ``[understand]/[compare]/[localize]/[confirm]``, then ``</think>`` and
    ``<answer>`` JSON. The chat template already opened ``<think>``; SFT
    supervises the four stages and the JSON (no extra XML blocks).

    ``multibox=True`` emits the outcome-multibox-v1 answer schema (``bboxes_2d`` as a
    list of one box per disconnected GT component, empty list for normal) instead of
    the single union box (``bbox_2d``). Use it for the SFT that seeds multi-box RL so
    the language LoRA is already aligned with the multi-box output format.

    Multibox ground/verify semantics (cold-start for the coarse-to-fine loop):

    * ``<ground>`` reads out the H region hints (``candidate_bboxes_2d`` =
      ``prior_candidates`` boxes, for BOTH normal and anomalous samples) and
      names each candidate's approximate image-frame location after a semicolon.
      This is a label-agnostic "hints -> coordinates" task, so the candidate
      stage is a high-recall proposal step rather than a copy of the final answer.
    * ``<verify>`` adjudicates the candidates against GT coverage: ``keep`` when
      they already bound the defect tightly, ``refine`` when bounds need
      tightening or spurious candidates must be dropped, ``discover`` when GT
      components sit beyond the marked hints, ``reject`` (normal with hints) /
      ``none`` (normal, no hints) when no candidate confirms a defect.
    * ``<answer>`` carries the precise per-component GT boxes (or [] for normal).
    """
    is_anom = bool(meta['is_anomaly'])
    cls = str(meta.get('class_name') or 'object').replace('_', ' ')
    understand = (
        f'Image 1 is a defect-free {cls} and sets the normal baseline: treat its '
        'material, structure, texture, print, and lighting as expected appearance. '
        f'Image 2 shows the same {cls} under inspection and should match that '
        'baseline aside from a true defect. Weigh later region hints as fallible; '
        'do not decide anomaly or coordinates yet'
    )
    if multibox:
        cands = _prior_candidate_boxes(meta)
        ground = _ground_multibox(cands)
        if is_anom:
            comps = list(meta.get('component_bboxes') or [])
            if not comps and meta.get('gt_box_px') is not None:
                comps = [meta['gt_box_px']]
            boxes = _boxes_to_1000(comps, meta['orig_size'])
            n = len(boxes)
            n_txt = _region_count(n)
            where = _where_join(boxes)
            compare = (
                f'against that Image 1 baseline, {n_txt} of the {cls} '
                f'{_plural(n, "differs", "differ")} locally in a way material or '
                'appearance variation on the reference cannot explain, so this is a '
                'true defect rather than normal variation'
            )
            comp_hit = [max((_box_iou_1000(c, g) for c in cands), default=0.0) for g in boxes]
            covered = [any(candidate_hits_comp(c, g, CAND_HIT_IOU) for c in cands) for g in boxes]
            n_missed = sum(1 for ok in covered if not ok)
            n_spurious = sum(1 for c in cands
                             if not any(candidate_hits_comp(c, g, CAND_HIT_IOU) for g in boxes))
            if not cands:
                verify = ('discover; the region evidence is silent, but direct reference '
                          f'comparison localizes {n_txt} of defect on the {cls}')
            elif n_missed > 0:
                verify = (f'discover; {_region_count(n_missed)} '
                          f'{_plural(n_missed, "lies", "lie")} beyond the marked candidates, '
                          'and the rest are tightened to the defect extent')
            elif n_spurious > 0:
                verify = (f'refine; dropped {_region_count(n_spurious)} matching the reference '
                          'and tightened the remaining candidates to the defect extent')
            elif all(v >= CAND_TIGHT_IOU for v in comp_hit):
                verify = ('keep; the candidate bounds already match the observed difference '
                          'and are absent from the reference')
            else:
                verify = ('refine; tightened the candidate bounds to the exact defect extent, '
                          'confirmed against the reference')
            description = (
                f'A localized defect is present on the {cls} {where}: {n_txt} '
                f'{_plural(n, "differs", "differ")} from the Image 1 baseline in a way '
                f'material or appearance variation cannot explain.'
            )
            answer = json.dumps({'is_anomaly': True, 'bboxes_2d': boxes, 'description': description})
        else:
            compare = (
                f'against that Image 1 baseline, Image 2 matches the reference {cls}; '
                'any apparent change is material or appearance variation rather than '
                'a true defect'
            )
            k = len(cands)
            if k:
                verify = (f'reject; the marked {_plural(k, "candidate matches", "candidates match")} '
                          'the reference, consistent with normal material or appearance '
                          'variation rather than a true defect')
            else:
                verify = f'none; no candidate region confirms a true defect on the {cls}'
            description = (
                f'The {cls} inspection image is consistent with the Image 1 baseline; '
                'apparent changes are material or appearance variation rather than a true defect.'
            )
            answer = json.dumps({'is_anomaly': False, 'bboxes_2d': [], 'description': description})
        if answer_only:
            return f'<answer>\n{answer}\n</answer>'
        if thinking:
            return staged_sft_target(understand, compare, ground, verify, answer)
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
        verify = 'keep; the grounded region matches the observed difference and is absent from the reference'
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
        verify = f'none; no candidate region confirms a true defect on the {cls}'
        description = (
            f'The {cls} inspection image is consistent with the Image 1 baseline; '
            'apparent changes are material or appearance variation rather than a true defect.'
        )
        answer = json.dumps({'is_anomaly': False, 'bbox_2d': None, 'description': description})
        ground = 'candidate_bbox_2d=null'
    if answer_only:
        return f'<answer>\n{answer}\n</answer>'
    if thinking:
        return staged_sft_target(understand, compare, ground, verify, answer)
    return (
        f'<understand>\n{understand}\n</understand>\n'
        f'<compare>\n{compare}\n</compare>\n'
        f'<ground>\n{ground}\n</ground>\n'
        f'<verify>\n{verify}\n</verify>\n'
        f'<answer>\n{answer}\n</answer>'
    )


def build_labels(prompt_ids, target_ids, target_text=None, tokenizer=None, thinking=False):
    """Teacher-forced labels: mask the prompt, supervise the target.

    For thinking SFT the ``[stage]`` headers / ``</think>`` / ``<answer>``
    wrappers are FSM-controlled markers (``outcome.staged_decode`` injects them,
    the model never samples them), so they are masked to keep teacher-forcing
    aligned with the staged decoder. Non-thinking SFT supervises the whole target.
    """
    if thinking and target_text is not None and tokenizer is not None:
        labels = staged_sft_labels(tokenizer, target_text)
        if len(labels) != len(target_ids):
            # Offset-mapping mismatch (should not happen); fall back to full sup.
            labels = list(target_ids)
        return [-100] * len(prompt_ids) + labels
    return [-100] * len(prompt_ids) + list(target_ids)


def default_batch_size(arch: str) -> int:
    """Per-GPU batch for ~80GB cards (current 4B SFT is ~21GB at batch=1)."""
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
    """Merge per-sample multimodal kwargs into a padded language batch.

    ``input_ids`` / ``attention_mask`` are rebuilt from the SFT target in
    ``pack_sft_batch`` and must not be concatenated here (their lengths differ).
    """
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


def pack_sft_batch(collator, device, samples, tokenizer, multibox, thinking=False, answer_only=False):
    """Collate one-at-a-time (vision cache is per-pair), then pad a language batch."""
    singles, seqs, labels_list, targets = [], [], [], []
    for sample in samples:
        batch = move_batch(collator([sample]), device)
        prompt_ids = batch['input_ids'][0].tolist()
        target = build_sft_target(batch['_meta'][0], multibox=multibox, thinking=thinking,
                                  answer_only=answer_only)
        target_ids = tokenizer(target, add_special_tokens=False).input_ids
        labels_list.append(build_labels(prompt_ids, target_ids, target_text=target,
                                        tokenizer=tokenizer, thinking=thinking))
        seqs.append(prompt_ids + target_ids)
        targets.append(target)
        singles.append(batch)
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
        region_token_id=int(singles[0]['region_token_id']),
        region_raw=dict(
            test=torch.cat([b['region_test'] for b in singles], dim=1),
            ref=torch.cat([b['region_ref'] for b in singles], dim=1),
            geom=torch.cat([b['region_geom'] for b in singles], dim=1),
            hstat=torch.cat([b['region_hstat'] for b in singles], dim=1),
            valid=torch.cat([b['region_valid'] for b in singles], dim=1),
        ),
        metas=[b['_meta'][0] for b in singles],
        targets=targets,
        seq_lens=[len(s) for s in seqs],
    )
    return packed, int((labels_t != -100).sum())


def load_sft_model(cfg, device):
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    model, processor = setup_model_and_processor(cfg, for_inference=False, freeze_vision=True)
    ensure_region_token(processor, model)
    model = apply_lora(model, cfg)
    enable_looped_qwen(model, cfg)
    freeze_vision_encoder(model)
    model.to(device)
    if cfg['training'].get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    force_vision_eval(model)
    prior = AnomalyPrior.from_qwen(model, cfg)
    feature_dim = int(prior.visual.config.hidden_size)
    hidden_size = int(model.config.text_config.hidden_size)
    model.region_adapter = build_region_adapter(cfg, feature_dim, hidden_size)
    # The adapter is created after model.to(device): sync it explicitly or the SFT
    # forward will hit a CPU/FP32 vs GPU/BF16 mismatch.
    dtype = next(model.parameters()).dtype
    model.region_adapter.to(device=device, dtype=dtype)
    print_trainable_params(model, note='LoRA + region adapter，视觉塔已冻结')
    return model, processor, prior


def batch_loss(model, packed, backward_scale=None):
    """Teacher-forced CE on a packed batch. Backward stays inside the bind contexts."""
    adapter = getattr(unwrap_model(model), 'region_adapter', None)
    with bind_cached_image_features(model, packed['image_embeds']), \
            bind_region_injection(model, adapter, packed['region_raw'], int(packed['region_token_id'])):
        out = forward_with_vision(model, packed['gen_in'], packed['input_ids'], packed['attention_mask'])
        shift_logits = out.logits[:, :-1, :].contiguous()
        shift_labels = packed['labels'][:, 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                               shift_labels.view(-1), ignore_index=-100)
        if backward_scale is not None:
            (loss * float(backward_scale)).backward()
    return loss


def evaluate_dev(model, collator, dev_dataset, device, tokenizer, multibox, limit, seed, epoch, thinking=False, answer_only=False):
    """Mean teacher-forced loss on the holdout split (no update).

    Every rank evaluates an equal-length disjoint shard using the unwrapped module
    (no DDP hooks fire, so eval issues no collectives), then the mean is all-reduced.
    Running this on rank 0 only deadlocks NCCL: rank 0 would issue DDP-wrapped
    forwards while rank 1 sits in the epoch-end barrier — mismatched collectives.
    """
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
            packed, _ = pack_sft_batch(collator, device, [dev_dataset[i]], tokenizer, multibox, thinking, answer_only)
            total += float(batch_loss(raw, packed))
    count = len(shard)
    if world > 1:
        t = torch.tensor([total, float(count)], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total, count = float(t[0]), int(t[1])
    model.train()
    force_vision_eval(model)
    return total / max(1, count), count


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
    save_region_adapter(getattr(core, 'region_adapter', None), path / 'region_adapter.pt')


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

    model, processor, prior = load_sft_model(cfg, device)
    if world > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == 'cuda' else None,
                    find_unused_parameters=False)
    tokenizer = getattr(processor, 'tokenizer', processor)
    thinking = thinking_enabled(cfg)
    reasoning_mode = str((cfg.get('outcome') or {}).get('reasoning_mode', 'fsm'))
    answer_only = reasoning_mode in ('loop', 'direct')

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
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(args.lr), weight_decay=0.0)

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
                packed, n_sup = pack_sft_batch(collator, device, samples, tokenizer, multibox, thinking, answer_only)
                loss = batch_loss(model, packed, backward_scale=1.0 / accum)
                running_loss += float(loss.detach())
                running_supervised += n_sup
                window_samples += len(samples)
                seen += 1
                if seen % accum == 0:
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        float(args.max_grad_norm)))
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    anomaly_frac = sum(bool(m.get('is_anomaly')) for m in packed['metas']) / max(1, len(packed['metas']))
                    seq_mean = sum(packed['seq_lens']) / max(1, len(packed['seq_lens']))
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
                        log_sft_case(writer, step, packed['metas'][0], packed['targets'][0],
                                     overlay_alpha=overlay_alpha)
                    if main_proc and args.save_steps > 0 and step % int(args.save_steps) == 0:
                        _save_adapter(model, processor, output_dir / f'checkpoint-{step}')
                    if world > 1 and dist.is_initialized() and args.save_steps > 0 and step % int(args.save_steps) == 0:
                        dist.barrier()
            if dev_dataset is not None:
                dev_loss, n_dev = evaluate_dev(model, collator, dev_dataset, device, tokenizer,
                                               multibox, int(args.dev_eval_samples), seed, epoch,
                                               thinking=thinking, answer_only=answer_only)
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
            print(f'[sft] saved SFT LoRA + region adapter to {output_dir} (steps={step})', flush=True)
    finally:
        writer.close()
        if world > 1 and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
