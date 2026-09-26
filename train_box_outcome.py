#!/usr/bin/env python3
"""Train the action-outcome predictor (``box_outcome.pt``).

Phase 2 of the world-model plan. Uses a *frozen* visual encoder + detector to
collect per-sample contexts (global ref+test features, crop features, selected
candidate box) and the realized quality label for every geometric action, then
fits the small ``BoxOutcomeModel`` by regression. The detector is frozen so the
training target is stable; the predictor is never used as a correctness reward.

Usage:
    python train_box_outcome.py --config configs/xxx.yaml \
        --init-sft <sft_ckpt_dir> --output <out_dir> [--max-samples N]
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from PIL import Image

from data.prior_dataset import build_train_ref_pool
from data.scan import load_prior_split, split_holdout_by_class
from models.box_outcome_model import (ALL_ACTIONS, BoxOutcomeModel,
                                      action_outcome_quality, apply_selected_action,
                                      default_action_ids, save_box_outcome)
from outcome.inputs import _smart_resize_image, encode_visual_merged
from outcome.inputs_multibox import OutcomeMultiboxCollator, OutcomeMultiboxDataset
from outcome.policy import generate_inspection_group
from outcome.zoom_crop import make_zoom_crop
from rl.grpo import move_batch
from train_region_sft import load_sft_model
from utils.common import set_seed
from utils.config import load_yaml_config


def _encode(prior, processor, images, device) -> torch.Tensor:
    """Pooled visual feature [D] for a list of PIL images."""
    img_proc = processor.image_processor
    enc = img_proc(images=images, return_tensors='pt')
    merged = encode_visual_merged(prior, enc['pixel_values'].to(device),
                                  enc['image_grid_thw'].to(device))
    return merged.mean(dim=0).detach()


def collect_features(model, processor, prior, dataset, collator, cfg, device,
                     indices, max_samples, seed):
    """Run the frozen detector and gather (global, crop, candidate, action labels).

    Returns lists of tensors (aligned rows). A sample contributes one row when it
    produced a selectable candidate; no-candidate samples are skipped (their B0 is
    empty, so there is no box to act on).
    """
    data = cfg.get('data') or {}
    max_size = int(data.get('max_image_size', 768))
    from models.qwen35 import qwen_vision_factor
    factor = qwen_vision_factor(processor, getattr(prior, 'visual', None))
    cap = max_size * max_size
    zcfg = (cfg.get('outcome') or {}).get('zoom') or {}
    expand = float(zcfg.get('expand', 1.0))
    min_pad_frac = float(zcfg.get('min_pad_frac', 0.12))
    max_area_frac = float(zcfg.get('max_area_frac', 0.6))
    crop_min_pixels = zcfg.get('crop_min_pixels') or max(cap // 2, 256 * 256)
    iou_threshold = float((cfg.get('outcome') or {}).get('localization', {}).get('iou_threshold', 0.30))
    geometry_weight = float((cfg.get('outcome') or {}).get('localization', {}).get('geometry_weight', 0.30))

    globals_, crops, boxes, labels = [], [], [], []
    n_used = 0
    n_skipped = 0
    for index in indices:
        batch = move_batch(collator([dataset[index]]), device)
        _completion, trace = generate_inspection_group(model, processor, prior, batch, cfg)
        meta = batch['_meta'][0]
        idx = int(trace.selected_box_index)
        if idx < 0 or not trace.initial_boxes:
            n_skipped += 1
            continue
        cand = [float(v) for v in trace.initial_boxes[idx]]
        ref_img, test_img = meta['ref'], meta['test']
        # Global features: ref + test (same resize convention as stage-1/2).
        test_rs = _smart_resize_image(test_img, max_size, factor, 256 * 256, cap)
        ref_rs = ref_img.resize(test_rs.size, Image.Resampling.BICUBIC)
        g = _encode(prior, processor, [ref_rs, test_rs], device)
        # Crop features (or None for a degenerate crop).
        crop_img = None
        if trace.zoom_executed:
            crop_img = make_zoom_crop(test_img, cand, orig_size=tuple(meta['orig_size']),
                                      expand=expand, min_pad_frac=min_pad_frac,
                                      max_area_frac=max_area_frac).image
        if crop_img is not None:
            crop_rs = _smart_resize_image(crop_img, max_size, factor, crop_min_pixels, cap)
            c = _encode(prior, processor, [crop_rs], device)
        else:
            c = None
        # Per-action labels.
        row_labels = []
        for ai in range(len(ALL_ACTIONS)):
            new_boxes = apply_selected_action(trace.initial_boxes, idx, ai)
            row_labels.append(action_outcome_quality(
                new_boxes, meta, iou_threshold=iou_threshold, geometry_weight=geometry_weight))
        globals_.append(g)
        crops.append(c)
        boxes.append(torch.tensor(cand, dtype=torch.float32))
        labels.append(torch.tensor(row_labels, dtype=torch.float32))
        n_used += 1
        if max_samples is not None and n_used >= int(max_samples):
            break
    print(f'[box_outcome] collected {n_used} contexts, skipped {n_skipped} no-candidate samples', flush=True)
    return globals_, crops, boxes, labels


def stack_features(globals_, crops, boxes, labels, device):
    G = torch.stack(globals_).to(device)
    B = torch.stack(boxes).to(device)
    Y = torch.stack(labels).to(device)
    A = default_action_ids(len(globals_), device)
    has_crop = [c is not None for c in crops]
    C = None
    if any(has_crop):
        # For samples without a crop, feed a zero vector; the model learns to use
        # its ``no_crop_emb`` for degenerate crops only when the batch says so.
        D = int(globals_[0].shape[-1])
        C = torch.zeros(len(globals_), D, device=device)
        for i, c in enumerate(crops):
            if c is not None:
                C[i] = c.to(device)
    return G, C, B, A, Y


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--init-sft', required=True, help='frozen SFT checkpoint (merged LoRA)')
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--val-frac', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, processor, prior = load_sft_model(cfg, device, init_sft=args.init_sft)
    model.eval()

    train, _ = load_prior_split(cfg)
    train, _dev = split_holdout_by_class(train, float(cfg['data']['holdout_ratio']), seed=args.seed)
    pool = build_train_ref_pool(train)
    dataset = OutcomeMultiboxDataset(train, cfg, processor, 'train', pool)
    collator = OutcomeMultiboxCollator(processor, prior, cfg)

    rng = random.Random(args.seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    if args.max_samples is not None:
        indices = indices[:int(args.max_samples) * 2]  # oversample to absorb skips

    globals_, crops, boxes, labels = collect_features(
        model, processor, prior, dataset, collator, cfg, device, indices,
        args.max_samples, args.seed)
    if not globals_:
        raise RuntimeError('no candidate contexts collected; check the SFT checkpoint / zoom config')
    G, C, B, A, Y = stack_features(globals_, crops, boxes, labels, device)

    n = int(G.shape[0])
    n_val = max(1, int(n * args.val_frac))
    perm = torch.randperm(n)
    tr_idx, va_idx = perm[n_val:], perm[:n_val]

    dim = int(G.shape[-1])
    net = BoxOutcomeModel(global_dim=dim, crop_dim=dim, num_actions=len(ALL_ACTIONS)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    best_val = float('inf')
    out = Path(args.output)
    for epoch in range(1, args.epochs + 1):
        net.train()
        for s in range(0, len(tr_idx), args.batch_size):
            ids = tr_idx[s:s + args.batch_size]
            pred = net(G[ids], None if C is None else C[ids], B[ids], A[ids])
            loss = loss_fn(pred, Y[ids])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(net(G[va_idx], None if C is None else C[va_idx],
                                         B[va_idx], A[va_idx]), Y[va_idx]))
            tr_acc = float((net(G[tr_idx], None if C is None else C[tr_idx],
                                B[tr_idx], A[tr_idx]).argmax(-1) == Y[tr_idx].argmax(-1)).float().mean())
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f'[box_outcome] epoch={epoch} train_loss={float(loss):.4f} '
                  f'val_mse={val_loss:.4f} train_argmax_acc={tr_acc:.3f}', flush=True)
        if val_loss < best_val:
            best_val = val_loss
            save_box_outcome(net, out / 'box_outcome.pt')
    print(f'[box_outcome] best val_mse={best_val:.4f} -> {out / "box_outcome.pt"}', flush=True)


if __name__ == '__main__':
    main()
