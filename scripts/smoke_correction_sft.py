#!/usr/bin/env python3
"""Smoke-test the correction-continuation SFT path (train_region_sft.py).

Loads the SFT model + collator, then exercises the *new* stage-2 correction
samples on four sample types:

1. normal       — prior false alarms must be rejected to an empty answer.
2. empty-cand   — an anomaly with an empty artificial prefix (discover).
3. single-box   — an anomaly with one GT component.
4. multi-box    — an anomaly with >= 2 GT components (dropped/perturbed prefix).

For each correction sample it verifies:

  (a) ``plan_observations`` returns a sane observation set and the resulting
      batch's image count equals 2 (ref + test) + len(observations), i.e. the
      prompt's "Image 3 / Image 4 ..." numbering matches the real image count;
  (b) the artificial wrong stage-1 prefix (``pre_text``) is label=-100 — it is a
      replayed condition, never a supervised target.

Finally it runs ``pack_sft_batch`` + ``batch_loss`` to confirm the mixed H /
H-free packing and the forward pass still work end-to-end.

Usage:
  python scripts/smoke_correction_sft.py \
      --config configs/qwen35_2b_world_model_sft.yaml --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from outcome.observation import plan_observations
from outcome.inputs import build_observation_batch
from rl.grpo import move_batch
from train_region_sft import (batch_loss, build_correction_staged_targets,
                              build_labels, load_sft_model, pack_sft_batch)
from data.scan import load_prior_split
from data.prior_dataset import build_train_ref_pool
from outcome.inputs_multibox import OutcomeMultiboxCollator, OutcomeMultiboxDataset
from utils.config import load_yaml_config


def _comps(meta):
    comps = list(meta.get('component_bboxes') or [])
    if not comps and meta.get('gt_box_px') is not None:
        comps = [meta['gt_box_px']]
    return comps


def _check_correction(meta, collator, device, tokenizer, sft_cfg):
    pre_text, post_text, cand_boxes = build_correction_staged_targets(meta, collator, sft_cfg)
    observations = plan_observations(meta['test'], cand_boxes, tuple(meta['orig_size']), collator.cfg)

    from outcome.observation import observation_prompt
    prompt = observation_prompt(str(meta.get('class_name', 'object')), observations,
                                tuple(meta['orig_size']))
    zbatch = build_observation_batch(
        collator.processor, collator.prior, collator.cfg,
        meta['ref'], meta['test'],
        [o.image for o in observations],
        prompt, device,
        crop_min_pixels=None,
        prefill_text=pre_text,
    )
    zbatch = move_batch(zbatch, device)
    zprompt_ids = zbatch['input_ids'][0].tolist()
    post_ids = tokenizer(post_text, add_special_tokens=False).input_ids
    labels = build_labels(zprompt_ids, post_ids)

    n_images = int(zbatch['image_grid_thw'].shape[0])
    assert n_images == 2 + len(observations), \
        f'image count {n_images} != 2 + {len(observations)} observations'

    # (b) the wrong stage-1 prefix must be fully masked (label=-100).
    assert all(l == -100 for l in labels[:len(zprompt_ids)]), \
        'stage-1 prefix leaked into supervised labels'
    assert labels[len(zprompt_ids):] == post_ids, 'stage-2 target labels mismatch'

    decoded = tokenizer.decode(zprompt_ids, skip_special_tokens=True)
    assert pre_text.strip() in decoded, 'prefill (wrong prefix) not present in prompt'

    return dict(
        kind='normal' if not meta['is_anomaly'] else f'anom({len(_comps(meta))} comps)',
        n_obs=len(observations),
        obs_kinds=[o.kind for o in observations],
        n_images=n_images,
        pre_masked=True,
        pre_tokens=len(tokenizer(pre_text, add_special_tokens=False).input_ids),
        post_tokens=len(post_ids),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/qwen35_2b_world_model_sft.yaml')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    cfg.setdefault('runtime', {})['mode'] = 'train'
    cfg.setdefault('distributed', {})['num_gpu'] = 1
    device = torch.device(args.device)

    model, processor, prior = load_sft_model(cfg, device)
    tokenizer = getattr(processor, 'tokenizer', processor)

    train, _ = load_prior_split(cfg)
    pool = build_train_ref_pool(train)
    ds = OutcomeMultiboxDataset(train, cfg, processor, 'train', pool)
    collator = OutcomeMultiboxCollator(processor, prior, cfg)

    anom_idx = [i for i in range(len(ds)) if ds[i].get('is_anomaly')]
    norm_idx = [i for i in range(len(ds)) if not ds[i].get('is_anomaly')]
    print(f'[smoke] dataset: {len(anom_idx)} anom, {len(norm_idx)} norm', flush=True)

    single = multi = None
    for i in anom_idx:
        n = len(_comps(ds[i]))
        if n == 1 and single is None:
            single = i
        if n >= 2 and multi is None:
            multi = i
        if single is not None and multi is not None:
            break
    print(f'[smoke] single-box={single} multi-box={multi} normal={norm_idx[0] if norm_idx else None}',
          flush=True)

    random.seed(0)
    base_sft = dict(cfg['outcome'].get('sft') or {})

    results = []
    # normal
    if norm_idx:
        sft_cfg = dict(base_sft)
        sft_cfg['zoom_prob'] = 1.0
        r = _check_correction(ds[norm_idx[0]], collator, device, tokenizer, sft_cfg)
        results.append(r)
    # single-box anomaly (perturbed)
    if single is not None:
        sft_cfg = dict(base_sft)
        sft_cfg['zoom_prob'] = 1.0
        sft_cfg['correction_p_empty'] = 0.0
        sft_cfg['correction_p_drop'] = 0.0
        results.append(_check_correction(ds[single], collator, device, tokenizer, sft_cfg))
    # multi-box anomaly (dropped/perturbed)
    if multi is not None:
        sft_cfg = dict(base_sft)
        sft_cfg['zoom_prob'] = 1.0
        sft_cfg['correction_p_empty'] = 0.0
        sft_cfg['correction_p_drop'] = 1.0
        results.append(_check_correction(ds[multi], collator, device, tokenizer, sft_cfg))
    # empty-candidate anomaly (discover)
    if single is not None:
        sft_cfg = dict(base_sft)
        sft_cfg['zoom_prob'] = 1.0
        sft_cfg['correction_p_empty'] = 1.0
        sft_cfg['correction_p_drop'] = 0.0
        results.append(_check_correction(ds[single], collator, device, tokenizer, sft_cfg))

    for r in results:
        print(f"[smoke] {r['kind']}: n_obs={r['n_obs']} kinds={r['obs_kinds']} "
              f"n_images={r['n_images']} pre_masked={r['pre_masked']} "
              f"pre_tokens={r['pre_tokens']} post_tokens={r['post_tokens']}", flush=True)

    # Full mixed pack + loss (H singles + H-free correction continuations).
    sample_idxs = [i for i in (norm_idx[:1] + [single, multi]) if i is not None]
    samples = [ds[i] for i in sample_idxs]
    sft_cfg = dict(base_sft)
    sft_cfg['zoom_prob'] = 1.0
    packed_list, n_sup = pack_sft_batch(collator, device, samples, tokenizer,
                                        multibox=True, thinking=True, answer_only=False,
                                        sft_cfg=sft_cfg)
    print(f'[smoke] packed sub-batches={len(packed_list)} supervised_tokens={n_sup}', flush=True)
    for k, packed in enumerate(packed_list):
        loss = batch_loss(model, packed)
        print(f'[smoke] sub-batch {k}: loss={float(loss.detach()):.6f} '
              f'image_embeds={packed["image_embeds"].shape} '
              f'grid={packed["gen_in"]["image_grid_thw"].shape}', flush=True)

    print('[smoke] DONE — all correction-path checks passed', flush=True)


if __name__ == '__main__':
    main()
