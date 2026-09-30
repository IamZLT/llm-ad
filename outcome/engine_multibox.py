"""outcome-multibox-v1 training loop (bounded, single/multi-GPU).

Same GRPO skeleton as outcome-v1 but with component-level metrics, set-level
reward and localization-collapse resampling (Range(R_set)). Evaluation lives in
``outcome/evaluate_multibox.py``; this module imports only the mid-training
"small eval" entry point.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from data.prior_dataset import build_train_ref_pool
from data.scan import load_prior_split, split_holdout_by_class
from models.anomaly_prior import AnomalyPrior
from models.lora import apply_lora
from models.qwen35 import setup_model_and_processor, freeze_vision_encoder, force_vision_eval, unwrap_model
from outcome.evaluate_multibox import evaluate, make_record
from outcome.inputs import build_zoom_train_batch
from outcome.inputs_multibox import OutcomeMultiboxCollator, OutcomeMultiboxDataset
from outcome.policy import (generate_group, generate_inspection_rollouts, group_advantages,
                             optimize_group, optimize_inspection_group)
from outcome.zoom_crop import make_zoom_crop
from outcome.protocol_multibox import VERSION, parse_output_cfg, score_output
from outcome.thinking import thinking_enabled as _thinking_enabled
from outcome.visualize_multibox import log_outcome_train_grid
from rl.grpo import avg_across_ranks, move_batch
from utils.common import is_main_process, set_seed


def validate_config(cfg):
    if cfg.get('outcome', {}).get('version') != VERSION:
        raise ValueError(f'expected outcome.version={VERSION}')
    think_on = _thinking_enabled(cfg)
    enable_thinking = bool(cfg.get('prompt', {}).get('enable_thinking', False))
    if think_on != enable_thinking:
        raise ValueError(
            'outcome.thinking.enabled and prompt.enable_thinking must match '
            '(true = Qwen native CoT; false = five-block XML)'
        )
    if not cfg['model'].get('freeze_vit', True) or not cfg['lora'].get('enabled', False):
        raise ValueError('outcome-multibox-v1 requires frozen ViT and language LoRA')
    gc = cfg['grpo']
    if (float(gc.get('temperature', 1)) != 1 or float(gc.get('top_p', 1)) != 1
            or int(gc.get('top_k', 0)) != 0):
        raise ValueError('raw-policy baseline requires temperature=1, top_p=1, top_k=0')
    if int(gc.get('policy_epochs', 1)) < 1 or int(gc.get('gradient_accumulation_steps', 1)) != 1:
        raise ValueError('outcome-multibox-v1 requires policy_epochs>=1 and gradient_accumulation_steps=1')
    if int(gc['group_size']) < 2 or int(gc['max_new_tokens']) < 1:
        raise ValueError('group_size >= 2 and positive generation budget required')
    if gc.get('reward'):
        raise ValueError('remove legacy grpo.reward: outcome uses only outcome.protocol_weight')
    if not 0 <= float(cfg.get('outcome', {}).get('protocol_weight', .01)) <= .1:
        raise ValueError('protocol_weight must be in [0,.1]')
    if int(cfg.get('outcome', {}).get('max_boxes', 16)) < 1:
        raise ValueError('outcome.max_boxes must be positive')
    if Path(cfg['model']['name'], 'adapter_config.json').exists():
        raise ValueError('model.name must be the base model; use outcome.sft_adapter or --adapter explicitly')


def _shrink_box_1000(box, scale):
    """Shrink a 0-1000 box about its center (scheme-B zoom candidate)."""
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


def _maybe_zoom_train_batch(cfg, processor, prior, device, batch, rng):
    """Scheme-B zoom: upgrade single-box anomalous samples to a 3-image batch.

    The crop is built from a *fixed* shrink-GT candidate box (not the model's own
    ``[localize]`` sample), so the whole GRPO group shares one 3-image vision cache
    and the standard group rollout / optimize path is reused unchanged. This keeps
    train and eval consistent (both feed a zoomed, outline-drawn crop) without the
    per-trajectory crop that would break batched group sampling.
    """
    zcfg = (cfg.get('outcome') or {}).get('zoom') or {}
    if not bool(zcfg.get('enabled', False)):
        return batch
    meta = batch['_meta'][0]
    if not bool(meta.get('is_anomaly')):
        return batch
    comps = list(meta.get('component_bboxes') or [])
    if not comps and meta.get('gt_box_px') is not None:
        comps = [meta['gt_box_px']]
    if len(comps) != 1:
        return batch
    if rng.random() >= float(zcfg.get('train_prob', 0.3)):
        return batch
    orig_size = meta.get('orig_size')
    if not orig_size or meta.get('test') is None:
        return batch
    w, h = float(orig_size[0]), float(orig_size[1])
    gt_1000 = [comps[0][0] * 1000.0 / w, comps[0][1] * 1000.0 / h,
               comps[0][2] * 1000.0 / w, comps[0][3] * 1000.0 / h]
    scale = max(0.1, min(float(zcfg.get('shrink_scale', 0.5)), 0.95))
    cand = _shrink_box_1000(gt_1000, scale)
    crop = make_zoom_crop(
        meta['test'], cand, orig_size=tuple(orig_size),
        expand=float(zcfg.get('expand', 1.0)),
        min_pad_frac=float(zcfg.get('min_pad_frac', 0.12)),
        max_area_frac=float(zcfg.get('max_area_frac', 0.6)),
    )
    if crop.degenerate:
        return batch
    return build_zoom_train_batch(processor, prior, cfg, device, batch, crop.image)


def load_model(cfg, adapter=None, fresh_lora=True, resume_adapter=None):
    from peft import PeftModel
    model, processor = setup_model_and_processor(cfg, for_inference=False, freeze_vision=True)
    h_box_cfg = (cfg.get('outcome', {}) or {}).get('h_box_prior', {}) or {}
    h_vpt_cfg = (cfg.get('outcome', {}) or {}).get('h_vpt', {}) or {}
    h_box_enabled = bool(h_box_cfg.get('enabled', False))
    h_vpt_enabled = bool(h_vpt_cfg.get('enabled', False))
    if h_box_enabled:
        # Register <|h_box|> + resize embedding BEFORE merging the SFT adapter so the
        # (resized) SFT embedding aligns with the base model.
        from models.h_box_prior import ensure_box_token
        ensure_box_token(processor, model)
    if h_vpt_enabled:
        # Register <|h_ctrl|> + <|h_feat|> before merging the SFT adapter.
        from models.h_vpt import ensure_control_token
        ensure_control_token(processor, model)
    sft = cfg.get('outcome', {}).get('sft_adapter')
    if sft:
        model = PeftModel.from_pretrained(model, sft, is_trainable=False).merge_and_unload()
    if adapter and resume_adapter:
        raise ValueError('adapter and resume_adapter cannot be used together')
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    elif resume_adapter:
        # Continue the existing RL LoRA. Optimizer / attempt counter still start fresh.
        model = PeftModel.from_pretrained(model, resume_adapter, is_trainable=True)
        print(f'[multibox] resume RL LoRA from {resume_adapter} (optimizer resets)', flush=True)
    elif fresh_lora:
        model = apply_lora(model, cfg)
    freeze_vision_encoder(model)
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', '0')))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        model = model.to(local_rank)
    else:
        model = model.to('cpu')
    if not adapter and cfg['training'].get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    force_vision_eval(model)
    prior = AnomalyPrior.from_qwen(model, cfg)
    if h_box_enabled:
        # Load the frozen HBoxProjector (trained by train_region_sft.py).
        from models.h_box_prior import attach_h_box_prior
        attach_h_box_prior(model, cfg, sft)
    if h_vpt_enabled:
        # Load the HVPT module (trained by train_region_sft.py). Frozen by default;
        # outcome.h_vpt.trainable controls which submodules RL may update.
        from models.h_vpt import attach_h_vpt
        attach_h_vpt(model, cfg, sft)
    return model, processor, prior


def datasets(cfg, processor):
    train, test = load_prior_split(cfg)
    train, dev = split_holdout_by_class(train, float(cfg['data']['holdout_ratio']), seed=int(cfg['training']['seed']))
    pool = build_train_ref_pool(train)
    return (OutcomeMultiboxDataset(train, cfg, processor, 'train', pool),
            OutcomeMultiboxDataset(dev, cfg, processor, 'eval', pool),
            OutcomeMultiboxDataset(test, cfg, processor, 'eval'))


class _NullWriter:
    """No-op SummaryWriter stand-in for non-main ranks (swallows all logging calls)."""

    def add_scalar(self, *a, **k): pass
    def add_text(self, *a, **k): pass
    def add_image(self, *a, **k): pass
    def flush(self): pass
    def close(self): pass


def run_train(cfg, model, processor, prior, train_set, dev_set, test_set, output_dir):
    oc, gc = cfg['outcome'], cfg['grpo']
    collator = OutcomeMultiboxCollator(processor, prior, cfg)

    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', '0')))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', str(local_rank)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend='nccl', timeout=timedelta(hours=2))
    if world > 1:
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=bool(cfg.get('distributed', {}).get('ddp_find_unused_parameters', False)),
        )
    main_proc = is_main_process()
    device = next(model.parameters()).device

    if not len(train_set):
        raise ValueError('empty train set')
    requested = gc.get('max_attempts')
    total_attempts = int(requested) if requested is not None else math.ceil(len(train_set)*float(gc['epochs']))
    if total_attempts <= 0:
        raise ValueError('max_attempts must be positive')
    # `attempts` is per-rank; each rank processes a disjoint shard so the whole job
    # covers `total_attempts` samples with a `world`x wall-clock speedup.
    attempts = max(1, math.ceil(total_attempts / world))
    max_boxes = int(oc.get('max_boxes', 16))
    eval_which = str(cfg['training'].get('eval_split', 'dev')).lower()
    if eval_which == 'test':
        eval_set, eval_name = test_set, 'test'
    elif eval_which == 'dev':
        eval_set, eval_name = dev_set, 'dev'
    else:
        eval_set, eval_name = None, None
    # H-VPT projector (when trainable) gets its own (larger) LR so the gate can
    # actually open under reward pressure. The scalar gate + cross-attn weights
    # barely move at the RL base LR, which would leave the H channel effectively
    # frozen despite being marked trainable.
    h_vpt = getattr(unwrap_model(model), 'h_vpt', None)
    hc = oc.get('h_vpt') or {}
    h_vpt_lr = float(hc.get('lr', float(gc['learning_rate']) * 100.0))
    if h_vpt is not None and any(p.requires_grad for p in h_vpt.projector.parameters()):
        proj, rest = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (proj if 'h_vpt.projector' in n else rest).append(p)
        opt = torch.optim.AdamW(
            [{'params': rest, 'lr': float(gc['learning_rate'])},
             {'params': proj, 'lr': h_vpt_lr}], weight_decay=0.)
        if main_proc:
            print(f'[multibox] h_vpt projector trainable: {len(proj)} params @ lr={h_vpt_lr} '
                  f'(base={gc["learning_rate"]})', flush=True)
    else:
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=float(gc['learning_rate']), weight_decay=0.)
    if main_proc:
        writer = SummaryWriter(str(Path(output_dir)/'tb'))
        writer.add_text('outcome/0_config', json.dumps({
            'outcome': oc, 'grpo': gc, 'lora': cfg.get('lora'),
            'prior': cfg.get('prior'), 'training': cfg.get('training'),
            'tensorboard': cfg.get('tensorboard'),
        }, ensure_ascii=False, indent=2, default=str), 0)
        writer.flush()
    else:
        writer = _NullWriter()
    updates = skipped = 0
    case_buffer = []
    rng = random.Random(int(cfg['training']['seed']))
    order = []
    output_dir = Path(output_dir)
    if main_proc:
        manifest = {name:[s.get('full_img_path') or s.get('image') for s in ds.samples]
                    for name,ds in [('train',train_set),('dev',dev_set),('test',test_set)]}
        (output_dir/'split_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    try:
        loc_cfg = {**(oc.get('localization') or {}), **(oc.get('reward') or {}),
                   **(oc.get('planner') or {})}
        zoom_enabled = bool((oc.get('zoom') or {}).get('enabled', False))
        resample_cfg = oc.get('resampling') or {}
        max_group_resamples = int(resample_cfg.get('max_group_resamples', 3))
        min_loc_range = float(resample_cfg.get('min_loc_range', 0.001))
        stream_path = output_dir/'rollouts.jsonl' if main_proc else None
        with (stream_path.open('w') if stream_path is not None else open(os.devnull, 'w')) as stream:
            for attempt in range(1, attempts+1):
                if not order:
                    idx = list(range(len(train_set)))
                    rng.shuffle(idx)
                    order = idx[rank::world]
                batch = move_batch(collator([train_set[order.pop()]]), device)
                # Scheme-B fixed shrink-GT zoom is only for the single-pass path;
                # the two-stage flow crops the model's own B0 inside the rollout.
                if not zoom_enabled:
                    batch = _maybe_zoom_train_batch(cfg, processor, prior, device, batch, rng)
                started = time.perf_counter()
                meta = batch['_meta'][0]
                is_anomaly = bool(meta.get('is_anomaly'))
                completions = parsed = scores = None
                traces = None
                task_std = loc_std = loc_range = 0.0
                rollout_sec = 0.0
                for resamples in range(max_group_resamples + 1):
                    t_gen = time.perf_counter()
                    if zoom_enabled:
                        rollouts = generate_inspection_rollouts(model, processor, prior, batch, cfg,
                                                                group=int(gc['group_size']), sample=True)
                        completions = [c for c, _t in rollouts]
                        traces = [t for _c, t in rollouts]
                    else:
                        rollouts = None
                        completions = generate_group(model, processor, batch, cfg, group=int(gc['group_size']), sample=True)
                    rollout_sec += time.perf_counter() - t_gen
                    parsed = [parse_output_cfg(c.text, cfg, max_boxes=max_boxes) for c in completions]
                    scores = [score_output(p, meta, float(oc['protocol_weight']), loc_cfg,
                                           max_boxes=max_boxes, trace=t)
                              for p, t in zip(parsed, traces)] if traces is not None else [
                        score_output(p, meta, float(oc['protocol_weight']), loc_cfg, max_boxes=max_boxes)
                        for p in parsed]
                    task_rewards = torch.tensor([s['task'] for s in scores], device=device)
                    loc_rewards = torch.tensor([s['loc_reward'] for s in scores], device=device)
                    task_std = float(task_rewards.std(unbiased=False))
                    loc_std = float(loc_rewards.std(unbiased=False))
                    loc_range = float(loc_rewards.max() - loc_rewards.min())
                    # Set-level localization collapse: resample only when the
                    # anomaly group's R_set has (near) zero range.
                    if not is_anomaly or loc_range >= min_loc_range:
                        break
                rewards = torch.tensor([s['total'] for s in scores], device=device)
                # Advantage uses task reward only; the tiny protocol term (0.01) must
                # not leak into the advantage, or it becomes the sole (std-scaled)
                # gradient whenever the group's task rewards collapse to a constant.
                advantages = group_advantages(task_rewards, bool(gc.get('scale_rewards', False)))
                # Localization gets its own group-normalized advantage so coordinate
                # tokens are not averaged away by the long prose/confirm text. When
                # it collapses (normal samples: loc_reward is always 0), fall back to
                # the task advantage so their [localize] candidate boxes still learn.
                loc_advantages = (group_advantages(loc_rewards, bool(gc.get('scale_rewards', False)))
                                  if bool(gc.get('per_token_advantage', True)) else None)
                if loc_advantages is not None and bool(loc_advantages.abs().max().item() <= 1e-8):
                    loc_advantages = None
                # Candidate (B0) quality advantage: in the two-stage flow the initial
                # boxes carry Q0 while the final boxes carry Q1 (=loc_rewards), so the
                # candidate tokens are not scored by the final-box signal.
                cand_rewards = torch.tensor([s.get('q0', 0.0) for s in scores], device=device)
                cand_advantages = (group_advantages(cand_rewards, bool(gc.get('scale_rewards', False)))
                                   if zoom_enabled and bool(gc.get('per_token_advantage', True)) else None)
                if cand_advantages is not None and bool(cand_advantages.abs().max().item() <= 1e-8):
                    cand_advantages = None
                # Rehearsal/verification calibration advantage: [imagine]/[confirm]
                # tokens get a focused advantage from the discriminator + imagine
                # calibration signal, so the "is my box right?" reasoning learns.
                verify_rewards = torch.tensor([s.get('verify_signal', 0.0) for s in scores], device=device)
                verify_advantages = (group_advantages(verify_rewards, bool(gc.get('scale_rewards', False)))
                                     if bool(gc.get('per_token_advantage', True)) else None)
                if verify_advantages is not None and bool(verify_advantages.abs().max().item() <= 1e-8):
                    verify_advantages = None
                planner_advantages = confirm_advantages = None
                if traces is not None and bool(gc.get('per_token_advantage', True)):
                    planner_rewards = torch.tensor([s.get('planner_signal', 0.0) for s in scores], device=device)
                    confirm_rewards = torch.tensor([s.get('confirm_signal', 0.0) for s in scores], device=device)
                    planner_advantages = group_advantages(planner_rewards, bool(gc.get('scale_rewards', False)))
                    confirm_advantages = group_advantages(confirm_rewards, bool(gc.get('scale_rewards', False)))
                    if bool(planner_advantages.abs().max().item() <= 1e-8):
                        planner_advantages = None
                    if bool(confirm_advantages.abs().max().item() <= 1e-8):
                        confirm_advantages = None
                    reward_cfg = oc.get('reward') or {}
                    if planner_advantages is not None:
                        planner_advantages = planner_advantages * float(reward_cfg.get('planner_weight', 0.1))
                    if confirm_advantages is not None:
                        confirm_advantages = confirm_advantages * float(reward_cfg.get('confirm_weight', 0.1))
                    verify_advantages = None
                # Skip the update only when EVERY valid-token advantage collapsed
                # (task + candidate + final + verify), not just the task term.
                zero = bool(advantages.abs().max().item() <= 1e-8)
                for a in (cand_advantages, loc_advantages, verify_advantages,
                          planner_advantages, confirm_advantages):
                    if a is not None:
                        zero = zero and bool(a.abs().max().item() <= 1e-8)
                if world > 1:
                    zero_t = torch.tensor([1 if zero else 0], device=device, dtype=torch.int64)
                    dist.all_reduce(zero_t, op=dist.ReduceOp.SUM)
                    all_zero = int(zero_t.item()) == world
                else:
                    all_zero = zero
                skipped += int(zero)
                loc_collapsed_group = float(is_anomaly and loc_range < min_loc_range)
                collapsed_after_resampling = bool(is_anomaly and loc_range < min_loc_range)
                loc_nonzero_rate = float((loc_rewards > 1e-6).float().mean())
                metrics = dict(attempts=attempt, updates=updates, skipped_total=skipped, zero_advantage_group=float(zero),
                    reward_mean=float(rewards.mean()), reward_std=float(rewards.std(unbiased=False)),
                    task_reward_mean=sum(s['task'] for s in scores)/len(scores),
                    task_reward_std=task_std, resamples_used=resamples, loc_collapsed_group=loc_collapsed_group,
                    collapsed_after_resampling=float(collapsed_after_resampling),
                    loc_reward_mean=float(loc_rewards.mean()), loc_reward_std=loc_std, loc_reward_range=loc_range,
                    loc_nonzero_rate=loc_nonzero_rate,
                    task_valid_rate=sum(p['task_valid'] for p in parsed)/len(parsed),
                    protocol_core_rate=sum(p['protocol_core'] for p in parsed)/len(parsed),
                    protocol_strict_rate=sum(p['protocol_strict'] for p in parsed)/len(parsed),
                    think_ok_rate=sum(bool(p.get('think_ok')) for p in parsed)/len(parsed),
                    think_filled_rate=sum(bool(p.get('think_filled')) for p in parsed)/len(parsed),
                    truncation_rate=sum(c.stop_reason == 'length' for c in completions)/len(completions))
                if traces:
                    def _defined(values):
                        nums = [float(v) for v in values if v is not None]
                        return sum(nums) / len(nums) if nums else 0.0
                    actions = [((t.rounds[-1].selected_observation_action if t.rounds else None)
                                 or t.selected_action or '') for t in traces]
                    metrics.update(
                        planner_valid_rate=sum(bool(t.rounds and t.rounds[-1].world_model_valid) for t in traces) / len(traces),
                        zoom_rate=sum(a.startswith('zoom_box_') for a in actions) / len(actions),
                        global_scan_rate=sum(a == 'global_scan' for a in actions) / len(actions),
                        stop_rate=sum(a == 'stop' for a in actions) / len(actions),
                        mean_predicted_gain=_defined(s.get('predicted_gain') for s in scores),
                        mean_actual_gain=_defined(
                            (t.rounds[-1].actual_gain if t.rounds else None) for t in traces),
                        mean_gain_error=_defined(s.get('gain_error') for s in scores),
                        mean_planner_regret=_defined(s.get('planner_regret') for s in scores),
                        mean_quality_before=_defined(s.get('state_q0') for s in scores),
                        mean_quality_after=_defined(s.get('state_q1') for s in scores),
                    )
                metrics.update(prompt_tokens=meta['prompt_tokens'], visual_tokens=meta['visual_tokens'],
                               mean_new_tokens=sum(len(c.ids)-int(batch['prompt_len'][0]) for c in completions)/len(completions))
                # Average the per-sample metrics across ranks so TB/console show the
                # global value; the counters (attempts/updates/skipped_total) are kept.
                metrics = {k: (avg_across_ranks(float(v), device) if k not in ('attempts', 'updates', 'skipped_total') else v)
                           for k, v in metrics.items()}
                rows = [make_record(p, s, meta, c, int(batch['prompt_len'][0]), 0., max_boxes, trace=t)
                        for p, s, c, t in zip(parsed, scores, completions, traces or [None] * len(parsed))]
                stream.write(json.dumps(dict(attempt=attempt, update_before=updates, zero_advantage=zero,
                                             task_reward_std=task_std, resamples_used=resamples,
                                             loc_reward_mean=float(loc_rewards.mean()), loc_reward_std=loc_std,
                                             loc_reward_range=loc_range, loc_collapsed_group=loc_collapsed_group,
                                             advantages=advantages.cpu().tolist(), trajectories=rows), ensure_ascii=False, default=str)+'\n')
                stream.flush()
                for name,value in metrics.items():
                    writer.add_scalar(f'train/{name}', value, attempt)
                split_prefix = 'train_anomaly' if is_anomaly else 'train_normal'
                for name,value in metrics.items():
                    writer.add_scalar(f'{split_prefix}/{name}', value, attempt)
                writer.flush()
                loss_stats = None
                if not all_zero:
                    if zoom_enabled and traces is not None:
                        # Real two-stage trajectory: each trace carries its own
                        # per-segment conditioning (stage-2 crops differ), so use
                        # the two-stage optimizer. Candidate/final boxes carry their
                        # own Q0/Q1 advantages, decoupled from the max() entanglement.
                        loss_stats = optimize_inspection_group(
                            model, processor, rollouts, advantages, opt, cfg, skip=zero,
                            cand_advantages=cand_advantages,
                            final_advantages=loc_advantages,
                            verify_advantages=verify_advantages,
                            imagine_advantages=planner_advantages,
                            confirm_advantages=confirm_advantages)
                    else:
                        loss_stats = optimize_group(model, processor, batch, completions, advantages, opt, cfg, skip=zero,
                                                    loc_advantages=loc_advantages,
                                                    verify_advantages=verify_advantages)
                    updates += 1
                    for name,value in loss_stats.items():
                        writer.add_scalar(f'optimizer/{name}', value, updates)
                metrics['collapsed_but_updated'] = float(collapsed_after_resampling and not zero)
                writer.add_scalar('train/collapsed_but_updated', metrics['collapsed_but_updated'], attempt)
                writer.add_scalar('train/updates_after', updates, attempt)
                writer.add_scalar('train/seconds_per_attempt', time.perf_counter()-started, attempt)
                writer.add_scalar('train/seconds_rollout', rollout_sec, attempt)

                def _mean(key, rows):
                    vals = [r[key] for r in rows if r.get(key) is not None]
                    return sum(vals) / len(vals) if vals else None
                anom_rows = [r for r in rows if r.get('is_anomaly')]
                mean_maskiou = _mean('mask_iou', anom_rows)
                mean_delta = _mean('delta_refine', anom_rows)
                mean_cand_cov = _mean('cand_coverage', anom_rows)
                mean_iou_h = _mean('iou_h_bestk', anom_rows)
                mean_loc = sum(s['loc_reward'] for s in scores) / len(scores)

                if loss_stats is not None:
                    ls = loss_stats
                    loss_part = (f"loss={ls.get('loss', float('nan')):.4f} "
                                 f"pg={ls.get('pg', float('nan')):.4f} "
                                 f"kl={ls.get('kl', float('nan')):.3f} "
                                 f"ratio={ls.get('ratio', float('nan')):.3f} "
                                 f"clip={ls.get('clip_fraction', float('nan')):.2f} "
                                 f"gnorm={ls.get('grad_norm', float('nan')):.2f} "
                                 f"ltok={ls.get('loc_tokens', 0)}")
                else:
                    loss_part = 'loss=-- pg=-- kl=-- ratio=-- clip=-- gnorm=-- ltok=--'

                def _f(v):
                    return f'{v:.3f}' if v is not None else '--'

                if main_proc:
                    print(f'[multibox] a={attempt}/{attempts} up={updates} sk={skipped} '
                          f'rw={metrics["reward_mean"]:.3f} loc={_f(mean_loc)} lrng={loc_range:.4f} '
                          f'lz={loc_nonzero_rate:.2f} lc={loc_collapsed_group:.0f} '
                          f'maskiou={_f(mean_maskiou)} delta={_f(mean_delta)} ccov={_f(mean_cand_cov)} iou_h={_f(mean_iou_h)} '
                          f'rs={resamples} tv={metrics["task_valid_rate"]:.2f} '
                          f'pc={metrics["protocol_core_rate"]:.2f} ps={metrics["protocol_strict_rate"]:.2f} '
                          f'tk={metrics["think_ok_rate"]:.2f} tok={metrics["mean_new_tokens"]:.0f} '
                          f'pvalid={metrics.get("planner_valid_rate", 0):.2f} '
                          f'zoom={metrics.get("zoom_rate", 0):.2f} '
                          f'scan={metrics.get("global_scan_rate", 0):.2f} '
                          f'stop={metrics.get("stop_rate", 0):.2f} '
                          f'pgain={_f(metrics.get("mean_predicted_gain"))} '
                          f'again={_f(metrics.get("mean_actual_gain"))} '
                          f'gerr={_f(metrics.get("mean_gain_error"))} '
                          f'regret={_f(metrics.get("mean_planner_regret"))} '
                          f'{loss_part} '
                          f'({time.perf_counter()-started:.1f}s)', flush=True)
                every = int(cfg['training'].get('eval_every_n_steps', 0))
                vis_every = int((cfg.get('tensorboard') or {}).get('vis_every_n_steps', 0) or 0)
                vis_n = int((cfg.get('tensorboard') or {}).get('vis_num_samples', 5) or 5)
                if main_proc and vis_every > 0:
                    best = max(range(len(parsed)), key=lambda i: (
                        scores[i]['correct'],
                        scores[i]['union_iou'],
                        scores[i]['loc_reward'],
                    ))
                    case_buffer.append(dict(
                        meta=meta, response=completions[best].text, parsed=parsed[best],
                        union_iou=scores[best]['union_iou'], loc_reward=scores[best]['loc_reward'],
                        correct=scores[best]['correct'], reward=scores[best]['total'],
                    ))
                    if attempt % vis_every == 0 and case_buffer:
                        picks = random.sample(case_buffer, min(vis_n, len(case_buffer)))
                        log_outcome_train_grid(writer, step=attempt, cases=picks, max_cases=vis_n)
                        case_buffer.clear()
                if every > 0 and attempt % every == 0 and eval_set is not None and len(eval_set):
                    if main_proc:
                        evaluate(cfg, unwrap_model(model), processor, prior, eval_set, output_dir/f'{eval_name}_{attempt:06d}.json',
                                 cfg['training'].get('eval_num_samples'), writer, attempt, eval_name)
                    if world > 1 and dist.is_initialized():
                        dist.barrier()
                save = int(gc.get('save_steps', 0))
                if save > 0 and attempt % save == 0:
                    if main_proc:
                        unwrap_model(model).save_pretrained(output_dir/f'checkpoint-{attempt}')
                        processor.save_pretrained(output_dir/f'checkpoint-{attempt}')
                        from models.h_vpt import save_h_vpt
                        save_h_vpt(getattr(unwrap_model(model), 'h_vpt', None),
                                   output_dir/f'checkpoint-{attempt}'/'h_vpt.pt')
                    if world > 1 and dist.is_initialized():
                        dist.barrier()
        if main_proc:
            unwrap_model(model).save_pretrained(output_dir/'adapter_final')
            processor.save_pretrained(output_dir/'adapter_final')
            from models.h_vpt import save_h_vpt
            save_h_vpt(getattr(unwrap_model(model), 'h_vpt', None),
                       output_dir/'adapter_final'/'h_vpt.pt')
        if main_proc:
            (output_dir/'training_summary.json').write_text(json.dumps(dict(attempts=attempts,updates=updates,skipped=skipped), indent=2))
    finally:
        writer.close()
        if world > 1 and dist.is_initialized():
            dist.barrier()
