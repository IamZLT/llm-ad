"""Finite rollouts, genuine completion lengths and one outcome advantage."""
from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
from transformers import GenerationConfig, StoppingCriteriaList, StopStringCriteria

from models.qwen35 import force_vision_eval, unwrap_model
from models.vision_cache import bind_cached_image_features
from models.region_injection import bind_region_injection, has_region, region_raw_from_batch
from rl.grpo import (clipped_pg_kl, disable_adapter_ctx, dropout_eval, expand_gen_in_for_group,
                     forward_with_vision, micro_batch_ranges, model_inputs, padded_completion_tensors,
                     token_logprobs, token_logprobs_nograd)


@dataclass
class Completion:
    ids: torch.Tensor
    text: str
    stop_reason: str
    sampled_mask: list = field(default_factory=list)

    stage_hits: dict = field(default_factory=dict)
    stage_lengths: dict = field(default_factory=dict)


def eos_ids(model, tokenizer):
    value = getattr(getattr(unwrap_model(model), 'generation_config', None), 'eos_token_id', None)
    ids = set(value if isinstance(value, (list, tuple)) else [value] if value is not None else [])
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    return sorted(ids)


def trim_completion(row, prompt_len, tokenizer, end_ids):
    """Keep first real EOS, or the token completing </answer>; never train batch padding.

    Inspect only completion tokens, because prompt contains the output template.
    A stop string can span tokens or end within a token; include that whole token.
    """
    tokens = row[prompt_len:].detach().cpu().tolist()
    prefix = []
    reason = 'length'
    for token in tokens:
        prefix.append(token)
        if '</answer>' in tokenizer.decode(prefix, skip_special_tokens=True):
            reason = 'answer'
            break
        if token in end_ids:
            reason = 'eos'
            break
    end = prompt_len+len(prefix)
    ids = row[:end].detach().clone()
    return Completion(ids, tokenizer.decode(ids[prompt_len:], skip_special_tokens=True), reason)


def group_advantages(rewards, scale=False, eps=1e-6):
    if bool((rewards.max() == rewards.min()).item()):
        return torch.zeros_like(rewards)
    adv = rewards-rewards.mean()
    if scale:
        adv = adv/rewards.std(unbiased=False).clamp(min=eps)
    return adv


def _region_bind(model, batch):
    """Region-token injection context for one forward; no-op without a mounted adapter.

    Returns a FRESH context manager on every call so it can be reused across the
    multiple ``with`` blocks inside ``optimize_group`` (a generator-based context
    manager cannot be re-entered).
    """
    adapter = getattr(unwrap_model(model), 'region_adapter', None)
    if adapter is None or not has_region(batch):
        return nullcontext()
    return bind_region_injection(model, adapter, region_raw_from_batch(batch), int(batch['region_token_id']))


def generate_group(model, processor, batch, cfg, group=1, sample=False):
    tokenizer = getattr(processor, 'tokenizer', processor)
    gcfg = cfg['grpo']
    if bool(gcfg.get('shared_prefill', False)):
        return generate_group_shared_prefill(model, processor, batch, cfg, group=group, sample=sample)
    limit = int(gcfg['max_new_tokens'])
    ends = eos_ids(model, tokenizer)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else ends[0]
    # A fresh config removes inherited top-k/penalties/forced tokens. Raw-policy
    # categorical sampling is exactly the distribution used by all logprobs.
    # Looped models use a per-depth LoopDepthCache (use_cache=True); plain models
    # use the vanilla cache.
    generation = GenerationConfig(max_new_tokens=limit, do_sample=sample,
        temperature=1.0, top_p=1.0, top_k=0, typical_p=1.0, repetition_penalty=1.0,
        eos_token_id=ends or None, pad_token_id=pad, use_cache=True, num_beams=1)
    stops = StoppingCriteriaList([StopStringCriteria(tokenizer=tokenizer, stop_strings=['</answer>'])])
    core = unwrap_model(model)
    was_training = core.training
    core.eval()
    result = []
    try:
        with torch.no_grad(), bind_cached_image_features(core, batch['image_embeds']), _region_bind(model, batch):
            for start, end in micro_batch_ranges(group, int(gcfg.get('rollout_micro_batch_size', 1))):
                generation.num_return_sequences = end-start
                generated = core.generate(**model_inputs(batch), generation_config=generation,
                                          stopping_criteria=stops)
                for row in generated:
                    result.append(trim_completion(row, int(batch['prompt_len'][0]), tokenizer, ends))
    finally:
        core.train(was_training)
        force_vision_eval(core)
    return result


def _sample_tokens(logits: torch.Tensor, sample: bool) -> torch.Tensor:
    """Raw categorical sampling (temperature=1, top_p=1, top_k=0) or greedy argmax."""
    if sample:
        probs = torch.softmax(logits.float(), dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    return logits.argmax(dim=-1)


def generate_group_shared_prefill(model, processor, batch, cfg, group=1, sample=False):
    """GRPO rollout with one shared prompt prefill (B=1) + batched decode (B=group).

    All ``group`` trajectories share the same prompt/image/region, so the prompt is
    prefilled once at B=1, the per-depth cache is batch-repeated to ``group``, and a
    manual decode loop (raw categorical / greedy) drives the batch. HF ``generate``
    is intentionally NOT used here: after a cache exactly covers the prompt, it would
    treat the input as a re-prefill (``next_sequence_length=0``) and recompute the
    whole prefix. We instead reuse the model's own ``compute_3d_position_ids`` decode
    path (``position_ids=None`` + ``attention_mask=None``) so M-RoPE positions are
    computed by the model, not reimplemented here.
    """
    tokenizer = getattr(processor, 'tokenizer', processor)
    gcfg = cfg['grpo']
    limit = int(gcfg['max_new_tokens'])
    ends = eos_ids(model, tokenizer)
    ends_set = set(ends)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else ends[0]
    core = unwrap_model(model)
    prompt_len = int(batch['prompt_len'][0])
    prompt_ids = batch['input_ids']
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    device = prompt_ids.device

    was_training = core.training
    core.eval()
    result = []
    try:
        with torch.no_grad(), bind_cached_image_features(core, batch['image_embeds']), _region_bind(model, batch):
            # Phase 1: single prefill at B=1.
            out = core(**dict(model_inputs(batch)), use_cache=True)
            cache = out.past_key_values
            first_logits = out.logits[:, -1:, :].repeat(group, 1, 1)  # [group, 1, vocab]

            # Phase 2: repeat the per-depth cache to the group, then batched decode.
            cache.batch_repeat_interleave(group)
            gen_ids = [[] for _ in range(group)]
            finished = torch.zeros(group, dtype=torch.bool, device=device)
            cur = _sample_tokens(first_logits[:, 0, :], sample)  # [group]

            for _ in range(limit):
                for i in range(group):
                    if finished[i]:
                        continue
                    tok = int(cur[i].item())
                    gen_ids[i].append(tok)
                    if tok in ends_set or '</answer>' in tokenizer.decode(gen_ids[i], skip_special_tokens=True):
                        finished[i] = True
                if bool(finished.all()):
                    break
                out = core(input_ids=cur.unsqueeze(-1), attention_mask=None, position_ids=None,
                           past_key_values=cache, use_cache=True)
                nxt = _sample_tokens(out.logits[:, -1, :], sample)
                cur = torch.where(finished, torch.tensor(pad, device=device, dtype=cur.dtype), nxt)
    finally:
        core.train(was_training)
        force_vision_eval(core)

    for i in range(group):
        new = torch.tensor(gen_ids[i], device=device, dtype=torch.long)
        full = torch.cat([prompt_ids[0], new], dim=-1)
        result.append(trim_completion(full, prompt_len, tokenizer, ends))
    return result


def _stop_reason_from_run(run) -> str:
    if run.ok:
        return 'answer'
    return 'error' if run.error else 'length'


def generate_group_staged(model, processor, batch, cfg, group=1, sample=False):
    """GRPO rollout through the FSM (staged) decoder, batched per stage.

    Each completion carries ``sampled_mask`` so ``optimize_group`` can mask the
    controller-injected ``[stage]`` markers out of the logprob/advantage. Returns
    the same ``Completion`` list contract as ``generate_group``.
    """
    from outcome.staged_decode import run_generate_stages_batch

    gcfg = cfg['grpo']
    max_stage = int(gcfg.get('max_stage_tokens', 96))
    max_answer = int(gcfg.get('max_answer_tokens', 192))
    max_localize = gcfg.get('max_localize_tokens', None)
    if max_localize is not None:
        max_localize = int(max_localize)
    prompt_len = int(batch['prompt_len'][0])
    prompt_ids = batch['input_ids']
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    prompt_ids = prompt_ids[0]  # [seq]
    device = prompt_ids.device
    runs = run_generate_stages_batch(model, processor, batch, group=group,
                                     max_stage_tokens=max_stage,
                                     max_answer_tokens=max_answer,
                                     max_localize_tokens=max_localize,
                                     greedy=not sample)
    result = []
    for run in runs:
        new_ids = torch.tensor(run.new_ids or [], device=device, dtype=torch.long)
        full = torch.cat([prompt_ids, new_ids], dim=-1)
        comp = Completion(full, run.text, _stop_reason_from_run(run))
        comp.sampled_mask = list(run.sampled_mask)

        comp.stage_hits = {
            t.name: bool(t.hit_end)
            for t in run.stages
        }

        comp.stage_lengths = {
            t.name: int(t.n_sampled)
            for t in run.stages
        }

        result.append(comp)
    return result


def optimize_group(model, processor, batch, completions, advantages, optimizer, cfg, skip=False):
    """On-policy group with multi-epoch PPO updates (clip engages after epoch 0).

    old/ref logprobs are computed once against the pre-update policy; each
    ``policy_epoch`` then recomputes ``new_lp`` against the (now-updated) policy
    so the importance ratio leaves 1.0 and the PPO clip becomes active. With
    ``policy_epochs == 1`` the ratio is pinned to 1.0 and the clip never fires,
    silently degrading to unclipped REINFORCE, so it is treated as a hard error
    unless explicitly allowed via ``allow_single_epoch``.

    ``skip=True`` runs the actor forward with a zero loss so every DDP rank still
    performs a ``backward()`` (keeping gradient all-reduce in lockstep) while
    contributing no update. This is used when a rank's group advantage collapsed
    to zero but another rank still has signal.
    """
    gc = cfg['grpo']
    policy_epochs = max(1, int(gc.get('policy_epochs', 1)))
    if policy_epochs == 1 and not bool(gc.get('allow_single_epoch', False)):
        raise ValueError('policy_epochs=1 pins ratio to 1.0 (clip never fires); '
                         'use policy_epochs>=2 or set allow_single_epoch=true')
    tokenizer = getattr(processor, 'tokenizer', processor)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    device = next(model.parameters()).device
    prompt_len = int(batch['prompt_len'][0])
    seqs = [c.ids for c in completions]
    masks = [c.sampled_mask for c in completions]
    outputs, attn, labels = padded_completion_tensors(
        seqs, prompt_len, pad, device,
        sampled_masks=masks if any(masks) else None)
    gen_in = model_inputs(batch)
    cache = batch['image_embeds']
    group = len(completions)
    t_start = time.perf_counter()
    if skip:
        old_lp = ref_lp = None
    else:
        old_parts, ref_parts = [], []
        for start, end in micro_batch_ranges(group, int(gc.get('logprob_micro_batch_size', 1))):
            chunk = expand_gen_in_for_group(gen_in, end-start)
            with bind_cached_image_features(model, cache), _region_bind(model, batch):
                old, _ = token_logprobs_nograd(model, chunk, outputs[start:end], attn[start:end], labels[start:end])
                if float(gc['kl_beta']) > 0:
                    with disable_adapter_ctx(model):
                        ref, _ = token_logprobs_nograd(model, chunk, outputs[start:end], attn[start:end], labels[start:end])
                else:
                    ref = old
            old_parts.append(old)
            ref_parts.append(ref)
        old_lp, ref_lp = torch.cat(old_parts), torch.cat(ref_parts)
    t_oldref = time.perf_counter()
    totals = dict(loss=0., pg=0., kl=0., ratio=0., clip_fraction=0., logprob_max_error=0.)
    model.train()
    force_vision_eval(model)
    last_grad_norm = 0.0
    err_tol = float(gc.get('logprob_error_tolerance', .1))
    for pe in range(policy_epochs):
        t_pe = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        pe_totals = dict(loss=0., pg=0., kl=0., ratio=0., clip_fraction=0., logprob_max_error=0.)
        for start, end in micro_batch_ranges(group, int(gc.get('actor_micro_batch_size', 1))):
            chunk = expand_gen_in_for_group(gen_in, end-start)
            with dropout_eval(model), bind_cached_image_features(model, cache), _region_bind(model, batch):
                out = forward_with_vision(model, chunk, outputs[start:end], attn[start:end])
                if skip:
                    # Keep the forward+backward so DDP gradient all-reduce stays in
                    # lockstep with the other ranks, but contribute a zero update.
                    loss = out.logits.float().sum() * 0.0
                    pg = kl = ratio = clipped = torch.zeros((), device=device)
                else:
                    new_lp, mask = token_logprobs(out.logits, labels[start:end])
                    error = ((new_lp.detach()-old_lp[start:end]).abs()*mask).max().item()
                    pe_totals['logprob_max_error'] = max(pe_totals['logprob_max_error'], error)
                    # On epoch 0 old_lp and new_lp come from the same policy, so any
                    # drift is a bf16/checkpointing/dropout bug. After an update the
                    # ratio is supposed to leave 1.0, so only check finiteness.
                    if not torch.isfinite(new_lp).all() or (pe == 0 and error > err_tol):
                        optimizer.zero_grad(set_to_none=True)
                        raise RuntimeError(f'behavior/new logprob mismatch before update: max={error}')
                    loss, pg, kl, ratio, clipped = clipped_pg_kl(new_lp, mask, old_lp[start:end], ref_lp[start:end],
                        advantages[start:end, None], float(gc['clip_low']), float(gc['clip_high']), float(gc['kl_beta']))
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite policy loss')
                weight = (end-start)/group
                (loss*weight).backward()
                for key, value in zip(('loss','pg','kl','ratio','clip_fraction'), (loss,pg,kl,ratio,clipped)):
                    pe_totals[key] += float(value.detach())*weight
                del out, loss
                if not skip:
                    del new_lp
        params = [p for p in model.parameters() if p.requires_grad]
        last_grad_norm = float(torch.nn.utils.clip_grad_norm_(params, float(gc['max_grad_norm']), error_if_nonfinite=True))
        optimizer.step()
        totals[f'seconds_actor_epoch_{pe}'] = time.perf_counter() - t_pe
        for key in ('loss', 'pg', 'kl', 'ratio', 'clip_fraction', 'logprob_max_error'):
            totals[key] += pe_totals[key]
    for key in ('loss', 'pg', 'kl', 'ratio', 'clip_fraction', 'logprob_max_error'):
        totals[key] /= policy_epochs
    totals['grad_norm'] = last_grad_norm
    totals['effective_tokens'] = sum(len(c.ids)-prompt_len for c in completions)
    totals['seconds_old_ref'] = t_oldref - t_start
    totals['seconds_optimize'] = time.perf_counter() - t_start
    return totals
