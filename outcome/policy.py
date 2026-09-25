"""Finite rollouts, genuine completion lengths and one outcome advantage."""
from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
from transformers import GenerationConfig, StoppingCriteriaList, StopStringCriteria

from models.qwen35 import force_vision_eval, unwrap_model
from models.vision_cache import bind_cached_image_features
from outcome.thinking import loc_token_mask, verify_token_mask
from rl.grpo import (clipped_pg_kl, disable_adapter_ctx, dropout_eval, expand_gen_in_for_group,
                     forward_with_vision, micro_batch_ranges, model_inputs, move_batch,
                     padded_completion_tensors, token_logprobs, token_logprobs_nograd)


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


def _h_box_ctx(model, batch):
    """Context manager scattering ``g_proj`` into ``<|h_box|>`` slots (or nullcontext).

    The frozen HBoxProjector maps ``batch['h_box_geom']`` ([K, GEOM_DIM]) to ``g_proj``
    ([K, hidden]); the projector is deterministic and detached, so recomputing it at
    every forward (rollout / old-ref / actor) yields identical embeddings.
    """
    core = unwrap_model(model)
    proj = getattr(core, 'h_box_prior', None)
    h_box_geom = batch.get('h_box_geom')
    box_id = int(batch.get('box_token_id', -1))
    if proj is None or h_box_geom is None or box_id < 0:
        return nullcontext()
    from models.h_box_prior import bind_h_box, compute_g_proj
    return bind_h_box(model, compute_g_proj(proj, h_box_geom), box_id)


def _h_vpt_h_H(model, batch):
    """Probe the control token's hidden state and compute the modulated H features.

    Returns h_H [N, hidden] (single-sample) or None when H-VPT is disabled. The
    control token lives in the prompt (single-shot world-model formulation), so a
    truncated forward up to its position reads the LLM's post-vision intent state,
    which then retrieves/modulates the H patch map via cross-attention.
    """
    core = unwrap_model(model)
    h_vpt = getattr(core, 'h_vpt', None)
    h_map = batch.get('h_map')
    ctrl_id = int(batch.get('control_token_id', -1))
    if h_vpt is None or h_map is None or ctrl_id < 0:
        return None
    from models.h_vpt import compute_h_H
    input_ids = batch['input_ids']
    attention_mask = batch.get('attention_mask')
    pos = (input_ids == ctrl_id).long().argmax(dim=-1)  # [B], one per row
    max_pos = int(pos.max().item())
    ids_trunc = input_ids[:, :max_pos + 1].contiguous()
    attn_trunc = attention_mask[:, :max_pos + 1].contiguous() if attention_mask is not None else None
    gen_in = model_inputs(batch)
    with torch.no_grad():
        out = forward_with_vision(core, gen_in, ids_trunc, attn_trunc, output_hidden_states=True)
    h = out.hidden_states[-1]  # [B, L, hidden]
    n = int(ids_trunc.shape[0])
    h_ctrl = h[torch.arange(n, device=h.device), pos].detach()  # [B, hidden]
    batch['_h_ctrl'] = h_ctrl
    return compute_h_H(h_vpt, h_map.to(h.device), h_ctrl)  # [N, hidden]


def _h_vpt_ctx(model, batch, h_H, with_grad=False):
    """Context manager scattering ``h_H`` into ``<|h_feat|>`` slots (or nullcontext).

    When ``with_grad`` and the HVPT projector is trainable, h_H is recomputed from
    the cached ``batch['_h_ctrl']`` with gradients flowing to the projector (actor
    forward only); otherwise the detached rollout value is reused.
    """
    core = unwrap_model(model)
    h_vpt = getattr(core, 'h_vpt', None)
    feat_id = int(batch.get('feat_token_id', -1))
    if h_vpt is None or feat_id < 0:
        return nullcontext()
    from models.h_vpt import bind_h_vpt, compute_h_H
    if with_grad and any(p.requires_grad for p in h_vpt.projector.parameters()):
        h_ctrl = batch.get('_h_ctrl')
        h_map = batch.get('h_map')
        if h_ctrl is not None and h_map is not None:
            h_H = compute_h_H(h_vpt, h_map.to(h_ctrl.device), h_ctrl)
    if h_H is None:
        return nullcontext()
    return bind_h_vpt(model, h_H, feat_id)


def generate_group(model, processor, batch, cfg, group=1, sample=False):
    """Single-shot autoregressive rollout (world-model internal thinking).

    The model writes the entire ``[understand]...[confirm]</think><answer>`` chain
    in one pass; no external FSM injects markers. ``batch['image_embeds']`` is the
    frozen vision cache (ref + test merger tokens).
    """
    tokenizer = getattr(processor, 'tokenizer', processor)
    gcfg = cfg['grpo']
    if bool(gcfg.get('shared_prefill', False)):
        return generate_group_shared_prefill(model, processor, batch, cfg, group=group, sample=sample)
    limit = int(gcfg['max_new_tokens'])
    ends = eos_ids(model, tokenizer)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else ends[0]
    # A fresh config removes inherited top-k/penalties/forced tokens. Raw-policy
    # categorical sampling is exactly the distribution used by all logprobs.
    generation = GenerationConfig(max_new_tokens=limit, do_sample=sample,
        temperature=1.0, top_p=1.0, top_k=0, typical_p=1.0, repetition_penalty=1.0,
        eos_token_id=ends or None, pad_token_id=pad, use_cache=True, num_beams=1)
    stops = StoppingCriteriaList([StopStringCriteria(tokenizer=tokenizer, stop_strings=['</answer>'])])
    core = unwrap_model(model)
    was_training = core.training
    core.eval()
    result = []
    try:
        with torch.no_grad(), bind_cached_image_features(core, batch['image_embeds']):
            h_H = _h_vpt_h_H(core, batch)
            with _h_box_ctx(model, batch), _h_vpt_ctx(model, batch, h_H):
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


def _trim_to_first_stop(text: str, stop_strings) -> str:
    """Cut ``text`` at the earliest occurrence of any stop string (exclusive)."""
    pos = len(text)
    for s in stop_strings:
        idx = text.find(s)
        if idx != -1:
            pos = min(pos, idx)
    return text[:pos]


def _generate_stage1(model, processor, batch, cfg, sample):
    """Latent pass: generate up to (but not including) ``[imagine]`` / ``[confirm]``.

    Mirrors ``generate_group`` but stops at the first stage marker so the candidate
    boxes in ``[localize]`` can be extracted for the zoom observation pass.
    """
    tokenizer = getattr(processor, 'tokenizer', processor)
    gcfg = cfg['grpo']
    limit = int(gcfg['max_new_tokens'])
    ends = eos_ids(model, tokenizer)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else ends[0]
    generation = GenerationConfig(max_new_tokens=limit, do_sample=sample,
        temperature=1.0, top_p=1.0, top_k=0, typical_p=1.0, repetition_penalty=1.0,
        eos_token_id=ends or None, pad_token_id=pad, use_cache=True, num_beams=1)
    stop_strings = ['[imagine]', '[confirm]', '</think>', '<answer>']
    stops = StoppingCriteriaList([StopStringCriteria(tokenizer=tokenizer, stop_strings=stop_strings)])
    core = unwrap_model(model)
    was_training = core.training
    core.eval()
    try:
        with torch.no_grad(), bind_cached_image_features(core, batch['image_embeds']):
            h_H = _h_vpt_h_H(core, batch)
            with _h_box_ctx(model, batch), _h_vpt_ctx(model, batch, h_H):
                generated = core.generate(**model_inputs(batch), generation_config=generation,
                                          stopping_criteria=stops)
    finally:
        core.train(was_training)
        force_vision_eval(core)
    row = generated[0]
    prompt_len = int(batch['prompt_len'][0])
    ids = row.detach().clone()
    raw = tokenizer.decode(ids[prompt_len:], skip_special_tokens=True)
    text = _trim_to_first_stop(raw, stop_strings)
    return Completion(ids, text, 'stage1')


def _zoom_continuation_text(cfg, class_name: str) -> str:
    """Second-pass prompt: re-observe the candidate box's extent from the crop."""
    return (
        f"Image 1 is a defect-free reference of {class_name}. Image 2 is the inspection "
        "image. Image 3 is a zoomed crop centered on your first candidate box; the red "
        "rectangle is that candidate box outline. You have already written your "
        "understand / compare / localize reasoning above. Now look closely at the crop: "
        "does the candidate box's extent match the true defect boundary, or is it too "
        "small / too large / shifted? Continue your thinking from [imagine] (judge the "
        "candidate box quality: keep / refine / reject / none), then [confirm] (whether "
        "your refinement improved / was unchanged / degraded the box). Close your "
        "thinking with </think> and output <answer>."
    )


def generate_group_zoom(model, processor, prior, batch, cfg, group=1, sample=False):
    """Two-stage world-model rollout with a zoom-crop observation step.

    Stage 1 writes ``[understand][compare][localize]`` in latent space. The first
    candidate box is cropped (padded window + drawn outline) and re-encoded; stage 2
    prefills a 3-image prompt (ref + test + crop) so the model *sees* the box extent
    before committing in ``[imagine][confirm]``. The two texts are concatenated so the
    downstream parser sees one uninterrupted think chain.

    This path is used for evaluation / prediction (``group==1``). For group sampling
    the per-trajectory crop makes a batched prefill non-trivial, so it falls back to
    the single-pass rollout (which keeps the training loop correct).
    """
    zcfg = (cfg.get('outcome') or {}).get('zoom') or {}
    if not bool(zcfg.get('enabled', False)) or group != 1:
        return generate_group(model, processor, batch, cfg, group=group, sample=sample)

    stage1 = _generate_stage1(model, processor, batch, cfg, sample)
    meta = (batch.get('_meta') or [{}])[0]
    test_img = meta.get('test')
    ref_img = meta.get('ref')
    class_name = str(meta.get('class_name', 'object'))
    orig_size = meta.get('orig_size') or (test_img.size if test_img is not None else None)

    if test_img is None or orig_size is None:
        return [stage1]

    from outcome.protocol_multibox import parse_boxes_list
    from outcome.zoom_crop import make_zoom_crop
    from outcome.inputs import build_zoom_batch

    state, boxes = parse_boxes_list(stage1.text)
    zoom = make_zoom_crop(
        test_img, boxes[0] if (state == 'list' and boxes) else None,
        orig_size=orig_size,
        expand=float(zcfg.get('expand', 1.0)),
        min_pad_frac=float(zcfg.get('min_pad_frac', 0.12)),
        max_area_frac=float(zcfg.get('max_area_frac', 0.6)),
    )
    if zoom.degenerate:
        # No meaningful crop (no box, or a whole-image window): complete single-pass.
        return generate_group(model, processor, batch, cfg, group=1, sample=sample)

    device = batch['input_ids'].device
    zbatch = build_zoom_batch(processor, prior, cfg, ref_img, test_img, zoom.image,
                              _zoom_continuation_text(cfg, class_name), device,
                              crop_min_pixels=zcfg.get('crop_min_pixels'),
                              prefill_text=stage1.text)
    zbatch = move_batch(zbatch, device)
    stage2 = generate_group(model, processor, zbatch, cfg, group=1, sample=sample)[0]

    # stage2 continues from the prefilled [understand][compare][localize] prefix,
    # so its text starts at [imagine]. Concatenate the two spans into one think
    # chain for the parser; ids are synthesized for `new_tokens` accounting only.
    p1 = int(batch['prompt_len'][0])
    p2 = int(zbatch['prompt_len'][0])
    ids = torch.cat([stage1.ids[:p1], stage1.ids[p1:], stage2.ids[p2:]])
    return [Completion(ids, stage1.text + stage2.text, stage2.stop_reason)]


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
        with torch.no_grad(), bind_cached_image_features(core, batch['image_embeds']):
            h_H = _h_vpt_h_H(core, batch)
            with _h_box_ctx(model, batch), _h_vpt_ctx(model, batch, h_H):
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


def optimize_group(model, processor, batch, completions, advantages, optimizer, cfg, skip=False,
                   loc_advantages=None, verify_advantages=None):
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

    ``loc_advantages`` (optional, ``[group]``) enables the per-token advantage
    split: localization tokens (``[localize]`` candidate boxes + final
    ``bboxes_2d``) get ``loc_advantages`` while all other completion tokens get
    ``advantages``. This keeps coordinate gradients from being averaged away by
    the long prose/confirm text in the single-shot world-model rollout.
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
    max_t = int(outputs.shape[1])
    # Per-token advantage: localization tokens use ``loc_advantages``, everything
    # else uses ``advantages``. ``adv_token`` is aligned to sequence positions; the
    # actor loop slices ``[:, 1:]`` to match the shifted logprob columns.
    n_loc_tokens = 0
    if loc_advantages is not None:
        loc_mask = torch.zeros(group, max_t, device=device, dtype=torch.bool)
        for i, c in enumerate(completions):
            flags = loc_token_mask(tokenizer, c.text)
            n_comp = int(c.ids.numel()) - prompt_len
            for j, flag in enumerate(flags[:n_comp]):
                if flag:
                    loc_mask[i, prompt_len + j] = True
        n_loc_tokens = int(loc_mask.sum().item())
        adv_token = torch.where(
            loc_mask,
            loc_advantages[:, None].expand(group, max_t),
            advantages[:, None].expand(group, max_t),
        )
    else:
        adv_token = advantages[:, None].expand(group, max_t)
    # Rehearsal/verification tokens ([imagine] + [confirm] bodies) get their own
    # calibration advantage so the "is my box right?" reasoning has focused gradient.
    n_verify_tokens = 0
    if verify_advantages is not None:
        verify_mask = torch.zeros(group, max_t, device=device, dtype=torch.bool)
        for i, c in enumerate(completions):
            flags = verify_token_mask(tokenizer, c.text)
            n_comp = int(c.ids.numel()) - prompt_len
            for j, flag in enumerate(flags[:n_comp]):
                if flag:
                    verify_mask[i, prompt_len + j] = True
        n_verify_tokens = int(verify_mask.sum().item())
        adv_token = torch.where(
            verify_mask,
            verify_advantages[:, None].expand(group, max_t),
            adv_token,
        )
    # H-VPT probe (once): the control token is in the shared prompt, so h_H is
    # identical across the whole group; scatter it on every forward (old/ref + actor).
    with torch.no_grad(), bind_cached_image_features(model, cache):
        h_H = _h_vpt_h_H(model, batch)
    t_start = time.perf_counter()
    if skip:
        old_lp = ref_lp = None
    else:
        old_parts, ref_parts = [], []
        for start, end in micro_batch_ranges(group, int(gc.get('logprob_micro_batch_size', 1))):
            chunk = expand_gen_in_for_group(gen_in, end-start)
            with bind_cached_image_features(model, cache), _h_box_ctx(model, batch), _h_vpt_ctx(model, batch, h_H):
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
    totals = dict(loss=0., pg=0., kl=0., ratio=0., clip_fraction=0., logprob_max_error=0.,
                  loc_tokens=n_loc_tokens)
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
            with dropout_eval(model), bind_cached_image_features(model, cache), _h_box_ctx(model, batch), _h_vpt_ctx(model, batch, h_H, with_grad=True):
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
                        adv_token[start:end, 1:], float(gc['clip_low']), float(gc['clip_high']), float(gc['kl_beta']))
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
