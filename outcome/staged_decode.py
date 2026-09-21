"""Staged incremental reasoning decode: one prefill, KV cache, U→C→L→V→answer.

Controller (FSM) owns the stage order. The model only samples tokens *inside* a
stage. Markers are injected as raw token ids (no decode→string→retokenize).
This is one assistant turn inside ``<think>``, not four chat rounds.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import torch

from models.h_box_prior import BOX_TOKEN, bind_h_box, compute_g_proj
from models.h_vpt import CONTROL_TOKEN, FEAT_TOKEN, bind_h_vpt, compute_h_H
from models.qwen35 import unwrap_model
from models.vision_cache import bind_cached_image_features
from rl.grpo import expand_gen_in_for_group, model_inputs

STAGES = ('U', 'C', 'L', 'V')
# Bracket-style markers are used because they are familiar to the model from
# the thinking SFT data and do not collide with Qwen's chat-template tokens.
OPEN = {
    'U': '[understand]\n',
    'C': '[compare]\n',
    'L': '[localize]\n',
    'V': '[confirm]\n',
}
# Stage boundaries are aligned with ``thinking.staged_sft_target``: the SFT
# targets have NO ``[/xxx]`` closing markers — a stage body is followed directly
# by the NEXT ``[xxx]`` header (and confirm by ``</think>``).  Asking the model
# to emit a ``[/understand]``-style close it was never trained on is exactly why
# the base model never closes a stage on its own and gets hard-truncated.
THINK_CLOSE = '</think>'
CLOSE_THINK = '</think>\n\n<answer>\n'
ANSWER_STOP = '</answer>'
STAGE_AFTER = {STAGES[i]: STAGES[i + 1] for i in range(len(STAGES) - 1)}
STAGE_AFTER['V'] = 'ANSWER'
STAGE_AFTER['ANSWER'] = 'DONE'


def next_stage(stage: str) -> str:
    if stage not in STAGE_AFTER:
        raise ValueError(f'no successor for stage {stage!r}')
    return STAGE_AFTER[stage]


def next_marker(stage: str) -> str:
    """The next stage's opening marker (an FSM transition, NOT a stop signal)."""
    nxt = next_stage(stage)
    if nxt == 'ANSWER':
        return THINK_CLOSE
    return OPEN[nxt].strip()


STAGE_STOP = '\n'


def stage_stop(stage: str) -> str:
    """Model-owned stop signal.

    U/C/L/V bodies are single-line in the current staged SFT, so they terminate
    with a newline. ANSWER still terminates with ``</answer>``.
    """
    if stage == 'ANSWER':
        return ANSWER_STOP
    return STAGE_STOP


def stage_finished(decoded: str, stage: str) -> bool:
    return stage_stop(stage) in decoded


def inject_after(stage: str) -> str:
    """Controller-owned transition marker injected when ``stage`` ends.

    The model only samples ``body + '\\n'``; the controller appends the next
    ``[xxx]`` header (or ``</think>\\n\\n<answer>\\n`` after V). This matches
    ``thinking.staged_sft_target``, where markers are masked in SFT.
    """
    nxt = next_stage(stage)

    if nxt == 'DONE':
        return ''

    if nxt == 'ANSWER':
        return CLOSE_THINK

    return OPEN[nxt]


@dataclass
class StageTrace:
    name: str
    n_sampled: int
    n_forced: int
    hit_end: bool
    text: str
    seconds: float


@dataclass
class StagedRun:
    ok: bool
    mode: str
    prompt_len: int
    total_new: int
    seconds: float
    stages: list
    text: str
    error: str = ''
    notes: dict = field(default_factory=dict)
    sampled_mask: list = field(default_factory=list)
    new_ids: list = field(default_factory=list)


def encode_ids(tokenizer, text: str, device) -> torch.Tensor:
    ids = tokenizer(text, add_special_tokens=False, return_tensors='pt').input_ids
    return ids.to(device)


_VERIFY_RE = re.compile(r'^\s*(keep|refine|reject|discover|none)\b', re.I)


def _confirm_action(text: str) -> str:
    """Extract the ``[confirm]`` verdict keyword ('' when none is present).

    Mirrors ``outcome.protocol_multibox.parse_verify`` but kept local to avoid a
    circular import (protocol_multibox -> policy -> staged_decode).
    """
    m = _VERIFY_RE.match((text or '').strip())
    return m.group(1).lower() if m else ''


def _strip_vpt_tokens(text: str) -> str:
    """Drop VPT controller-injected placeholder tokens from decoded text.

    ``<|h_ctrl|>`` / ``<|h_feat|>`` / ``<|h_box|>`` are injected by the FSM
    controller (never sampled by the model) to host the cross-attention /
    geometry-token slots. They must not leak into the visible completion / saved
    records, but their ids stay in ``new_ids`` so RL logprob masking stays aligned.
    """
    for tok in (CONTROL_TOKEN, FEAT_TOKEN, BOX_TOKEN):
        text = text.replace(tok, '')
    return text


def _eos_ids(model, tokenizer):
    """All EOS ids (generation_config eos_token_id + tokenizer eos_token_id).

    Qwen's generation config carries several end tokens (``<|im_end|>`` etc.);
    the staged decoder must suppress them in U/C/L/V so the model only stops on
    the FSM newline, not on an early EOS.
    """
    core = unwrap_model(model)

    value = getattr(
        getattr(core, 'generation_config', None),
        'eos_token_id',
        None,
    )

    ids = set()

    if isinstance(value, (list, tuple)):
        ids.update(int(x) for x in value if x is not None)
    elif value is not None:
        ids.add(int(value))

    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))

    return sorted(ids)


def _vision_ctx(model, batch):
    from contextlib import nullcontext, ExitStack
    stack = ExitStack()
    stack.enter_context(bind_cached_image_features(unwrap_model(model), batch['image_embeds']))
    # VPT H injection: the two-pass probe (control-token hidden -> h_H) is done by
    # the caller; here we only scatter h_H into the <|h_feat|> slots.
    from models.h_vpt import bind_h_vpt
    h_H = batch.get('_h_H')
    if h_H is not None:
        stack.enter_context(bind_h_vpt(model, h_H, int(batch['feat_token_id'])))
    else:
        stack.enter_context(nullcontext())
    return stack


@torch.no_grad()
def run_generate_stages(model, processor, batch, *, max_stage_tokens=96, max_answer_tokens=192,
                        max_localize_tokens=None, greedy=True, mode='staged') -> StagedRun:
    """FSM U→C→L→V→answer via a single prefill + KV-cache continuation.

    Stage 1 uses ``core.generate`` to do the vision prefill.  Each subsequent
    stage calls ``core.generate`` again with ``past_key_values`` and the full
    growing prefix; ``pixel_values`` is cleared after the first stage so vision
    is not recomputed.  This is faster than re-prefilling every stage while
    remaining stable on Qwen3.5's M-RoPE / mixed-attention stack.
    """
    from transformers import GenerationConfig, StoppingCriteriaList, StopStringCriteria

    tokenizer = getattr(processor, 'tokenizer', processor)
    device = batch['input_ids'].device
    gen_in = dict(model_inputs(batch))
    prompt_ids = batch['input_ids']
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    prompt_len = int(prompt_ids.shape[-1])
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    ends = _eos_ids(model, tokenizer)

    mm_type = gen_in.pop('mm_token_type_ids', None)
    if mm_type is None:
        mm_type = torch.zeros_like(prompt_ids)
    if mm_type.ndim == 1:
        mm_type = mm_type.unsqueeze(0)

    def align_mm(prefix_ids):
        cur, need = int(mm_type.shape[-1]), int(prefix_ids.shape[-1])
        if cur < need:
            pad = torch.zeros(mm_type.shape[0], need - cur, dtype=mm_type.dtype, device=mm_type.device)
            return torch.cat([mm_type, pad], dim=-1)
        if cur > need:
            return mm_type[:, :need]
        return mm_type

    t0 = time.perf_counter()
    prefix = torch.cat([prompt_ids, encode_ids(tokenizer, OPEN['U'], device)], dim=-1)
    mask = [0] * int(prefix.shape[-1] - prompt_len)  # OPEN['U'] is controller-injected
    traces = []
    core = unwrap_model(model)
    was_training = core.training
    core.eval()
    cache = None
    try:
        with _vision_ctx(model, batch):
            for stage in STAGES + ('ANSWER',):
                st0 = time.perf_counter()
                is_answer = stage == 'ANSWER'
                stop = stage_stop(stage)
                if is_answer:
                    budget = max_answer_tokens
                elif stage == 'L' and max_localize_tokens is not None:
                    budget = max_localize_tokens
                else:
                    budget = max_stage_tokens

                generation = GenerationConfig(
                    max_new_tokens=budget, do_sample=not greedy, temperature=1.0,
                    top_p=1.0, top_k=0,
                    # Only ANSWER may terminate by EOS.
                    eos_token_id=ends if is_answer else None,
                    # U/C/L/V are FSM-controlled and may not emit EOS.
                    suppress_tokens=None if is_answer else ends,
                    pad_token_id=pad,
                    use_cache=True, num_beams=1,
                    return_dict_in_generate=(cache is None))
                stops = StoppingCriteriaList(
                    [StopStringCriteria(tokenizer=tokenizer, stop_strings=[stop])])

                kwargs = dict(gen_in)
                kwargs['input_ids'] = prefix
                kwargs['attention_mask'] = torch.ones_like(prefix)
                kwargs['mm_token_type_ids'] = align_mm(prefix)
                if cache is not None:
                    kwargs['past_key_values'] = cache
                    kwargs['pixel_values'] = None

                old_len = int(prefix.shape[-1])
                out = core.generate(
                    generation_config=generation, stopping_criteria=stops, **kwargs)
                if hasattr(out, 'sequences'):
                    seqs = out.sequences
                    cache = out.past_key_values
                else:
                    seqs = out
                new = seqs[0, old_len:]
                mask.extend([1] * int(new.numel()))  # model-sampled tokens
                body = tokenizer.decode(new.tolist(), skip_special_tokens=False)
                hit_end = stage_finished(body, stage)
                prefix = seqs

                forced = 0
                if stage != 'ANSWER':
                    extra = inject_after(stage)
                    inj = encode_ids(tokenizer, extra, device)
                    forced = int(inj.shape[-1])
                    mask.extend([0] * forced)  # controller-injected tokens
                    prefix = torch.cat([prefix, inj], dim=-1)

                traces.append(StageTrace(
                    stage, int(new.numel()), forced, hit_end, body, time.perf_counter() - st0))
    except Exception as exc:
        core.train(was_training)
        return StagedRun(False, mode, prompt_len, int(prefix.shape[-1]) - prompt_len,
                         time.perf_counter() - t0, traces,
                         _strip_vpt_tokens(tokenizer.decode(prefix[0, prompt_len:].tolist(), skip_special_tokens=False)),
                         error=f'{type(exc).__name__}: {exc}', sampled_mask=mask)
    core.train(was_training)
    new_ids = prefix[0, prompt_len:].tolist()
    expected = list(STAGES) + ['ANSWER']
    names = [t.name for t in traces]
    ok = (
        names == expected
        and len(traces) == len(expected)
        and all(t.hit_end for t in traces)
    )
    return StagedRun(ok, mode, prompt_len, len(new_ids), time.perf_counter() - t0,
                     traces, _strip_vpt_tokens(tokenizer.decode(new_ids, skip_special_tokens=False)),
                     notes={'kv_cache': True, 'stages': names}, sampled_mask=mask,
                     new_ids=new_ids)


def run_cached(model, processor, batch, **kw) -> StagedRun:
    return run_generate_stages(model, processor, batch, mode='cache', **kw)


def run_naive(model, processor, batch, **kw) -> StagedRun:
    return run_generate_stages(model, processor, batch, mode='naive', **kw)


def _trim_new_at_stop(
    tokenizer,
    new_ids: torch.Tensor,
    stop: str,
    device,
    end_ids=(),
):
    """Trim at the first expected stop or real EOS.

    Batch ``generate`` runs until *every* sequence hits the stop string, so
    trajectories that finish early carry garbage tokens after their stop; trim
    them so each trajectory keeps only its own stage content.

    Returns ``(kept, hit_stop, hit_eos)``.
    """
    ids = new_ids.tolist()
    kept = []

    end_ids = set(int(x) for x in end_ids)

    for t in ids:
        kept.append(t)

        text = tokenizer.decode(
            kept,
            skip_special_tokens=False,
        )

        # Expected FSM stop has priority.
        if stop in text:
            return (
                torch.tensor(
                    kept,
                    dtype=torch.long,
                    device=device,
                ),
                True,
                False,
            )

        if int(t) in end_ids:
            return (
                torch.tensor(
                    kept,
                    dtype=torch.long,
                    device=device,
                ),
                False,
                True,
            )

    return (
        torch.tensor(
            kept,
            dtype=torch.long,
            device=device,
        ),
        False,
        False,
    )


@torch.no_grad()
def run_generate_stages_batch(model, processor, batch, *, group=1,
                              max_stage_tokens=96, max_answer_tokens=192,
                              max_localize_tokens=None, greedy=False,
                              confirm_reloop=True, max_reloop=3,
                              reject_logit_bias=0.0) -> list:
    """Batched FSM rollout: all ``group`` trajectories advance stage-by-stage.

    ``group`` trajectories share one prompt/image, so each stage packs them into
    a single ``batch=group`` forward. Per-stage re-prefill is used instead of
    cross-stage KV-cache continuation: the benchmark showed re-prefill is ~10%
    *faster* than cache continuation (vision is already cached), and it avoids
    the variable-length-cache stacking problem entirely. Returns one ``StagedRun``
    per trajectory, each carrying ``sampled_mask`` / ``new_ids`` for RL.
    """
    from transformers import GenerationConfig, LogitsProcessor, StoppingCriteriaList, StopStringCriteria

    class _RejectBias(LogitsProcessor):
        """Add a (decaying) logit bias to the ``reject`` verdict token on the very
        first [confirm] token only, so the policy can explore a reject despite SFT
        teaching only keep/none. After the first token the bias is removed so the
        rest of the confirm evidence is sampled under the model's own distribution.
        """

        def __init__(self, reject_ids, bias, start_len):
            self.reject_ids = list(reject_ids)
            self.bias = float(bias)
            self.start_len = int(start_len)

        def __call__(self, input_ids, scores):
            if input_ids.shape[-1] == self.start_len:
                for tid in self.reject_ids:
                    scores[:, tid] = scores[:, tid] + self.bias
            return scores

    tokenizer = getattr(processor, 'tokenizer', processor)
    device = batch['input_ids'].device
    gen_in = dict(model_inputs(batch))
    prompt_ids = batch['input_ids']
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    prompt_len = int(prompt_ids.shape[-1])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    # Token ids that can start the "reject" verdict (with/without a leading space).
    reject_ids = []
    if reject_logit_bias > 0:
        reject_ids = sorted({int(encode_ids(tokenizer, form, device)[0][0].item())
                             for form in ('reject', ' reject')})
    ends = _eos_ids(model, tokenizer)

    mm_prompt = gen_in.pop('mm_token_type_ids', None)
    if mm_prompt is None:
        mm_prompt = torch.zeros_like(prompt_ids)
    if mm_prompt.ndim == 1:
        mm_prompt = mm_prompt.unsqueeze(0)
    mm_prompt = mm_prompt[0]  # [L_prompt]

    # Prompt-side multimodal metadata to group (image_grid_thw -> [group, 3]) so
    # the cached vision features expand to cover every trajectory.
    gen_in_group = expand_gen_in_for_group(gen_in, group)

    open_u = encode_ids(tokenizer, OPEN['U'], device)[0]  # [n_open]
    prefixes = [torch.cat([prompt_ids[0], open_u], dim=-1) for _ in range(group)]
    masks = [[0] * int(open_u.shape[-1]) for _ in range(group)]
    traces_per = [[] for _ in range(group)]

    core = unwrap_model(model)
    h_vpt = getattr(core, 'h_vpt', None)
    h_map = batch.get('h_map')
    ctrl_id = int(batch.get('control_token_id', -1))
    feat_id = int(batch.get('feat_token_id', -1))
    n_feat = int(batch.get('n_feat_tokens', 0))
    vpt_on = h_vpt is not None and h_map is not None and n_feat > 0 and ctrl_id >= 0 and feat_id >= 0
    h_box_prior = getattr(core, 'h_box_prior', None)
    box_id = int(batch.get('box_token_id', -1))
    n_box = int(batch.get('n_box_tokens', 0))
    h_box_geom = batch.get('h_box_geom')
    box_on = h_box_prior is not None and h_box_geom is not None and n_box > 0 and box_id >= 0
    was_training = core.training
    core.eval()
    h_H_val = None  # [N, hidden] — computed once at the L stage, reused for V/ANSWER
    g_proj = None   # [K, hidden] — H-Box geometry tokens, computed once at L

    def _pack_prefixes():
        """Right-align the current per-trajectory prefixes into a padded batch."""
        mx = max(int(p.shape[-1]) for p in prefixes)
        ids = torch.full((group, mx), pad_id, device=device, dtype=torch.long)
        am = torch.zeros((group, mx), device=device, dtype=torch.long)
        mm = torch.zeros((group, mx), dtype=mm_prompt.dtype, device=device)
        mlen = int(mm_prompt.shape[-1])
        for i, p in enumerate(prefixes):
            n = int(p.shape[-1])
            off = mx - n
            ids[i, off:] = p
            am[i, off:] = 1
            # mm_token_type_ids: prompt multimodal types right-aligned, generated
            # tail is text (0). Left padding keeps image tokens' relative
            # positions intact so M-RoPE computes correct positions.
            mm[i, off:off + mlen] = mm_prompt
        return mx, ids, am, mm

    try:
        with _vision_ctx(model, batch):
            linear = list(STAGES) + ['ANSWER']
            L_IDX = linear.index('L')
            stage_idx = 0
            reloop = 0
            while stage_idx < len(linear):
                stage = linear[stage_idx]
                is_answer = stage == 'ANSWER'
                stop = stage_stop(stage)
                if is_answer:
                    budget = max_answer_tokens
                elif stage == 'L' and max_localize_tokens is not None:
                    budget = max_localize_tokens
                else:
                    budget = max_stage_tokens

                # VPT two-pass: at the L stage append <|h_ctrl|>, probe its hidden,
                # project H -> h_H, then append N <|h_feat|> slots. h_H is kept for
                # every later stage because re-prefill re-embeds the feat tokens.
                if stage == 'L':
                    # On the first L pass, inject the H priors (box + control +
                    # feat). On a confirm-reject reloop (reloop > 0) H is distrusted,
                    # so the second L pass is a PURE text+image localization with no
                    # H tokens at all — this keeps a single set of H slots in the
                    # completion so ``_h_H``/``_g_proj`` scatter stays aligned in the
                    # RL logprob pass (two different H sets would need per-slot
                    # scatter, which ``bind_h_vpt`` does not support).
                    if box_on and reloop == 0:
                        boxes = torch.full((n_box,), box_id, device=device, dtype=torch.long)
                        for i in range(group):
                            prefixes[i] = torch.cat([prefixes[i], boxes], dim=-1)
                            masks[i].extend([0] * n_box)
                        g_proj = compute_g_proj(h_box_prior, h_box_geom)
                        batch['_g_proj'] = g_proj.detach()
                    # VPT control token (appended AFTER the box tokens so the -1
                    # probe position below is still the control token).
                    if vpt_on and reloop == 0:
                        ctrl = torch.tensor([ctrl_id], device=device, dtype=torch.long)
                        for i in range(group):
                            prefixes[i] = torch.cat([prefixes[i], ctrl], dim=-1)
                            masks[i].extend([0])

                max_len, input_ids, attn, mm_batch = _pack_prefixes()

                generation = GenerationConfig(
                    max_new_tokens=budget, do_sample=not greedy, temperature=1.0,
                    top_p=1.0, top_k=0,
                    # Only ANSWER may terminate by EOS.
                    eos_token_id=ends if is_answer else None,
                    # U/C/L/V are FSM-controlled and may not emit EOS.
                    suppress_tokens=None if is_answer else ends,
                    pad_token_id=pad_id,
                    use_cache=True, num_beams=1)
                stops = StoppingCriteriaList(
                    [StopStringCriteria(tokenizer=tokenizer, stop_strings=[stop])])

                kwargs = dict(gen_in_group)
                kwargs['input_ids'] = input_ids
                kwargs['attention_mask'] = attn
                kwargs['mm_token_type_ids'] = mm_batch
                # Re-prefill re-embeds the FULL sequence (image tokens included)
                # every stage, so pixel_values + image_grid_thw must be present on
                # EVERY stage. The _vision_ctx bind swaps get_image_features for a
                # frozen cache keyed off image_grid_thw, so this stays cheap; dropping
                # pixel_values here is what previously made stages 2+ see garbage
                # image embeddings (the batched rollout degraded FPR -> 1.0).
                kwargs['pixel_values'] = gen_in_group.get('pixel_values')

                if stage == 'L' and vpt_on and reloop == 0:
                    # Probe: control token is the last token of every right-aligned
                    # row; all rows share one prompt, so take row 0's hidden. The
                    # box tokens must be scattered during the probe so h_ctrl is the
                    # model's hidden state *after* seeing the geometry prior.
                    with torch.no_grad():
                        with bind_h_box(model, g_proj, box_id):
                            probe = core(**kwargs, output_hidden_states=True)
                        h_ctrl = probe.hidden_states[-1][0:1, -1, :].detach()  # [1, hidden]
                    h_H_val = compute_h_H(h_vpt, h_map, h_ctrl)  # [N, hidden]
                    batch['_h_H'] = h_H_val  # reuse in optimize_group (scatter tiles)
                    # Keep the control-token hidden + the H map so the RL actor can
                    # recompute h_H with a *differentiable* projector (when
                    # h_vpt.trainable=gate_projector) instead of reusing this detached
                    # rollout value. Detached, since rollout sampling must not carry grads.
                    batch['_h_ctrl'] = h_ctrl.detach()
                    # Append the N feat slots (controller-injected) after the control token.
                    feats = torch.full((n_feat,), feat_id, device=device, dtype=torch.long)
                    for i in range(group):
                        prefixes[i] = torch.cat([prefixes[i], feats], dim=-1)
                        masks[i].extend([0] * n_feat)
                    max_len, input_ids, attn, mm_batch = _pack_prefixes()
                    kwargs['input_ids'] = input_ids
                    kwargs['attention_mask'] = attn
                    kwargs['mm_token_type_ids'] = mm_batch

                # Confirm-stage reject exploration: bias the "reject" verdict token on
                # the first token only (decaying via the caller-computed bias) so the
                # policy can explore a reject despite SFT teaching only keep/none.
                logits_processor = None
                if stage == 'V' and reject_ids and reject_logit_bias > 0:
                    logits_processor = [_RejectBias(reject_ids, reject_logit_bias,
                                                     int(input_ids.shape[-1]))]

                with bind_h_box(model, g_proj, box_id), bind_h_vpt(model, h_H_val, feat_id):
                    out = core.generate(
                        generation_config=generation, stopping_criteria=stops,
                        logits_processor=logits_processor, **kwargs)
                seqs = out.sequences if hasattr(out, 'sequences') else out
                new_all = seqs[:, max_len:]

                bodies = []
                kepts = []
                hit_ends = []
                for i in range(group):
                    kept, hit_end, hit_eos = _trim_new_at_stop(
                        tokenizer,
                        new_all[i],
                        stop,
                        device,
                        end_ids=ends if is_answer else (),
                    )
                    kepts.append(kept)
                    hit_ends.append(hit_end)
                    bodies.append(tokenizer.decode(kept.tolist(), skip_special_tokens=False))

                # Confirm-reject reloop: if any trajectory's [confirm] rejects the
                # H-guided localization, distrust H and CONTINUE into a second
                # [localize]->[confirm] pass with zeroed H (rather than rewinding).
                # The reject tokens stay in the completion so their logprob is kept:
                # GRPO's sequence-level advantage then learns *when* to reject — a
                # reject that improves the final IoU raises the whole trajectory's
                # reward and back-propagates to the reject token.
                do_reloop = False
                if stage == 'V' and vpt_on and confirm_reloop:
                    rejected = any(_confirm_action(b) == 'reject' for b in bodies)
                    do_reloop = bool(rejected and reloop < int(max_reloop))

                for i in range(group):
                    kept = kepts[i]
                    hit_end = hit_ends[i]
                    n_kept = int(kept.shape[-1])
                    masks[i].extend([1] * n_kept)
                    prefixes[i] = torch.cat([prefixes[i], kept], dim=-1)
                    forced = 0
                    if stage != 'ANSWER':
                        # On a reloop, re-open [localize] (zero-H) instead of the
                        # normal </think><answer> transition.
                        extra = OPEN['L'] if do_reloop else inject_after(stage)
                        inj = encode_ids(tokenizer, extra, device)[0]
                        forced = int(inj.shape[-1])
                        masks[i].extend([0] * forced)
                        prefixes[i] = torch.cat([prefixes[i], inj], dim=-1)
                    traces_per[i].append(StageTrace(stage, n_kept, forced, hit_end, bodies[i], 0.0))

                if do_reloop:
                    reloop += 1
                    stage_idx = L_IDX
                    continue
                stage_idx += 1
    finally:
        core.train(was_training)

    runs = []
    expected = list(STAGES) + ['ANSWER']
    for i in range(group):
        new_ids = prefixes[i][prompt_len:].tolist()
        names = [t.name for t in traces_per[i]]
        # Multi-round: a confirm-reject appends extra L->V pairs before ANSWER, so
        # the stage list is U,C,L,V,(L,V)*,ANSWER. Validate the mandatory prefix +
        # suffix and that the extras come in L,V pairs.
        base_ok = names[:len(STAGES)] == list(STAGES) and names[-1] == 'ANSWER'
        extras = names[len(STAGES):-1]
        rounds_ok = (len(extras) % 2 == 0 and all(n in ('L', 'V') for n in extras)
                     and extras[0::2] == ['L'] * (len(extras) // 2)
                     and extras[1::2] == ['V'] * (len(extras) // 2))
        ok = (
            base_ok
            and rounds_ok
            and len(traces_per[i]) == len(names)
            and all(t.hit_end for t in traces_per[i])
        )
        runs.append(StagedRun(
            ok, 'cache', prompt_len, len(new_ids), 0.0, traces_per[i],
            _strip_vpt_tokens(tokenizer.decode(new_ids, skip_special_tokens=False)),
            notes={'kv_cache': False, 'stages': names}, sampled_mask=masks[i],
            new_ids=new_ids))
    return runs


@torch.no_grad()
def run_oneshot(model, processor, batch, *, max_new_tokens=768, greedy=True) -> StagedRun:
    from outcome.policy import generate_group
    cfg = {'grpo': {
        'max_new_tokens': int(max_new_tokens),
        'rollout_micro_batch_size': 1,
        'temperature': 1.0, 'top_p': 1.0, 'top_k': 0,
    }}
    t0 = time.perf_counter()
    cs = generate_group(model, processor, batch, cfg, group=1, sample=not greedy)
    c = cs[0]
    elapsed = time.perf_counter() - t0
    n = int(c.ids.numel()) - int(batch['prompt_len'][0])
    return StagedRun(True, 'oneshot', int(batch['prompt_len'][0]), n, elapsed,
                     [StageTrace('oneshot', n, 0, c.stop_reason != 'length', c.text, elapsed)],
                     c.text, notes={'stop': c.stop_reason})
