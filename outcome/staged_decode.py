"""Staged incremental reasoning decode: one prefill, KV cache, U→C→L→V→answer.

Controller (FSM) owns the stage order. The model only samples tokens *inside* a
stage. Markers are injected as raw token ids (no decode→string→retokenize).
This is one assistant turn inside ``<think>``, not four chat rounds.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from models.qwen35 import unwrap_model
from models.region_injection import bind_region_injection, has_region, region_raw_from_batch
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
    """The token text that ends ``stage``: the next stage's opening marker.

    ``V`` ends at ``</think>``. Mirrors ``thinking.staged_sft_target``, where a
    stage body is followed directly by the next ``[xxx]`` header.
    """
    nxt = next_stage(stage)
    if nxt == 'ANSWER':
        return THINK_CLOSE
    return OPEN[nxt].strip()


def stage_finished(decoded: str, stage: str) -> bool:
    if stage == 'ANSWER':
        return ANSWER_STOP in decoded
    return next_marker(stage) in decoded


def inject_after(stage: str, advanced: bool) -> str:
    """Token text the controller appends when ``stage`` ends.

    ``advanced`` is True when the model already emitted the next stage's marker
    (or ``</think>`` for V); then only the separator / answer opener is inserted.
    Otherwise the controller injects the missing boundary marker itself.
    """
    nxt = next_stage(stage)
    if nxt == 'DONE':
        return ''
    if nxt == 'ANSWER':
        return '\n\n<answer>\n' if advanced else CLOSE_THINK
    return '\n' if advanced else '\n' + next_marker(stage) + '\n'


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


def _vision_ctx(model, batch):
    from contextlib import nullcontext, ExitStack
    stack = ExitStack()
    stack.enter_context(bind_cached_image_features(unwrap_model(model), batch['image_embeds']))
    adapter = getattr(unwrap_model(model), 'region_adapter', None)
    if adapter is not None and has_region(batch):
        stack.enter_context(bind_region_injection(
            model, adapter, region_raw_from_batch(batch), int(batch['region_token_id'])))
    else:
        stack.enter_context(nullcontext())
    return stack


@torch.no_grad()
def run_generate_stages(model, processor, batch, *, max_stage_tokens=96, max_answer_tokens=192,
                        greedy=True, mode='staged') -> StagedRun:
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
    ends = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None

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
                stop = ANSWER_STOP if stage == 'ANSWER' else next_marker(stage)
                budget = max_answer_tokens if stage == 'ANSWER' else max_stage_tokens

                generation = GenerationConfig(
                    max_new_tokens=budget, do_sample=not greedy, temperature=1.0,
                    top_p=1.0, top_k=0, eos_token_id=ends, pad_token_id=pad,
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
                advanced = stage_finished(body, stage)
                prefix = seqs

                forced = 0
                if stage != 'ANSWER':
                    extra = inject_after(stage, advanced=advanced)
                    inj = encode_ids(tokenizer, extra, device)
                    forced = int(inj.shape[-1])
                    mask.extend([0] * forced)  # controller-injected tokens
                    prefix = torch.cat([prefix, inj], dim=-1)

                traces.append(StageTrace(
                    stage, int(new.numel()), forced, advanced, body, time.perf_counter() - st0))
    except Exception as exc:
        core.train(was_training)
        return StagedRun(False, mode, prompt_len, int(prefix.shape[-1]) - prompt_len,
                         time.perf_counter() - t0, traces,
                         tokenizer.decode(prefix[0, prompt_len:].tolist(), skip_special_tokens=False),
                         error=f'{type(exc).__name__}: {exc}', sampled_mask=mask)
    core.train(was_training)
    new_ids = prefix[0, prompt_len:].tolist()
    names = [t.name for t in traces]
    ok = names[-1] == 'ANSWER' and all(s in names for s in STAGES)
    return StagedRun(ok, mode, prompt_len, len(new_ids), time.perf_counter() - t0,
                     traces, tokenizer.decode(new_ids, skip_special_tokens=False),
                     notes={'kv_cache': True, 'stages': names}, sampled_mask=mask,
                     new_ids=new_ids)


def run_cached(model, processor, batch, **kw) -> StagedRun:
    return run_generate_stages(model, processor, batch, mode='cache', **kw)


def run_naive(model, processor, batch, **kw) -> StagedRun:
    return run_generate_stages(model, processor, batch, mode='naive', **kw)


def _trim_new_at_stop(tokenizer, new_ids: torch.Tensor, stop: str, device) -> torch.Tensor:
    """Keep tokens up to and including the first occurrence of ``stop``.

    Batch ``generate`` runs until *every* sequence hits the stop string, so
    trajectories that finish early carry garbage tokens after their stop; trim
    them so each trajectory keeps only its own stage content (matching the
    single-trajectory ``StopStringCriteria`` semantics).
    """
    ids = new_ids.tolist()
    kept = []
    for t in ids:
        kept.append(t)
        if stop in tokenizer.decode(kept, skip_special_tokens=False):
            break
    return torch.tensor(kept, dtype=torch.long, device=device)


@torch.no_grad()
def run_generate_stages_batch(model, processor, batch, *, group=1,
                              max_stage_tokens=96, max_answer_tokens=192,
                              greedy=False) -> list:
    """Batched FSM rollout: all ``group`` trajectories advance stage-by-stage.

    ``group`` trajectories share one prompt/image, so each stage packs them into
    a single ``batch=group`` forward. Per-stage re-prefill is used instead of
    cross-stage KV-cache continuation: the benchmark showed re-prefill is ~10%
    *faster* than cache continuation (vision is already cached), and it avoids
    the variable-length-cache stacking problem entirely. Returns one ``StagedRun``
    per trajectory, each carrying ``sampled_mask`` / ``new_ids`` for RL.
    """
    from transformers import GenerationConfig, StoppingCriteriaList, StopStringCriteria

    tokenizer = getattr(processor, 'tokenizer', processor)
    device = batch['input_ids'].device
    gen_in = dict(model_inputs(batch))
    prompt_ids = batch['input_ids']
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    prompt_len = int(prompt_ids.shape[-1])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    ends = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None

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
    was_training = core.training
    core.eval()
    try:
        with _vision_ctx(model, batch):
            for stage in STAGES + ('ANSWER',):
                stop = ANSWER_STOP if stage == 'ANSWER' else next_marker(stage)
                budget = max_answer_tokens if stage == 'ANSWER' else max_stage_tokens

                max_len = max(int(p.shape[-1]) for p in prefixes)
                input_ids = torch.full((group, max_len), pad_id, device=device, dtype=torch.long)
                attn = torch.zeros((group, max_len), device=device, dtype=torch.long)
                mm_batch = torch.zeros((group, max_len), dtype=mm_prompt.dtype, device=device)
                for i, p in enumerate(prefixes):
                    n = int(p.shape[-1])
                    input_ids[i, :n] = p
                    attn[i, :n] = 1
                    mlen = int(mm_prompt.shape[-1])
                    mm_batch[i, :min(n, mlen)] = mm_prompt[:min(n, mlen)]

                generation = GenerationConfig(
                    max_new_tokens=budget, do_sample=not greedy, temperature=1.0,
                    top_p=1.0, top_k=0, eos_token_id=ends, pad_token_id=pad_id,
                    use_cache=True, num_beams=1)
                stops = StoppingCriteriaList(
                    [StopStringCriteria(tokenizer=tokenizer, stop_strings=[stop])])

                kwargs = dict(gen_in_group)
                kwargs['input_ids'] = input_ids
                kwargs['attention_mask'] = attn
                kwargs['mm_token_type_ids'] = mm_batch
                kwargs['pixel_values'] = None

                out = core.generate(
                    generation_config=generation, stopping_criteria=stops, **kwargs)
                seqs = out.sequences if hasattr(out, 'sequences') else out
                new_all = seqs[:, max_len:]

                for i in range(group):
                    kept = _trim_new_at_stop(tokenizer, new_all[i], stop, device)
                    n_kept = int(kept.shape[-1])
                    body = tokenizer.decode(kept.tolist(), skip_special_tokens=False)
                    advanced = stage_finished(body, stage)
                    masks[i].extend([1] * n_kept)
                    prefixes[i] = torch.cat([prefixes[i], kept], dim=-1)
                    forced = 0
                    if stage != 'ANSWER':
                        extra = inject_after(stage, advanced=advanced)
                        inj = encode_ids(tokenizer, extra, device)[0]
                        forced = int(inj.shape[-1])
                        masks[i].extend([0] * forced)
                        prefixes[i] = torch.cat([prefixes[i], inj], dim=-1)
                    traces_per[i].append(StageTrace(stage, n_kept, forced, advanced, body, 0.0))
    finally:
        core.train(was_training)

    runs = []
    for i in range(group):
        new_ids = prefixes[i][prompt_len:].tolist()
        names = [t.name for t in traces_per[i]]
        ok = names[-1] == 'ANSWER' and all(s in names for s in STAGES)
        runs.append(StagedRun(
            ok, 'cache', prompt_len, len(new_ids), 0.0, traces_per[i],
            tokenizer.decode(new_ids, skip_special_tokens=False),
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
