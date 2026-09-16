#!/usr/bin/env python3
"""Three-way compare: KV-cache continuation vs per-stage re-prefill (serial & batch).

Determines whether the batched FSM rollout diverges from the (correct) KV-cache
path because of *batching* or because of *per-stage re-prefill*. Greedy decode
is deterministic, so any mismatch is a real semantic difference, not noise.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('PYTHONUNBUFFERED', '1')

import torch
from transformers import GenerationConfig, StoppingCriteriaList, StopStringCriteria

from models.qwen35 import unwrap_model
from rl.grpo import model_inputs
from outcome.staged_decode import (STAGES, OPEN, ANSWER_STOP, next_marker,
                                   stage_finished, inject_after, _vision_ctx, encode_ids)
from outcome.engine_multibox import datasets, load_model, validate_config
from outcome.inputs_multibox import OutcomeMultiboxCollator
from outcome.policy import generate_group_staged
from outcome.staged_decode import run_cached
from rl.grpo import move_batch
from utils.config import load_yaml_config
from utils.common import set_seed


def run_serial_reprefill(model, processor, batch, max_stage_tokens=96, max_answer_tokens=192):
    """Per-stage re-prefill, single trajectory (no cross-stage KV cache)."""
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
            z = torch.zeros(mm_type.shape[0], need - cur, dtype=mm_type.dtype, device=mm_type.device)
            return torch.cat([mm_type, z], dim=-1)
        if cur > need:
            return mm_type[:, :need]
        return mm_type

    prefix = torch.cat([prompt_ids, encode_ids(tokenizer, OPEN['U'], device)], dim=-1)
    core = unwrap_model(model)
    was_training = core.training
    core.eval()
    try:
        with _vision_ctx(model, batch):
            for stage_i, stage in enumerate(STAGES + ('ANSWER',)):
                stop = ANSWER_STOP if stage == 'ANSWER' else next_marker(stage)
                budget = max_answer_tokens if stage == 'ANSWER' else max_stage_tokens
                generation = GenerationConfig(
                    max_new_tokens=budget, do_sample=False, temperature=1.0,
                    top_p=1.0, top_k=0, eos_token_id=ends, pad_token_id=pad,
                    use_cache=True, num_beams=1)
                stops = StoppingCriteriaList(
                    [StopStringCriteria(tokenizer=tokenizer, stop_strings=[stop])])
                kwargs = dict(gen_in)
                kwargs['input_ids'] = prefix
                kwargs['attention_mask'] = torch.ones_like(prefix)
                kwargs['mm_token_type_ids'] = align_mm(prefix)
                # Re-prefill re-embeds the FULL sequence (image tokens included)
                # every stage, so pixel_values + image_grid_thw must be present on
                # EVERY stage (the cached get_image_features provides the frozen
                # image embeddings). Dropping them only breaks stage 1+.
                kwargs['pixel_values'] = gen_in.get('pixel_values')
                old_len = int(prefix.shape[-1])
                out = core.generate(generation_config=generation, stopping_criteria=stops, **kwargs)
                seqs = out.sequences if hasattr(out, 'sequences') else out
                new = seqs[0, old_len:]
                body = tokenizer.decode(new.tolist(), skip_special_tokens=False)
                advanced = stage_finished(body, stage)
                prefix = seqs
                if stage != 'ANSWER':
                    extra = inject_after(stage, advanced=advanced)
                    prefix = torch.cat([prefix, encode_ids(tokenizer, extra, device)], dim=-1)
    finally:
        core.train(was_training)
    return tokenizer.decode(prefix[0, prompt_len:].tolist(), skip_special_tokens=False)


def main():
    cfg = load_yaml_config('configs/qwen35_2b_annos_probe_v2_fsm_rl.yaml')
    validate_config(cfg)
    set_seed(int(cfg['training']['seed']))
    model, processor, prior = load_model(cfg, adapter=None, fresh_lora=False)
    model.eval()
    _, dev, test = datasets(cfg, processor)
    collator = OutcomeMultiboxCollator(processor, prior, cfg)
    device = next(model.parameters()).device
    batch = move_batch(collator([dev[0]]), device)

    kv = run_cached(model, processor, batch, max_stage_tokens=96, max_answer_tokens=192, greedy=True).text
    rp = run_serial_reprefill(model, processor, batch)
    comps = generate_group_staged(model, processor, batch, cfg, group=2, sample=False)
    bt = comps[0].text

    print(f'kv == re-prefill(serial): {kv == rp}', flush=True)
    print(f'kv == batch(re-prefill):  {kv == bt}', flush=True)
    print(f're-prefill(serial) == batch: {rp == bt}', flush=True)
    for a, b, na, nb in ((kv, rp, 'kv', 'reprefill-serial'), (kv, bt, 'kv', 'batch'), (rp, bt, 'reprefill-serial', 'batch')):
        if a != b:
            n = min(len(a), len(b))
            i = next((k for k in range(n) if a[k] != b[k]), n)
            print(f'--- diff {na} vs {nb} at char {i}:', flush=True)
            print(f'  {na}: ...{a[max(0,i-30):i+30]!r}', flush=True)
            print(f'  {nb}: ...{b[max(0,i-30):i+30]!r}', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
