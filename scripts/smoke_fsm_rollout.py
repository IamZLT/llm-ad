#!/usr/bin/env python3
"""Smoke test: FSM staged rollout mask alignment + sampling (one sample, group=2)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('PYTHONUNBUFFERED', '1')

import torch

from outcome.engine_multibox import datasets, load_model, validate_config
from outcome.inputs_multibox import OutcomeMultiboxCollator
from outcome.policy import generate_group_staged
from rl.grpo import move_batch
from utils.config import load_yaml_config
from utils.common import set_seed


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
    prompt_len = int(batch['prompt_len'][0])
    print(f'prompt_len={prompt_len}', flush=True)

    comps = generate_group_staged(model, processor, batch, cfg, group=3, sample=True)
    tokenizer = processor.tokenizer
    for i, c in enumerate(comps):
        n_ids = int(c.ids.numel())
        n_new = n_ids - prompt_len
        mlen = len(c.sampled_mask)
        ok_len = (n_new == mlen)
        # Decode controller-injected tokens (mask==0) and model-sampled heads.
        injected = []
        sampled_head = []
        for j, flag in enumerate(c.sampled_mask):
            tok = c.ids[prompt_len + j]
            txt = tokenizer.decode([int(tok)], skip_special_tokens=False).strip()
            if not flag:
                injected.append(txt)
            elif len(sampled_head) < 12:
                sampled_head.append(txt)
        # count stage markers in text
        n_stages = sum(1 for m in ('[understand]', '[compare]', '[localize]', '[confirm]') if m in c.text)
        print(f'--- comp {i}: n_ids={n_ids} n_new={n_new} mask_len={mlen} '
              f'len_ok={ok_len} stop={c.stop_reason} stages={n_stages}', flush=True)
        print(f'    injected(first 8): {injected[:8]}', flush=True)
        print(f'    sampled(head 12): {sampled_head}', flush=True)
        print(f'    text head: {c.text[:160]!r}', flush=True)

    # sanity: trajectories differ under sampling
    texts = [c.text for c in comps]
    print(f'unique texts: {len(set(texts))}/{len(texts)}', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
