#!/usr/bin/env python3
"""Depth-wise progress probe for the Internal-Loop (recurrent-depth) stack.

For one SFT checkpoint, run teacher-forcing over a small batch and, for each
recurrent depth k, compute

    \\tilde H_k = Norm(H_k)          (H_k = raw hidden state BEFORE final norm)
    logits_k   = LMHead(\\tilde H_k)
    L_k        = CE(logits_k, answer labels)

and report, per depth, the answer CE split by anomaly/normal plus the decision
margin m_k = log p(true) - log p(false) at the ``is_anomaly`` value token.

Interpretation (the signal that matters, not hidden-state cosine):

    L_1 < L_2 < L_3 < L_4   => the recurrence is refining the answer
    L_1 > L_2 > L_3 > L_4   => recurrent depth is destroying the answer repr

    anomaly m_k falling from +a to -b  => depth pushes anomaly evidence to the
                                          normal basin (direct mechanism for the
                                          observed recall drop with K).

Run:
    python scripts/probe_loop_progress.py [config] [--n-anomaly A] [--n-normal N]
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('PYTHONUNBUFFERED', '1')

import torch
import torch.nn.functional as F

from models.looped_qwen import get_qwen_text_model
from models.qwen35 import unwrap_model
from models.region_injection import bind_region_injection
from models.vision_cache import bind_cached_image_features
from rl.grpo import forward_with_vision, move_batch
from outcome.engine_multibox import datasets, load_model, validate_config
from outcome.inputs_multibox import OutcomeMultiboxCollator
from train_region_sft import pack_sft_batch
from utils.config import load_yaml_config
from utils.common import set_seed


def _tok_id(tokenizer, word):
    # JSON boolean values are space-prefixed (``": true`` -> `` true``), so try
    # the space-prefixed token first.
    for w in (' ' + word, word):
        ids = tokenizer(w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            return ids[0]
    return None


def _decision_margin(logits_list, labels, true_id, false_id, prompt_len):
    """Per-depth margin log p(true)-log p(false) at the first true/false label token.

    For the answer-only target the first ``true``/``false`` token after the prompt
    is the ``is_anomaly`` JSON value (``is_anomaly`` precedes ``bboxes_2d`` and
    ``description``).
    """
    lab = labels[0]
    pos = None
    for p in range(int(prompt_len), int(lab.numel())):
        v = int(lab[p].item())
        if v == true_id or v == false_id:
            pos = p
            break
    if pos is None or pos == 0 or true_id is None or false_id is None:
        return [None] * len(logits_list)
    margins = []
    for logits in logits_list:
        lp = torch.log_softmax(logits[0, pos - 1, :].float(), dim=-1)
        margins.append(float((lp[true_id] - lp[false_id]).item()))
    return margins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config', nargs='?', default='configs/qwen35_2b_annos_probe_v2_loop4_rl.yaml')
    ap.add_argument('--n-anomaly', type=int, default=16)
    ap.add_argument('--n-normal', type=int, default=16)
    ap.add_argument('--sft-adapter', default=None,
                    help='override outcome.sft_adapter (probe a specific SFT checkpoint)')
    ap.add_argument('--loop-steps', type=int, default=None,
                    help='force outcome.loop.enabled=True + steps=K (probe bare recurrence at any depth)')
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    if args.sft_adapter:
        cfg.setdefault('outcome', {})['sft_adapter'] = args.sft_adapter
    if args.loop_steps is not None:
        lc = cfg.setdefault('outcome', {}).setdefault('loop', {})
        lc['enabled'] = True
        lc['steps'] = int(args.loop_steps)
    validate_config(cfg)
    set_seed(int(cfg['training']['seed']))
    model, processor, prior = load_model(cfg, fresh_lora=False)
    _, dev, _ = datasets(cfg, processor)
    collator = OutcomeMultiboxCollator(processor, prior, cfg)
    device = next(model.parameters()).device
    tokenizer = getattr(processor, 'tokenizer', processor)

    core = unwrap_model(model)
    text_model = get_qwen_text_model(model)
    K = int(text_model._loop_steps)
    text_model._capture_loop_states = True
    norm = text_model.norm
    lm_head = core.lm_head
    true_id = _tok_id(tokenizer, 'true')
    false_id = _tok_id(tokenizer, 'false')

    anom_idx = [i for i, s in enumerate(dev.samples)
                if (s.get('metadata') or {}).get('anomaly', False)]
    norm_idx = [i for i, s in enumerate(dev.samples)
                if not (s.get('metadata') or {}).get('anomaly', False)]
    sel = anom_idx[:args.n_anomaly] + norm_idx[:args.n_normal]

    ce_sum = [[0.0] * K for _ in range(3)]  # [overall, anomaly, normal]
    ce_cnt = [[0] * K for _ in range(3)]
    margin_anom = [[] for _ in range(K)]
    margin_norm = [[] for _ in range(K)]

    was_training = core.training
    core.eval()
    try:
        with torch.no_grad():
            for i in sel:
                batch = move_batch(collator([dev[i]]), device)
                is_anom = bool(batch['_meta'][0]['is_anomaly'])
                packed, _ = pack_sft_batch(collator, device, [dev[i]], tokenizer, True,
                                           thinking=False, answer_only=True)
                adapter = getattr(core, 'region_adapter', None)
                with bind_cached_image_features(core, packed['image_embeds']), \
                        bind_region_injection(core, adapter, packed['region_raw'],
                                              int(packed['region_token_id'])):
                    forward_with_vision(core, packed['gen_in'], packed['input_ids'],
                                        packed['attention_mask'])
                states = text_model._last_loop_states
                assert states is not None and len(states) == K, \
                    f'expected {K} captured states, got {None if states is None else len(states)}'
                logits_list = [lm_head(norm(s)) for s in states]
                labels = packed['labels']
                prompt_len = int(batch['prompt_len'][0])
                margins = _decision_margin(logits_list, labels, true_id, false_id, prompt_len)
                for k in range(K):
                    sl = logits_list[k][:, :-1, :].contiguous().float()
                    tl = labels[:, 1:].contiguous()
                    loss = F.cross_entropy(sl.view(-1, sl.size(-1)), tl.view(-1), ignore_index=-100)
                    g = 1 if is_anom else 2
                    ce_sum[0][k] += float(loss); ce_cnt[0][k] += 1
                    ce_sum[g][k] += float(loss); ce_cnt[g][k] += 1
                    (margin_anom if is_anom else margin_norm)[k].append(margins[k])
    finally:
        core.train(was_training)
        text_model._capture_loop_states = False

    names = ['overall', 'anomaly', 'normal']
    print(f'\n== depth-wise answer CE (K={K}) ==')
    print('depth  ' + '  '.join(f'{n:>9}' for n in names))
    for k in range(K):
        row = f'  {k + 1}   '
        for g in range(3):
            row += f'{ce_sum[g][k] / max(1, ce_cnt[g][k]):>9.4f}'
        print(row)

    print(f'\n== is_anomaly decision margin (log p(true) - log p(false)) ==')
    for k in range(K):
        ma_vals = [m for m in margin_anom[k] if m is not None]
        mn_vals = [m for m in margin_norm[k] if m is not None]
        ma = sum(ma_vals) / len(ma_vals) if ma_vals else float('nan')
        mn = sum(mn_vals) / len(mn_vals) if mn_vals else float('nan')
        print(f'  depth {k + 1}: anomaly={ma:+.3f}  normal={mn:+.3f}')

    print('\n(anomaly margin should be >0, normal margin <0; a falling anomaly '
          'margin with depth is the mechanism behind the recall drop)', flush=True)


if __name__ == '__main__':
    main()
