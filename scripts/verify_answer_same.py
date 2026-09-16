#!/usr/bin/env python3
"""Compare the PARSED answer (reward-relevant) across KV / re-prefill / batch.

If the answer JSON (is_anomaly + bboxes_2d) is identical across paths, the
reasoning-text divergence does not affect RL rewards, and re-prefill is a valid
fast batched rollout. If the answer differs, the rollout is semantically broken.
"""
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
from outcome.protocol_multibox import parse_output_cfg
from outcome.staged_decode import run_cached
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
    max_boxes = int(cfg['outcome'].get('max_boxes', 16))

    # test several samples (both normal & anomalous)
    results = []
    for idx in range(6):
        batch = move_batch(collator([dev[idx]]), device)
        meta = (batch.get('_meta') or [{}])[0]

        kv = run_cached(model, processor, batch, max_stage_tokens=96, max_answer_tokens=192, greedy=True)
        comps = generate_group_staged(model, processor, batch, cfg, group=2, sample=False)
        bt = comps[0].text

        pkv = parse_output_cfg(kv.text, cfg, max_boxes=max_boxes)
        pbt = parse_output_cfg(bt, cfg, max_boxes=max_boxes)

        ans_same = (pkv['is_anomaly'] == pbt['is_anomaly']
                    and pkv['bboxes_2d'] == pbt['bboxes_2d'])
        results.append((idx, bool(meta.get('is_anomaly')), ans_same,
                        pkv['is_anomaly'], pbt['is_anomaly'],
                        pkv['bboxes_2d'], pbt['bboxes_2d']))

        print(f'sample {idx} is_anom={bool(meta.get("is_anomaly"))} '
              f'answer_same={ans_same} '
              f'kv_anom={pkv["is_anomaly"]} batch_anom={pbt["is_anomaly"]}', flush=True)
        if not ans_same:
            print(f'  kv   bboxes={pkv["bboxes_2d"]}', flush=True)
            print(f'  batch bboxes={pbt["bboxes_2d"]}', flush=True)

    n_same = sum(1 for r in results if r[2])
    print(f'ANSWER-SAME {n_same}/{len(results)}', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
