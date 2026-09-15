#!/usr/bin/env python3
"""Probe Staged Incremental Reasoning Decoding (feasibility, not training).

Same assistant turn, one <think> block, FSM-forced U→C→L→V→answer.
Cache path keeps past_key_values; oneshot is unconstrained generate.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/probe_staged_decode.py \
    --config configs/qwen35_2b_annos_probe_v2.yaml --base-llm --n 4

  PYTHONPATH=. python scripts/probe_staged_decode.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _trace_dict(t):
    return dict(name=t.name, n_sampled=t.n_sampled, n_forced=t.n_forced,
                hit_end=t.hit_end, seconds=round(t.seconds, 3),
                text=t.text[:800])


def _run_dict(r):
    return dict(ok=r.ok, mode=r.mode, prompt_len=r.prompt_len, total_new=r.total_new,
                seconds=round(r.seconds, 3), error=r.error, notes=r.notes,
                stages=[_trace_dict(s) for s in r.stages], text=r.text[:4000])


def dry_run() -> dict:
    from outcome.staged_decode import OPEN, STAGES, inject_after, next_marker, next_stage, stage_finished
    chain = ['U']
    while chain[-1] != 'ANSWER':
        chain.append(next_stage(chain[-1]))
    assert chain == ['U', 'C', 'L', 'V', 'ANSWER']
    assert next_stage('ANSWER') == 'DONE'
    try:
        next_stage('DONE')
        skipped = False
    except ValueError:
        skipped = True
    assert skipped
    assert not stage_finished('still thinking', 'U')
    assert stage_finished('... [compare]', 'U')
    inj_miss = inject_after('U', advanced=False)
    inj_hit = inject_after('U', advanced=True)
    assert '[compare]' in inj_miss
    assert inj_hit == '\n'
    assert '</think>' in inject_after('V', advanced=False)
    return dict(
        ok=True, mode='dry-run',
        chain=chain, cannot_skip=True,
        markers={s: dict(open=OPEN[s], next=next_marker(s)) for s in STAGES},
        inject_if_missing_U=inj_miss, inject_if_model_advanced_U=inj_hit,
    )


def _is_anom(sample):
    if not isinstance(sample, dict):
        return False
    if 'is_anomaly' in sample:
        return bool(sample['is_anomaly'])
    md = sample.get('metadata') or {}
    return bool(md.get('is_anomaly') if 'is_anomaly' in md else md.get('anomaly'))


def _pick_indices(ds, n):
    need_a = max(1, n // 2)
    need_n = n - need_a
    order = []
    for i in range(len(ds)):
        flag = _is_anom(ds.samples[i])
        if flag and need_a:
            order.append(i); need_a -= 1
        elif (not flag) and need_n:
            order.append(i); need_n -= 1
        if len(order) >= n:
            break
    if len(order) < n:
        order = list(range(min(n, len(ds))))
    return order


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', default='configs/qwen35_2b_annos_probe_v2.yaml')
    p.add_argument('--adapter', default=None)
    p.add_argument('--base-llm', action='store_true',
                   help='skip language LoRA merge; still load region_adapter.pt')
    p.add_argument('--n', type=int, default=1)
    p.add_argument('--split', choices=['dev', 'test'], default='test')
    p.add_argument('--max-stage-tokens', type=int, default=64)
    p.add_argument('--max-answer-tokens', type=int, default=128)
    p.add_argument('--modes', default='cache,oneshot',
                   help='comma list: cache,naive,oneshot')
    p.add_argument('--greedy', action='store_true', default=True)
    p.add_argument('--out', default='outputs/probe_staged_decode.json')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    if args.dry_run:
        report = dry_run()
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return

    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    import torch
    from outcome.engine_multibox import datasets, load_model, validate_config
    from outcome.inputs_multibox import OutcomeMultiboxCollator
    from outcome.protocol_multibox import parse_output, score_output
    from outcome.staged_decode import run_cached, run_naive, run_oneshot
    from rl.grpo import move_batch
    from utils.common import set_seed
    from utils.config import load_yaml_config

    cfg = load_yaml_config(args.config)
    validate_config(cfg)
    set_seed(int(cfg['training']['seed']))
    if args.base_llm:
        from models.anomaly_prior import AnomalyPrior
        from models.qwen35 import force_vision_eval, freeze_vision_encoder, setup_model_and_processor
        from models.region_injection import attach_region_adapter, ensure_region_token
        model, processor = setup_model_and_processor(cfg, for_inference=True, freeze_vision=True)
        ensure_region_token(processor, model)
        freeze_vision_encoder(model)
        if torch.cuda.is_available():
            model = model.cuda()
        force_vision_eval(model)
        prior = AnomalyPrior.from_qwen(model, cfg)
        attach_region_adapter(model, prior, cfg, cfg['outcome']['sft_adapter'])
    else:
        model, processor, prior = load_model(cfg, adapter=args.adapter, fresh_lora=False)
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    model.eval()
    _, dev, test = datasets(cfg, processor)
    ds = test if args.split == 'test' else dev
    order = _pick_indices(ds, int(args.n))
    collator = OutcomeMultiboxCollator(processor, prior, cfg)
    device = next(model.parameters()).device
    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    loc_cfg = {**(cfg.get('outcome', {}).get('localization') or {}),
               **(cfg.get('outcome', {}).get('reward') or {})}
    rows = []
    for i in order:
        item = ds[i]
        batch = move_batch(collator([item]), device)
        meta = (batch.get('_meta') or [{}])[0]
        rec = dict(i=i, image=meta.get('image_path'), class_name=meta.get('class_name'),
                   is_anomaly=bool(meta.get('is_anomaly')), runs={})
        print(f"[probe] sample {i} anom={rec['is_anomaly']} {rec['class_name']} {rec['image']}", flush=True)
        for mode in modes:
            if mode == 'cache':
                r = run_cached(model, processor, batch,
                               max_stage_tokens=args.max_stage_tokens,
                               max_answer_tokens=args.max_answer_tokens,
                               greedy=args.greedy)
            elif mode == 'naive':
                r = run_naive(model, processor, batch,
                              max_stage_tokens=args.max_stage_tokens,
                              max_answer_tokens=args.max_answer_tokens,
                              greedy=args.greedy)
            elif mode == 'oneshot':
                r = run_oneshot(model, processor, batch,
                                max_new_tokens=args.max_stage_tokens * 4 + args.max_answer_tokens,
                                greedy=args.greedy)
            else:
                raise ValueError(mode)
            blob = _run_dict(r)
            parsed = parse_output(r.text, thinking_required=True)
            try:
                sc = score_output(parsed, meta, float(cfg['outcome']['protocol_weight']), loc_cfg)
                blob['score'] = {k: (float(v) if isinstance(v, (int, float, bool)) else v)
                                 for k, v in sc.items() if k != 'matched_pairs'}
            except Exception as exc:
                blob['score'] = dict(error=f'{type(exc).__name__}: {exc}')
            blob['pred'] = dict(is_anomaly=parsed.get('is_anomaly'),
                                bboxes_2d=parsed.get('bboxes_2d'),
                                task_valid=parsed.get('task_valid'),
                                think_ok=parsed.get('think_ok'),
                                protocol_core=parsed.get('protocol_core'))
            rec['runs'][mode] = blob
            sc = blob.get('score') or {}
            print(f"  {mode}: ok={r.ok} correct={sc.get('correct')} miou={sc.get('mask_iou')} "
                  f"pred={blob['pred']['is_anomaly']} {r.seconds:.1f}s {r.error}", flush=True)
        rows.append(rec)
    summary = {}
    for mode in modes:
        scored = [r['runs'][mode] for r in rows if 'correct' in (r['runs'].get(mode, {}).get('score') or {})]
        if scored:
            summary[mode] = dict(
                n=len(scored),
                acc=sum(bool(x['score']['correct']) for x in scored) / len(scored),
                miou=sum(float(x['score'].get('mask_iou') or 0) for x in scored) / len(scored),
                task_valid=sum(bool(x['pred']['task_valid']) for x in scored) / len(scored),
                seconds=sum(x['seconds'] for x in scored) / len(scored),
            )
    report = dict(n=len(rows), modes=modes, base_llm=bool(args.base_llm), summary=summary, rows=rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(dict(out=str(out), n=len(rows), summary=summary), indent=2), flush=True)


if __name__ == '__main__':
    main()
