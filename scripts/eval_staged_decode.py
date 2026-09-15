#!/usr/bin/env python3
"""Full MVTec-200 eval: KV-cached staged decode vs oneshot on the base LLM.

Same sample selection, parsing and scoring as ``outcome.evaluate_multibox.evaluate``
so the summary JSON is directly comparable with the SFT / RL ``test.json`` files.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/eval_staged_decode.py \
    --config configs/qwen35_2b_annos_probe_v2.yaml --base-llm --n 200 \
    --modes cache,oneshot --out outputs/eval/staged_decode_mvtec200
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


class _Completion:
    """Minimal stand-in for ``outcome.policy.Completion`` (only .text/.stop_reason/.ids len are used)."""

    def __init__(self, text, stop_reason, n_ids):
        self.text = text
        self.stop_reason = stop_reason
        self.ids = list(range(n_ids))


def _stop_reason(run) -> str:
    if run.ok:
        return 'answer'
    return 'error' if run.error else 'length'


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--config', default='configs/qwen35_2b_annos_probe_v2.yaml')
    ap.add_argument('--base-llm', action=argparse.BooleanOptionalAction, default=True,
                    help='skip language LoRA merge; still load region_adapter.pt')
    ap.add_argument('--adapter', default=None,
                    help='optional RL LoRA adapter dir (SFT merge + this adapter)')
    ap.add_argument('--sft-adapter', default=None,
                    help='override outcome.sft_adapter (merged SFT)')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--split', choices=['dev', 'test'], default='test')
    ap.add_argument('--modes', default='cache,oneshot', help='comma list: cache,oneshot')
    ap.add_argument('--max-stage-tokens', type=int, default=96)
    ap.add_argument('--max-answer-tokens', type=int, default=192)
    ap.add_argument('--oneshot-tokens', type=int, default=768)
    ap.add_argument('--greedy', action='store_true', default=True)
    ap.add_argument('--classes', default=None,
                    help='comma list of class names to restrict eval to (e.g. bottle)')
    ap.add_argument('--out', default='outputs/eval/staged_decode_mvtec200')
    args = ap.parse_args()

    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    import torch
    from outcome.engine_multibox import datasets, validate_config
    from outcome.evaluate_multibox import make_record, stratified_eval_indices, summarize
    from outcome.inputs_multibox import OutcomeMultiboxCollator
    from outcome.protocol_multibox import parse_output_cfg, score_output
    from outcome.staged_decode import run_cached, run_oneshot
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
        from outcome.engine_multibox import load_model
        if args.sft_adapter:
            cfg['outcome']['sft_adapter'] = args.sft_adapter
        model, processor, prior = load_model(cfg, adapter=args.adapter, fresh_lora=False)

    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    model.eval()

    _, dev, test = datasets(cfg, processor)
    ds = test if args.split == 'test' else dev
    if args.classes:
        wanted = {c.strip().lower() for c in args.classes.split(',') if c.strip()}
        indices = [i for i in range(len(ds))
                   if str((ds.samples[i].get('metadata') or {}).get('class', '')).lower() in wanted
                   or str((ds.samples[i].get('metadata') or {}).get('class_name', '')).lower() in wanted]
        if args.n:
            indices = indices[:int(args.n)]
        print(f'[eval] class filter {wanted} -> {len(indices)} samples', flush=True)
    else:
        indices = stratified_eval_indices(ds, int(args.n), seed=int(cfg['training']['seed']))
    collator = OutcomeMultiboxCollator(processor, prior, cfg)
    device = next(model.parameters()).device
    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    max_boxes = int(cfg['outcome'].get('max_boxes', 16))
    loc_cfg = {**(cfg.get('outcome', {}).get('localization') or {}),
               **(cfg.get('outcome', {}).get('reward') or {})}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    per_mode = {m: [] for m in modes}
    with (out.with_suffix('.jsonl')).open('w') as stream:
        for pos, index in enumerate(indices):
            batch = move_batch(collator([ds[index]]), device)
            meta = (batch.get('_meta') or [{}])[0]
            prompt_len = int(batch['prompt_len'][0])
            rec = dict(i=index, image=meta.get('image_path'), class_name=meta.get('class_name'),
                       is_anomaly=bool(meta.get('is_anomaly')), runs={})
            for mode in modes:
                if mode == 'cache':
                    run = run_cached(model, processor, batch,
                                     max_stage_tokens=args.max_stage_tokens,
                                     max_answer_tokens=args.max_answer_tokens,
                                     greedy=args.greedy)
                elif mode == 'oneshot':
                    run = run_oneshot(model, processor, batch,
                                      max_new_tokens=args.oneshot_tokens,
                                      greedy=args.greedy)
                else:
                    raise ValueError(mode)
                parsed = parse_output_cfg(run.text, cfg, max_boxes=max_boxes)
                score = score_output(parsed, meta, float(cfg['outcome']['protocol_weight']),
                                     loc_cfg, max_boxes=max_boxes)
                comp = _Completion(run.text, _stop_reason(run), prompt_len + int(run.total_new))
                row = make_record(parsed, score, meta, comp, prompt_len, run.seconds, max_boxes)
                per_mode[mode].append(row)
                rec['runs'][mode] = dict(ok=run.ok, error=run.error, seconds=round(run.seconds, 3),
                                         total_new=run.total_new, pred=parsed['is_anomaly'],
                                         task_valid=parsed['task_valid'], think_ok=parsed['think_ok'],
                                         protocol_core=parsed['protocol_core'],
                                         mask_iou=score['mask_iou'], correct=score['correct'],
                                         text=run.text)
                print(f"[{args.split}] {pos+1}/{len(indices)} {mode}: ok={run.ok} "
                      f"pred={parsed['is_anomaly']} tv={parsed['task_valid']} "
                      f"miou={score['mask_iou']:.3f} {run.seconds:.1f}s", flush=True)
            stream.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
            stream.flush()

    report = dict(n=len(indices), modes=modes, base_llm=bool(args.base_llm), split=args.split)
    for mode in modes:
        stats = summarize(per_mode[mode])
        report[mode] = stats
        rows_path = out.parent / f'{out.name}_{mode}.jsonl'
        with rows_path.open('w') as f:
            for row in per_mode[mode]:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
        stats_path = out.parent / f'{out.name}_{mode}.json'
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
        print(f'\n===== {mode} (rows: {rows_path}, stats: {stats_path}) =====', flush=True)
        for k in ('n', 'balanced_accuracy', 'anomaly_recall', 'normal_correct_rate',
                  'mask_miou', 'task_valid_rate', 'protocol_core_rate', 'protocol_strict_rate',
                  'think_ok_rate', 'mean_seconds', 'mean_new_tokens'):
            print(f'  {k}: {stats.get(k)}', flush=True)
    report_path = out.with_suffix('.json')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'\nreport: {report_path}', flush=True)


if __name__ == '__main__':
    main()
