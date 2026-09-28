#!/usr/bin/env python3
"""Merge per-shard eval jsonl files into one full summary.

Reads shard_*ofN.jsonl (plus optionally the full jsonl if present), concatenates
the per-sample records, and recomputes the summary via ``summarize`` (so the
merged numbers are identical to a single full run, not an average of shards).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from outcome.evaluate_multibox import summarize


def _loads(line):
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return json.loads(line.replace('NaN', 'null').replace('Infinity', 'null'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pattern', required=True,
                    help='glob pattern for shard jsonl files, e.g. "outputs/.../eval_mvtec_full_shard*of2.jsonl"')
    ap.add_argument('--out', required=True, help='merged summary .json path')
    args = ap.parse_args()

    files = sorted(Path().glob(args.pattern))
    if not files:
        raise SystemExit(f'no files matched {args.pattern}')
    rows = []
    for f in files:
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(_loads(line))
    print(f'merged {len(rows)} records from {len(files)} shard files: {[f.name for f in files]}')

    stats = summarize(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    # Also write the merged jsonl for downstream cross-analysis.
    merged = out.with_suffix('.jsonl')
    with merged.open('w') as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False, default=str) + '\n')
    print(f'wrote {out} and {merged}')
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
