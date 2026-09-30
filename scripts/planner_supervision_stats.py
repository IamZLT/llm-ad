#!/usr/bin/env python3
"""Print the planner-target distribution before a world-model SFT run."""
from __future__ import annotations

import json

from outcome.planner_supervision import (
    build_action_supervision, sample_belief_boxes, summarize_supervision,
)

CFG = {"outcome": {"planner": {"fp_weight": 0.25, "max_zoom_candidates": 3}}}
GT = [[120.0, 80.0, 360.0, 300.0], [700.0, 600.0, 900.0, 880.0]]


def main():
    records = []
    for i in range(2000):
        _mode, boxes = sample_belief_boxes(GT, is_anomaly=True)
        records.append(build_action_supervision(boxes, GT, (1000, 1000), CFG))
        if i % 4 == 0:
            _mode, normal_boxes = sample_belief_boxes([], is_anomaly=False)
            records.append(build_action_supervision(normal_boxes, [], (1000, 1000), CFG))
    summary = summarize_supervision(records, CFG)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
