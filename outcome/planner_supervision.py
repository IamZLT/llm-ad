"""Training-only targets for every legal observation action.

Runtime planning never imports this module. Targets use ground-truth boxes to
label evidence and the localization change that would follow an oracle
correction of that action.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from outcome.planner import build_observation_actions
from outcome.protocol import iou
from outcome.protocol_multibox import set_localization_reward


@dataclass
class ActionSupervision:
    action: str
    evidence: str
    target_gain: float


def _to_1000(box, orig_size):
    w, h = float(orig_size[0]), float(orig_size[1])
    return [float(box[0]) * 1000.0 / w, float(box[1]) * 1000.0 / h,
            float(box[2]) * 1000.0 / w, float(box[3]) * 1000.0 / h]


def _box_relation(cand, gt) -> str:
    overlap = iou(cand, gt)
    if overlap >= 0.5:
        return "supported"
    if overlap < 0.1:
        return "false_alarm"
    cw, ch = max(cand[2] - cand[0], 1e-6), max(cand[3] - cand[1], 1e-6)
    gw, gh = max(gt[2] - gt[0], 1e-6), max(gt[3] - gt[1], 1e-6)
    if cw * ch < 0.7 * gw * gh:
        return "undershoot"
    if cw * ch > 1.4 * gw * gh:
        return "overshoot"
    return "shifted"


def _quality(boxes_1000, gt_px, orig_size) -> float:
    if not gt_px:
        return 0.0
    return float(set_localization_reward(boxes_1000, gt_px, orig_size)["reward"])


def _clip_gain(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def build_action_supervision(b0, gt_components, orig_size, cfg) -> List[ActionSupervision]:
    """One supervised prediction for every action ``build_observation_actions`` would offer."""
    boxes = [list(b) for b in (b0 or [])]
    gt_px = [list(g) for g in (gt_components or [])]
    gt_1000 = [_to_1000(g, orig_size) for g in gt_px] if orig_size else []
    q0 = _quality(boxes, gt_px, orig_size)
    actions = build_observation_actions(boxes, cfg)
    rows = []
    for action in actions:
        if action.name == "stop":
            rows.append(ActionSupervision(action.name, "no_new_evidence", 0.0))
            continue
        if action.name == "global_scan":
            missed = []
            for g, gpx in zip(gt_1000, gt_px):
                if not any(iou(c, g) >= 0.1 for c in boxes):
                    missed.append(g)
            if missed:
                corrected = boxes + missed
                gain = _clip_gain(_quality(corrected, gt_px, orig_size) - q0)
                rows.append(ActionSupervision(action.name, "missing_region", gain))
            else:
                rows.append(ActionSupervision(action.name, "complete_coverage", 0.0))
            continue
        idx = action.candidate_index
        if idx is None or not (0 <= idx < len(boxes)) or not gt_1000:
            rows.append(ActionSupervision(action.name, "false_alarm", 0.0))
            continue
        cand = boxes[idx]
        best_i = max(range(len(gt_1000)), key=lambda i: iou(cand, gt_1000[i]))
        evidence = _box_relation(cand, gt_1000[best_i])
        if evidence == "false_alarm":
            corrected = [b for j, b in enumerate(boxes) if j != idx]
        elif evidence == "supported":
            corrected = list(boxes)
        else:
            corrected = list(boxes)
            corrected[idx] = list(gt_1000[best_i])
        gain = 0.0 if evidence == "supported" else _clip_gain(
            _quality(corrected, gt_px, orig_size) - q0)
        rows.append(ActionSupervision(action.name, evidence, gain))
    return rows


def render_action_supervisions(rows: Sequence[ActionSupervision]) -> str:
    return "\n".join(
        f"action={row.action}; evidence={row.evidence}; gain={row.target_gain:.2f}"
        for row in rows
    )


def confirm_for_action(row: ActionSupervision) -> str:
    """Teacher confirm for the action that was actually executed."""
    update = {
        "supported": "keep",
        "complete_coverage": "keep",
        "no_new_evidence": "keep",
        "undershoot": "refine",
        "overshoot": "refine",
        "shifted": "refine",
        "false_alarm": "reject",
        "missing_region": "discover",
        "uncertain": "refine",
    }.get(row.evidence, "refine")
    return (
        f"observed={row.evidence};\n"
        f"consistency=supported;\n"
        f"update={update}"
    )


def sample_observation_action(rows: Sequence[ActionSupervision], sft_cfg: dict, rng=None) -> ActionSupervision:
    """Sample the executed action. Imagine still supervises every row."""
    import random
    rng = rng or random
    weights = (sft_cfg or {}).get("observation_action_sampling") or {
        "zoom": 0.5, "global_scan": 0.3, "stop": 0.2,
    }
    buckets = {"zoom": [], "global_scan": [], "stop": []}
    for row in rows:
        if row.action.startswith("zoom_box_"):
            buckets["zoom"].append(row)
        elif row.action in buckets:
            buckets[row.action].append(row)
    names = [name for name, group in buckets.items() if group and float(weights.get(name, 0.0)) > 0]
    if not names:
        return rows[-1]
    total = sum(float(weights[name]) for name in names)
    draw = rng.random() * total
    acc = 0.0
    chosen = names[-1]
    for name in names:
        acc += float(weights[name])
        if draw <= acc:
            chosen = name
            break
    group = buckets[chosen]
    return group[rng.randrange(len(group))]
