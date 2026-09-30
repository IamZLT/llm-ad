"""Training-only targets for every legal observation action.

Runtime planning never imports this module. Each action carries the belief that
action is allowed to produce: stop keeps B0, zoom edits only its own box, and
global scan may add missed components.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

from outcome.planner import build_observation_actions
from outcome.protocol import iou
from outcome.state_quality import state_quality


PLANNER_STATE_WEIGHTS = {
    "perfect": 0.20,
    "undershoot": 0.20,
    "overshoot": 0.10,
    "shifted": 0.15,
    "missing_component": 0.20,
    "false_positive": 0.10,
    "empty_anomaly": 0.05,
}


@dataclass
class ActionSupervision:
    action: str
    evidence: str
    target_gain: float
    target_boxes: list


def _to_1000(box, orig_size):
    w, h = float(orig_size[0]), float(orig_size[1])
    return [float(box[0]) * 1000.0 / w, float(box[1]) * 1000.0 / h,
            float(box[2]) * 1000.0 / w, float(box[3]) * 1000.0 / h]


def _box_relation(cand, gt) -> str:
    overlap = iou(cand, gt)
    if overlap >= 0.85:
        return "candidate_complete"
    if overlap < 0.10:
        return "false_alarm"
    cw = max(cand[2] - cand[0], 1e-6)
    ch = max(cand[3] - cand[1], 1e-6)
    gw = max(gt[2] - gt[0], 1e-6)
    gh = max(gt[3] - gt[1], 1e-6)
    area_ratio = (cw * ch) / (gw * gh)
    if area_ratio < 0.75:
        return "boundary_undershoot"
    if area_ratio > 1.33:
        return "boundary_overshoot"
    return "candidate_shifted"


def _clip_gain(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _copy_boxes(boxes) -> list:
    return [[float(v) for v in box] for box in boxes]


def build_action_supervision(b0, gt_components, orig_size, cfg) -> List[ActionSupervision]:
    """One supervised prediction for every action ``build_observation_actions`` would offer."""
    boxes = _copy_boxes(b0 or [])
    gt_px = [list(g) for g in (gt_components or [])]
    gt_1000 = [_to_1000(g, orig_size) for g in gt_px] if orig_size else []
    q0 = state_quality(boxes, gt_px, orig_size, cfg)
    actions = build_observation_actions(boxes, cfg)
    rows = []
    for action in actions:
        if action.name == "stop":
            corrected = _copy_boxes(boxes)
            evidence = "no_new_evidence"
        elif action.name == "global_scan":
            missed = []
            for component in gt_1000:
                if not any(iou(cand, component) >= 0.1 for cand in boxes):
                    missed.append(list(component))
            corrected = _copy_boxes(boxes) + missed
            evidence = "missing_region" if missed else "no_missing_region"
        else:
            idx = action.candidate_index
            if idx is None or not (0 <= idx < len(boxes)):
                corrected = _copy_boxes(boxes)
                evidence = "false_alarm"
            elif not gt_1000:
                corrected = [box for j, box in enumerate(boxes) if j != idx]
                evidence = "false_alarm"
            else:
                cand = boxes[idx]
                best_i = max(range(len(gt_1000)), key=lambda i: iou(cand, gt_1000[i]))
                evidence = _box_relation(cand, gt_1000[best_i])
                if evidence == "false_alarm":
                    corrected = [box for j, box in enumerate(boxes) if j != idx]
                elif evidence == "candidate_complete":
                    corrected = _copy_boxes(boxes)
                else:
                    corrected = _copy_boxes(boxes)
                    corrected[idx] = list(gt_1000[best_i])
        gain = _clip_gain(state_quality(corrected, gt_px, orig_size, cfg) - q0)
        rows.append(ActionSupervision(action.name, evidence, gain, corrected))
    return rows


def render_action_supervisions(rows: Sequence[ActionSupervision]) -> str:
    return "\n".join(
        f"action={row.action}; evidence={row.evidence}; gain={row.target_gain:.2f}"
        for row in rows
    )


def update_for_evidence(evidence: str) -> str:
    return {
        "candidate_complete": "keep",
        "no_missing_region": "keep",
        "no_new_evidence": "keep",
        "boundary_undershoot": "refine",
        "boundary_overshoot": "refine",
        "candidate_shifted": "refine",
        "false_alarm": "reject",
        "missing_region": "discover",
        "uncertain": "refine",
    }.get(evidence, "refine")


def confirm_for_action(row: ActionSupervision, consistency: str = "supported") -> str:
    """Teacher confirm for the action that was actually executed."""
    return (
        f"observed={row.evidence};\n"
        f"consistency={consistency};\n"
        f"update={update_for_evidence(row.evidence)}"
    )


_CONTRADICTED_EVIDENCE = {
    "candidate_complete": "boundary_undershoot",
    "boundary_undershoot": "candidate_complete",
    "boundary_overshoot": "candidate_complete",
    "candidate_shifted": "candidate_complete",
    "false_alarm": "candidate_complete",
    "missing_region": "no_missing_region",
    "no_missing_region": "missing_region",
    "no_new_evidence": "missing_region",
}


def sample_prediction_view(row: ActionSupervision, sft_cfg: dict, rng=None):
    """Replay a possibly wrong prediction. The prefix that carries it is unlabeled."""
    rng = rng or random
    weights = (sft_cfg or {}).get("confirm_prediction_corruption") or {
        "supported": 0.60,
        "contradicted": 0.30,
        "uncertain": 0.10,
    }
    names = [name for name, weight in weights.items() if float(weight) > 0]
    total = sum(float(weights[name]) for name in names) or 1.0
    draw = rng.random() * total
    acc = 0.0
    kind = names[-1] if names else "supported"
    for name in names:
        acc += float(weights[name])
        if draw <= acc:
            kind = name
            break
    if kind == "contradicted":
        return _CONTRADICTED_EVIDENCE.get(row.evidence, "uncertain"), "contradicted"
    if kind == "uncertain":
        return "uncertain", "uncertain"
    return row.evidence, "supported"


def render_imagine_prefix(rows: Sequence[ActionSupervision], chosen: ActionSupervision,
                          predicted_evidence: str) -> str:
    """True plan, with only the executed action's evidence replaced in the prefix."""
    lines = []
    for row in rows:
        evidence = predicted_evidence if row.action == chosen.action else row.evidence
        lines.append(
            f"action={row.action}; evidence={evidence}; gain={row.target_gain:.2f}"
        )
    return "\n".join(lines)


def sample_observation_action(rows: Sequence[ActionSupervision], sft_cfg: dict, rng=None,
                              include_stop: bool = True) -> ActionSupervision:
    """Sample the executed action. Imagine still supervises every row."""
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
    if not include_stop:
        buckets["stop"] = []
    names = [name for name, group in buckets.items() if group and float(weights.get(name, 0.0)) > 0]
    if not names:
        usable = [row for row in rows if include_stop or row.action != "stop"]
        return usable[-1] if usable else rows[-1]
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


def _weighted_name(weights: dict, rng) -> str:
    names = [name for name, weight in weights.items() if float(weight) > 0]
    total = sum(float(weights[name]) for name in names)
    draw = rng.random() * total
    acc = 0.0
    chosen = names[-1]
    for name in names:
        acc += float(weights[name])
        if draw <= acc:
            chosen = name
            break
    return chosen


def _false_positive_box(gt_boxes) -> list:
    corners = ([0.0, 0.0, 80.0, 80.0], [920.0, 920.0, 1000.0, 1000.0],
               [0.0, 920.0, 80.0, 1000.0], [920.0, 0.0, 1000.0, 80.0])
    for box in corners:
        if all(iou(box, gt) < 0.1 for gt in gt_boxes):
            return box
    return [0.0, 0.0, 40.0, 40.0]


def sample_belief_boxes(gt_boxes, *, is_anomaly: bool, weights: Optional[dict] = None,
                        rng=None) -> tuple:
    """Artificial B0 for planner and updater prefixes. Never a localization target."""
    rng = rng or random
    boxes = [[float(v) for v in box] for box in (gt_boxes or [])]
    if not is_anomaly or not boxes:
        if rng.random() < 0.5:
            return "normal_empty", []
        return "normal_false_positive", [_false_positive_box(boxes)]
    weights = weights or PLANNER_STATE_WEIGHTS
    mode = _weighted_name(weights, rng)
    if mode == "perfect":
        return mode, [list(box) for box in boxes]
    if mode == "undershoot":
        return mode, [_scaled_box(box, 0.5) for box in boxes]
    if mode == "overshoot":
        return mode, [_scaled_box(box, 1.6) for box in boxes]
    if mode == "shifted":
        return mode, [_shifted_box(box, 0.45, rng) for box in boxes]
    if mode == "missing_component":
        if len(boxes) == 1:
            return mode, []
        kept = [list(box) for i, box in enumerate(boxes) if i != rng.randrange(len(boxes))]
        return mode, kept
    if mode == "false_positive":
        return mode, [_false_positive_box(boxes)]
    return "empty_anomaly", []


def _scaled_box(box, scale: float) -> list:
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    return [round(max(0.0, cx - w / 2.0), 3), round(max(0.0, cy - h / 2.0), 3),
            round(min(1000.0, cx + w / 2.0), 3), round(min(1000.0, cy + h / 2.0), 3)]


def _shifted_box(box, frac: float, rng) -> list:
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = x2 - x1, y2 - y1
    dx = (rng.random() * 2 - 1) * w * frac
    dy = (rng.random() * 2 - 1) * h * frac
    return [round(max(0.0, x1 + dx), 3), round(max(0.0, y1 + dy), 3),
            round(min(1000.0, x2 + dx), 3), round(min(1000.0, y2 + dy), 3)]


def _action_cost(name: str, cfg) -> float:
    planner = ((cfg or {}).get("outcome") or {}).get("planner") or {}
    if str(name).startswith("zoom_box_"):
        return float(planner.get("zoom_cost", 1.0))
    if name == "global_scan":
        return float(planner.get("global_scan_cost", 2.0))
    return 0.0


def summarize_dataset_beliefs(samples, cfg, limit: int = 400, seed: int = 0) -> dict:
    """Planner-target distribution on real boxes, without loading full training images."""
    from PIL import Image
    rng = random.Random(seed)
    chosen = list(samples or [])
    rng.shuffle(chosen)
    records = []
    for sample in chosen:
        if len(records) >= limit:
            break
        meta = sample.get("metadata") or sample
        comps = list(meta.get("component_bboxes") or [])
        is_anom = bool(meta.get("anomaly", meta.get("is_anomaly", False)))
        path = sample.get("full_img_path") or sample.get("image_path") or sample.get("image")
        if not path:
            continue
        with Image.open(path) as image:
            orig = image.size
        gt_1000 = [_to_1000(box, orig) for box in comps] if is_anom else []
        _mode, boxes = sample_belief_boxes(gt_1000, is_anomaly=is_anom and bool(gt_1000), rng=rng)
        records.append(build_action_supervision(boxes, comps if is_anom else [], orig, cfg))
    return summarize_supervision(records, cfg)


def summarize_supervision(records: Sequence[dict], cfg=None) -> dict:
    """Aggregate planner targets. ``records`` are lists of ActionSupervision rows.

    The best action is the one the runtime planner would pick: gain minus cost.
    """
    planner = ((cfg or {}).get("outcome") or {}).get("planner") or {}
    cost_weight = float(planner.get("cost_weight", 0.05))
    n = len(records)
    all_zero = 0
    zoom_pos = scan_pos = 0
    best = {"stop": 0, "zoom": 0, "global_scan": 0}
    gains = []
    evidence = {}
    for rows in records:
        if rows and all(abs(row.target_gain) < 1e-8 for row in rows):
            all_zero += 1
        if any(row.action.startswith("zoom_box_") and row.target_gain > 1e-8 for row in rows):
            zoom_pos += 1
        if any(row.action == "global_scan" and row.target_gain > 1e-8 for row in rows):
            scan_pos += 1
        winner = max(
            rows,
            key=lambda row: row.target_gain - cost_weight * _action_cost(row.action, cfg),
        )
        if winner.action.startswith("zoom_box_"):
            best["zoom"] += 1
        else:
            best[winner.action] = best.get(winner.action, 0) + 1
        for row in rows:
            gains.append(row.target_gain)
            evidence[row.evidence] = evidence.get(row.evidence, 0) + 1
    mean = sum(gains) / len(gains) if gains else 0.0
    var = sum((g - mean) ** 2 for g in gains) / len(gains) if gains else 0.0
    def rate(count):
        return count / n if n else 0.0
    return {
        "n": n,
        "all_zero_rate": rate(all_zero),
        "zoom_positive_rate": rate(zoom_pos),
        "scan_positive_rate": rate(scan_pos),
        "best_action_stop_rate": rate(best.get("stop", 0)),
        "best_action_zoom_rate": rate(best.get("zoom", 0)),
        "best_action_scan_rate": rate(best.get("global_scan", 0)),
        "gain_mean": mean,
        "gain_std": var ** 0.5,
        "evidence": evidence,
    }
