"""One state-quality function for supervision, reward, and evaluation.

An empty belief on a normal image is the best state. Each unsupported box
lowers that quality, so deleting a false alarm has a positive gain.
"""
from __future__ import annotations

from outcome.protocol import iou
from outcome.protocol_multibox import set_localization_reward


def _fp_weight(cfg) -> float:
    if not isinstance(cfg, dict):
        return 0.25
    if "fp_weight" in cfg:
        return float(cfg["fp_weight"])
    nested = ((cfg.get("outcome") or {}).get("planner") or {})
    return float(nested.get("fp_weight", 0.25))


def _to_1000(box, orig_size):
    w, h = float(orig_size[0]), float(orig_size[1])
    return [float(box[0]) * 1000.0 / w, float(box[1]) * 1000.0 / h,
            float(box[2]) * 1000.0 / w, float(box[3]) * 1000.0 / h]


def state_quality(boxes_1000, gt_components, orig_size, cfg=None) -> float:
    """Quality of a 0-1000 box belief against pixel ground truth.

    Normal images (no ground-truth components) score ``1 - fp_weight * n_boxes``,
    clipped to [-1, 1]. Anomalous images use the set localization reward minus an
    explicit penalty for boxes that miss every component.
    """
    boxes = [list(b) for b in (boxes_1000 or [])]
    gt_px = [list(g) for g in (gt_components or [])]
    weight = _fp_weight(cfg)
    if not gt_px:
        return max(-1.0, 1.0 - weight * len(boxes))
    gt_1000 = [_to_1000(g, orig_size) for g in gt_px] if orig_size else []
    loc = float(set_localization_reward(boxes, gt_px, orig_size)["reward"]) if boxes else 0.0
    false_positives = 0
    for box in boxes:
        if max((iou(box, g) for g in gt_1000), default=0.0) < 0.10:
            false_positives += 1
    penalty = weight * false_positives / max(len(gt_1000), 1)
    return max(-1.0, min(1.0, loc - penalty))
