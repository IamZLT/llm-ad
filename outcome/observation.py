"""Execute an observation action. This module does not choose the action.

``planner.py`` selects ``zoom_box_i``, ``global_scan``, or ``stop``. A failed
zoom stays a failed zoom: the environment must not silently replace it with a
scan, or the executed action would no longer match the action the world model
was scored on.

``plan_observations`` remains the old rule-planner helper for scripts that still
want "no box or many boxes -> scan, one box -> crop".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from PIL import Image

from outcome.zoom_crop import make_zoom_crop


@dataclass
class Observation:
    image: Image.Image
    window_px: Tuple[int, int, int, int]
    kind: str
    candidate_index: Optional[int] = None


@dataclass
class ObservationExecution:
    action: str
    observations: List[Observation]
    executed: bool
    skip_reason: Optional[str] = None


def global_scan(test_img):
    """Two overlapping partitions that jointly cover the full image."""
    w, h = test_img.size

    if w >= h:
        windows = [
            (0, 0, max(1, int(w * 0.6)), h),
            (int(w * 0.4), 0, w, h),
        ]
    else:
        windows = [
            (0, 0, w, max(1, int(h * 0.6))),
            (0, int(h * 0.4), w, h),
        ]

    return [
        Observation(
            image=test_img.crop(window),
            window_px=window,
            kind="scan",
        )
        for window in windows
    ]


def execute_observation_action(test_img, boxes, orig_size, action, cfg):
    """Run ``action``. Zoom failure does not become ``global_scan``."""
    if tuple(test_img.size) != tuple(orig_size):
        raise ValueError("Observation execution requires the original-size image")
    action = str(action or "")
    if action == "stop":
        return ObservationExecution(action=action, observations=[], executed=False)
    if action == "global_scan":
        return ObservationExecution(
            action=action, observations=global_scan(test_img), executed=True)
    if action.startswith("zoom_box_"):
        try:
            idx = int(action[len("zoom_box_"):])
        except ValueError:
            return ObservationExecution(
                action=action, observations=[], executed=False,
                skip_reason="invalid_zoom_action")
        if not (0 <= idx < len(boxes)):
            return ObservationExecution(
                action=action, observations=[], executed=False,
                skip_reason="candidate_index_out_of_range")
        zcfg = (cfg.get("outcome") or {}).get("zoom") or {}
        start_expand = float(zcfg.get("expand", 0.3))
        min_pad = float(zcfg.get("min_pad_frac", 0.04))
        max_area = float(zcfg.get("max_area_frac", 0.6))
        for scale in (1.0, 0.5, 0.0):
            zoom = make_zoom_crop(
                test_img, boxes[idx], orig_size=orig_size,
                expand=start_expand * scale,
                min_pad_frac=min_pad * scale,
                max_area_frac=max_area,
            )
            if not zoom.degenerate:
                obs = Observation(
                    image=zoom.image,
                    window_px=tuple(zoom.window_px),
                    kind="candidate",
                    candidate_index=idx,
                )
                return ObservationExecution(
                    action=action, observations=[obs], executed=True)
        return ObservationExecution(
            action=action, observations=[], executed=False,
            skip_reason="degenerate_zoom")
    return ObservationExecution(
        action=action, observations=[], executed=False,
        skip_reason="unknown_action")


def rule_plan_observations(test_img, boxes, orig_size, cfg):
    """Legacy rule planner: empty/multi/coverage -> scan, else crop box 0.

    A degenerate single-box crop still falls back to a scan. That fallback is
    only for this baseline. ``execute_observation_action`` does not do it.
    """
    from outcome.planner import select_rule_action
    action = select_rule_action(boxes, cfg)
    execution = execute_observation_action(test_img, boxes, orig_size, action, cfg)
    if execution.executed:
        return execution.observations
    return global_scan(test_img)


def plan_observations(test_img, boxes, orig_size, cfg):
    """Backward-compatible name for ``rule_plan_observations``."""
    return rule_plan_observations(test_img, boxes, orig_size, cfg)


def observation_prompt(class_name, observations, orig_size, selected_action=None,
                       predicted_evidence=None, predicted_gain=None,
                       include_full_test=False):
    w, h = orig_size
    lines = [f"Image 1 is a defect-free reference of {class_name}."]
    next_image = 2
    if include_full_test:
        lines.append("Image 2 is the full inspection image.")
        next_image = 3
    else:
        lines.append(
            "The full inspection image is not shown again. "
            "Use only the supplied observation images and the frozen candidate text."
        )

    for image_id, obs in enumerate(observations, start=next_image):
        x1, y1, x2, y2 = obs.window_px
        window = [
            round(x1 * 1000 / w, 1),
            round(y1 * 1000 / h, 1),
            round(x2 * 1000 / w, 1),
            round(y2 * 1000 / h, 1),
        ]
        text = f"Image {image_id} shows full-image window {window}."
        if obs.kind == "candidate":
            text += " Its red rectangle marks an existing candidate."
        else:
            text += " This is a search region, not a confirmed defect."
        lines.append(text)

    if not observations:
        if selected_action == "stop":
            lines.append(
                "The observation controller selected action=stop. "
                "No additional visual observation was acquired. "
                "Commit the current belief and output the final answer. "
                "Do not invent unseen evidence."
            )
        else:
            lines.append(
                f"The observation controller selected action={selected_action}, "
                "but that observation could not be acquired. "
                "No substitute crop or scan was supplied. "
                "Commit the current belief and do not invent unseen evidence."
            )
    else:
        lines.append(
            "The earlier candidate set and [imagine] judgment may be wrong "
            "or incomplete. Check both candidate accuracy and missing regions. "
            "You may retain, modify, add, or remove boxes. "
            "Return the complete box set for Image 2, not just the crop. "
            "Use full-image coordinates normalized to 0-1000. "
            "Write [confirm], close </think>, then output <answer>."
        )
    if selected_action:
        lines.append(
            f"The observation controller selected action={selected_action}."
        )
    if predicted_evidence is not None and predicted_gain is not None:
        lines.append(
            "Before this observation, the world model predicted "
            f"evidence={predicted_evidence} with expected_gain={float(predicted_gain):.3f}."
        )
        lines.append(
            "Now inspect only the actual supplied observation evidence. "
            "In [confirm], report what was actually observed, whether it "
            "supports or contradicts the pre-observation prediction, and "
            "how the current box belief should be updated. "
            "Use observed=<type>; consistency=<supported|contradicted|uncertain>; "
            "update=<keep|refine|reject|discover>."
        )
    lines.append(
        "In the final description, summarize the visible abnormality and "
        "its approximate location, or state that no clear defect is found. "
        "Do not claim to have inspected an image that was not supplied."
    )
    return "\n".join(lines)
