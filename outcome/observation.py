"""Shared observation planning for the two-stage world-model flow.

Both training (SFT correction samples) and inference (``generate_inspection_group``)
route through ``plan_observations`` so the *same* observation policy decides what
the model gets to see before it commits in ``[confirm]``:

* A single candidate -> an adaptive local crop (zoom) around that candidate.
* No candidate, a multi-candidate set, or ``coverage_scan`` -> two overlapping
  full-image partitions (``global_scan``) so missing regions stay visible.

``plan_observations`` reads ONLY the image, the predicted candidates, and the
config. It must never read GT: the observation policy is what a deployed model
would execute, and GT leakage would make the observation baselines meaningless.
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


def plan_observations(test_img, boxes, orig_size, cfg):
    """Plan the observation windows for stage 2 (no GT access)."""
    if tuple(test_img.size) != tuple(orig_size):
        raise ValueError(
            "Observation planning requires the original-size image"
        )

    zcfg = (cfg.get("outcome") or {}).get("zoom") or {}

    if not boxes:
        return global_scan(test_img)

    # Multi-candidate sets first check global coverage, not just the first box.
    if len(boxes) > 1:
        return global_scan(test_img)

    # Ablation switch: a single candidate may still hide other components.
    if bool(zcfg.get("coverage_scan", False)):
        return global_scan(test_img)

    start_expand = float(zcfg.get("expand", 0.3))
    min_pad = float(zcfg.get("min_pad_frac", 0.04))
    max_area = float(zcfg.get("max_area_frac", 0.6))

    # Shrink the context stepwise without truncating the candidate itself.
    for scale in (1.0, 0.5, 0.0):
        zoom = make_zoom_crop(
            test_img,
            boxes[0],
            orig_size=orig_size,
            expand=start_expand * scale,
            min_pad_frac=min_pad * scale,
            max_area_frac=max_area,
        )
        if not zoom.degenerate:
            return [
                Observation(
                    image=zoom.image,
                    window_px=tuple(zoom.window_px),
                    kind="candidate",
                    candidate_index=0,
                )
            ]

    return global_scan(test_img)


def observation_prompt(class_name, observations, orig_size):
    w, h = orig_size
    lines = [
        f"Image 1 is a defect-free reference of {class_name}.",
        "Image 2 is the full inspection image.",
    ]

    for image_id, obs in enumerate(observations, start=3):
        x1, y1, x2, y2 = obs.window_px
        window = [
            round(x1 * 1000 / w, 1),
            round(y1 * 1000 / h, 1),
            round(x2 * 1000 / w, 1),
            round(y2 * 1000 / h, 1),
        ]

        text = (
            f"Image {image_id} shows full-image window {window}."
        )
        if obs.kind == "candidate":
            text += " Its red rectangle marks an existing candidate."
        else:
            text += " This is a search region, not a confirmed defect."

        lines.append(text)

    lines.append(
        "The earlier candidate set and [imagine] judgment may be wrong "
        "or incomplete. Check both candidate accuracy and missing regions. "
        "You may retain, modify, add, or remove boxes. "
        "Return the complete box set for Image 2, not just the crop. "
        "Use full-image coordinates normalized to 0-1000. "
        "Write [confirm], close </think>, then output <answer>."
    )
    lines.append(
        "The earlier judgment may be incorrect. "
        "Use the available images to reassess the candidates. "
        "Explain the specific visual evidence for retaining, modifying, "
        "or rejecting them. Do not merely repeat the earlier conclusion."
    )
    lines.append(
        "In the final description, summarize the visible abnormality and "
        "its approximate location, or state that no clear defect is found. "
        "Do not claim to have inspected an image that was not supplied."
    )
    return "\n".join(lines)
