"""Zoom-crop visual action for the world-model ``[confirm]`` stage.

The model first localizes a candidate box in latent space, then — instead of
guessing whether the box's *extent* is correct from a handful of coarse global
tokens — we crop a padded window around the candidate, re-encode it at full
resolution, and draw the candidate outline on it. The model then *sees* whether
its box undershoots / overshoots the real defect boundary before committing in
``[confirm]``.

Critical design point (why not crop the box itself): cropping exactly to the box
boundary would hide everything outside it and make "box too small" undetectable.
We instead crop a *neighborhood* (``expand`` × box size per side, plus a minimum
pad), and draw the candidate rectangle inside it, so the gap between the red box
and the defect's true extent is directly visible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from PIL import Image

from utils.common import qwen_norm1000_to_original_pixels


@dataclass
class ZoomCrop:
    """A padded crop around a candidate box, with the box outline drawn on it."""
    image: Image.Image
    window_px: Tuple[int, int, int, int]   # (x1, y1, x2, y2) on the ORIGINAL image
    box_px: Tuple[int, int, int, int]      # candidate box on the ORIGINAL image
    orig_size: Tuple[int, int]
    degenerate: bool                        # True when the crop is skipped (no-op)


def _clamp_window(box: Tuple[int, int, int, int], w: int, h: int,
                  expand: float, min_pad_frac: float) -> Tuple[int, int, int, int]:
    """Expand ``box`` by ``expand`` per side (relative to box size) plus a floor."""
    x1, y1, x2, y2 = box
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
    pad_x = int(round(expand * bw))
    pad_y = int(round(expand * bh))
    # Floor the window so a tiny box still sees surrounding context.
    floor = int(round(min_pad_frac * max(w, h)))
    pad_x = max(pad_x, floor)
    pad_y = max(pad_y, floor)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    return int(x1), int(y1), int(x2), int(y2)


def crop_window_from_box(box_2d: List[float], orig_size: Tuple[int, int],
                         expand: float = 1.0, min_pad_frac: float = 0.12) -> Tuple[int, int, int, int]:
    """Compute the padded crop window (original-image px) around a 0-1000 box."""
    w, h = int(orig_size[0]), int(orig_size[1])
    box = qwen_norm1000_to_original_pixels(list(box_2d), (w, h))
    return _clamp_window(tuple(box), w, h, float(expand), float(min_pad_frac))


def _draw_box(image: Image.Image, box_px: Tuple[int, int, int, int],
              window: Tuple[int, int, int, int]) -> Image.Image:
    """Draw the candidate outline on the crop, offset by the window origin."""
    from PIL import ImageDraw
    x1, y1, x2, y2 = box_px
    wx1, wy1, _, _ = window
    draw = ImageDraw.Draw(image)
    draw.rectangle([x1 - wx1, y1 - wy1, x2 - wx1, y2 - wy1], outline=(255, 0, 0), width=2)
    return image


def make_zoom_crop(test_img: Image.Image, box_2d: Optional[List[float]],
                   orig_size: Optional[Tuple[int, int]] = None,
                   expand: float = 1.0, min_pad_frac: float = 0.12,
                   max_area_frac: float = 0.6) -> ZoomCrop:
    """Build the zoom crop for a single candidate box.

    Args:
        test_img: original (un-resized) inspection image.
        box_2d: candidate box in 0-1000 coords (None / empty -> no crop).
        orig_size: (w, h) of the original image; defaults to ``test_img.size``.
        expand: per-side padding as a fraction of the box size.
        min_pad_frac: floor padding as a fraction of the image's longer side.
        max_area_frac: if the padded window covers more than this fraction of the
            image, the crop is a no-op (a whole-image window adds no resolution).

    Returns:
        ZoomCrop with ``degenerate=True`` when cropping is skipped (the caller
        should then fall back to the global view only).
    """
    w, h = orig_size if orig_size is not None else test_img.size
    w, h = int(w), int(h)
    if not box_2d or len(box_2d) != 4:
        return ZoomCrop(image=test_img, window_px=(0, 0, w, h),
                        box_px=(0, 0, 0, 0), orig_size=(w, h), degenerate=True)
    box = qwen_norm1000_to_original_pixels(list(box_2d), (w, h))
    window = _clamp_window(tuple(box), w, h, float(expand), float(min_pad_frac))
    wx1, wy1, wx2, wy2 = window
    area = (wx2 - wx1) * (wy2 - wy1)
    if area <= 0 or area >= max_area_frac * w * h:
        return ZoomCrop(image=test_img, window_px=(0, 0, w, h),
                        box_px=tuple(box), orig_size=(w, h), degenerate=True)
    crop = test_img.crop((wx1, wy1, wx2, wy2))
    crop = _draw_box(crop, tuple(box), window)
    return ZoomCrop(image=crop, window_px=window, box_px=tuple(box),
                    orig_size=(w, h), degenerate=False)
