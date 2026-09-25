"""Synthetic defect injection for industrial anomaly detection.

Insert irregular textured blobs (small / medium / large) into an inspection image
so we obtain (image, component GT boxes) pairs for free. The anomaly-prior H-map
naturally lights up on the inserted patch because it encodes ref-vs-test patch
distance, so a synthetic defect is self-consistent with the whole H pipeline
(H-Box geometry tokens + H-VPT cross-attention + H candidate text all see it).

No scipy: everything uses numpy + cv2 (both already required by the pipeline).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# size bin -> (min, max) defect area as a fraction of the full image area.
# Calibrated to the real VisA mask_area_fraction distribution: P50=0.19%,
# P75=0.45%, P90=1.67%, P95=2.9%. These bins sit slightly above the raw
# percentiles to stay above the H-map patch granularity (~0.5%/patch at 448
# vision factor 32) so the anomaly-prior still lights up on the injected patch.
SIZE_BINS: Dict[str, Tuple[float, float]] = {
    "small": (0.0015, 0.006),
    "medium": (0.006, 0.020),
    "large": (0.020, 0.060),
}
# Number of blobs per size bin: small defects stay a single blob so they are not
# fragmented below the H-map patch granularity.
N_BLOBS: Dict[str, Tuple[int, int]] = {
    "small": (1, 1),
    "medium": (1, 2),
    "large": (1, 3),
}
DEFAULT_SIZE_BINS: List[str] = ["small", "medium", "large"]
_TEXTURES = ["perlin", "gauss", "cutout"]


def sample_defect_spec(rng, size_bins: Optional[List[str]] = None) -> Dict:
    """Draw a reproducible defect spec: a size bin plus a seed for the exact shape.

    ``rng`` is a ``random.Random`` (the spec is chosen at scan time); the returned
    ``seed`` is later fed to ``np.random.RandomState`` inside ``synthesize_defect``
    so the same sample always produces the same pixels and GT boxes.
    """
    bins = list(size_bins or DEFAULT_SIZE_BINS)
    return {"size_bin": rng.choice(bins), "seed": rng.randint(0, 2**31 - 1)}


def _value_noise(h: int, w: int, rng: np.random.RandomState, octaves: int = 4) -> np.ndarray:
    """Multi-octave value noise in [0,1] via bilinear upsample (no scipy)."""
    acc = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    total = 0.0
    for o in range(octaves):
        scale = 2**o
        gh, gw = max(2, h // scale), max(2, w // scale)
        grid = rng.uniform(0.0, 1.0, size=(gh + 1, gw + 1)).astype(np.float32)
        acc += amp * cv2.resize(grid, (w, h), interpolation=cv2.INTER_LINEAR)
        total += amp
        amp *= 0.5
    return acc / max(total, 1e-6)


def _blob_field(H, W, center, rx, ry, angle, rng):
    """Soft elliptical field in [0,1] with a noisy, irregular boundary.

    Computed only over the blob's local patch (not the full image) and returned
    together with its top-left offset so it can be folded into a global mask.
    """
    rx, ry = max(rx, 2.0), max(ry, 2.0)
    pad = max(2.0, max(rx, ry))
    x0 = int(max(0, center[0] - rx - pad))
    x1 = int(min(W, center[0] + rx + pad))
    y0 = int(max(0, center[1] - ry - pad))
    y1 = int(min(H, center[1] + ry + pad))
    ph, pw = y1 - y0, x1 - x0
    if ph <= 0 or pw <= 0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    cx, cy = float(center[0]), float(center[1])
    ca, sa = np.cos(angle), np.sin(angle)
    xr = (xx - cx) * ca + (yy - cy) * sa
    yr = -(xx - cx) * sa + (yy - cy) * ca
    d = (xr / rx) ** 2 + (yr / ry) ** 2
    noise = _value_noise(ph, pw, rng, octaves=3) * 0.3 - 0.15
    return np.clip(1.0 - d - noise, 0.0, 1.0), y0, x0


def _texture(h, w, kind, rng, source_bgr=None):
    """Grayscale fill texture in [0,1] for the defect body."""
    if kind == "perlin":
        return _value_noise(h, w, rng, octaves=5)
    if kind == "gauss":
        n = rng.normal(0.5, 0.25, size=(h, w)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), sigmaX=1.2)
        return (n - n.min()) / (n.max() - n.min() + 1e-6)
    # cutout: crop a random patch from the source image, grayscale + normalize.
    if source_bgr is not None:
        sh, sw = source_bgr.shape[:2]
        if sh > 8 and sw > 8:
            ch = rng.randint(8, max(9, sh))
            cw = rng.randint(8, max(9, sw))
            sy = rng.randint(0, max(1, sh - ch))
            sx = rng.randint(0, max(1, sw - cw))
            patch = source_bgr[sy:sy + ch, sx:sx + cw]
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)
            return (gray - gray.min()) / (gray.max() - gray.min() + 1e-6)
    return _value_noise(h, w, rng, octaves=5)


def _mask_boxes(binary: np.ndarray, min_area: int = 16) -> List[List[int]]:
    """xyxy integer boxes per 8-connected component (matches ``extract_targets_from_mask``)."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes: List[List[int]] = []
    for lab in range(1, n):
        x = int(stats[lab, cv2.CC_STAT_LEFT])
        y = int(stats[lab, cv2.CC_STAT_TOP])
        w = int(stats[lab, cv2.CC_STAT_WIDTH])
        h = int(stats[lab, cv2.CC_STAT_HEIGHT])
        if w < 2 or h < 2 or w * h < min_area:
            continue
        boxes.append([x, y, x + w, y + h])
    return boxes


def synthesize_defect(image_bgr: np.ndarray, rng: np.random.RandomState, size_bin: str = "medium"):
    """Inject one synthetic defect into ``image_bgr`` (H, W, 3) uint8 BGR.

    Returns ``(synthesized_image, component_bboxes, union_bbox)`` with xyxy
    integer pixel coordinates (half-open). ``component_bboxes`` follows the same
    convention as ``data.scan.extract_targets_from_mask`` (one box per disjoint
    8-connected component), so downstream mask_iou / set-reward logic works
    unchanged.
    """
    H, W = image_bgr.shape[:2]
    area = float(H * W)
    lo, hi = SIZE_BINS.get(size_bin, SIZE_BINS["medium"])
    # 1.35x compensates for image-edge clipping + morphological open (which shrinks
    # the final mask area); net effect keeps the delivered area inside the bin.
    target_area = area * rng.uniform(lo, hi) * 1.35

    mask = np.zeros((H, W), dtype=np.float32)
    lo_n, hi_n = N_BLOBS.get(size_bin, (1, 2))
    n_blobs = lo_n if lo_n >= hi_n else int(rng.randint(lo_n, hi_n))
    remaining = target_area
    for i in range(n_blobs):
        share = remaining / max(1, n_blobs - i)
        remaining -= share
        r = float(np.sqrt(max(share, 4.0) / np.pi))
        margin = int(max(r, 2.0))
        cx = rng.uniform(margin, max(margin + 1.0, W - margin))
        cy = rng.uniform(margin, max(margin + 1.0, H - margin))
        aspect = rng.uniform(0.5, 2.0)
        rx = r * np.sqrt(aspect)
        ry = r / np.sqrt(aspect)
        angle = rng.uniform(0.0, np.pi)
        blob = _blob_field(H, W, (cx, cy), rx, ry, angle, rng)
        if blob is None:
            continue
        field, y0, x0 = blob
        mask[y0:y0 + field.shape[0], x0:x0 + field.shape[1]] = np.maximum(
            mask[y0:y0 + field.shape[0], x0:x0 + field.shape[1]], field)

    binary = (mask > 0.5).astype(np.uint8)
    if int(binary.sum()) < 4:
        binary = (mask >= mask.max() * 0.3).astype(np.uint8)
    # Morphological cleanup: open removes isolated specks, close fills pinholes so
    # the defect reads as a few contiguous regions instead of noise fragments.
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)
    ys, xs = np.nonzero(binary)
    if ys.size == 0:
        return image_bgr, [], []
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    ph, pw = y1 - y0, x1 - x0

    kind = rng.choice(_TEXTURES)
    tex = _texture(ph, pw, str(kind), rng, source_bgr=image_bgr)

    base = rng.uniform(0.0, 1.0, size=3).astype(np.float32)
    if rng.random() < 0.5:
        base *= 0.4
    else:
        base = 0.4 + base * 0.6
    color = tex[..., None] * base[None, None, :] * 255.0

    alpha = cv2.GaussianBlur(binary[y0:y1, x0:x1].astype(np.float32), (0, 0), sigmaX=1.5)
    alpha = alpha[..., None] / max(float(alpha.max()), 1e-6)

    out = image_bgr.copy()
    region = out[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = np.clip((1.0 - alpha) * region + alpha * color, 0, 255).astype(np.uint8)

    comps = _mask_boxes(binary) or [[x0, y0, x1, y1]]
    union = [min(b[0] for b in comps), min(b[1] for b in comps),
             max(b[2] for b in comps), max(b[3] for b in comps)]
    return out, comps, union
