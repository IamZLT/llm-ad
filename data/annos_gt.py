"""MMR-AD / ANNOS human boxes as optional detection GT.

Coordinates are labeled on a canvas that keeps aspect ratio and scales the
shorter edge to 448 (not a stretched 448×448 square).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from PIL import Image

PAPER_SHORT_EDGE = 448.0
PAPER_CANVAS = PAPER_SHORT_EDGE

_REPO = Path(__file__).resolve().parents[1]
DEFAULT_ANNOS_ROOT = _REPO / "datasets" / "ANNOS"

_LAYOUT_DIR = {"visa": "visa", "mvtec": "mvtec"}


def paper_canvas_size(orig_size, resized_width=None, resized_height=None):
    """Return the (W, H) canvas ANNOS used when labeling ``bbox_2d``."""
    if resized_width and resized_height:
        return float(resized_width), float(resized_height)
    w, h = float(orig_size[0]), float(orig_size[1])
    short = min(w, h) or 1.0
    scale = PAPER_SHORT_EDGE / short
    return w * scale, h * scale


def paper_to_px(box, size, canvas: float | None = None, resized_width=None, resized_height=None):
    """Map ANNOS ``bbox_2d`` back to original pixels."""
    w, h = float(size[0]), float(size[1])
    if canvas is not None and resized_width is None and resized_height is None:
        cw = ch = float(canvas)
    else:
        cw, ch = paper_canvas_size(size, resized_width, resized_height)
    return [
        box[0] * w / cw,
        box[1] * h / ch,
        box[2] * w / cw,
        box[3] * h / ch,
    ]


def union_boxes(boxes):
    if not boxes:
        return None
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def annos_entry_name(layout: str, defect_type: str, image_path: str) -> str:
    stem = Path(image_path).stem
    layout = (layout or "").lower()
    if layout == "visa":
        return f"anomaly-{stem}.png"
    return f"{defect_type}-{stem}.png"


@lru_cache(maxsize=128)
def load_bbox_annos(annos_root: str, layout: str, cls: str):
    folder = _LAYOUT_DIR.get((layout or "").lower())
    if not folder:
        return ()
    path = Path(annos_root) / folder / cls / "bbox_annos.json"
    if not path.is_file():
        return ()
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        return ()
    return tuple((e.get("name"), json.dumps(e, ensure_ascii=False)) for e in data if isinstance(e, dict))


def _index(annos_root: str, layout: str, cls: str) -> dict:
    out = {}
    for name, raw in load_bbox_annos(annos_root, layout, cls):
        if name:
            out[name] = json.loads(raw)
    return out


def boxes_px_from_entry(entry: dict, orig_size) -> list:
    rw, rh = entry.get("resized_width"), entry.get("resized_height")
    boxes = []
    for b in entry.get("bboxes") or []:
        raw = b.get("bbox_2d")
        if not raw or len(raw) != 4:
            continue
        boxes.append(paper_to_px([float(v) for v in raw], orig_size, resized_width=rw, resized_height=rh))
    return boxes


def lookup_annos_boxes(sample: dict, annos_root: str | Path):
    """Return original-pixel boxes or None if this sample has no ANNOS entry."""
    meta = sample.get("metadata") or {}
    if not meta.get("anomaly"):
        return None
    layout = str(meta.get("layout") or "")
    cls = str(meta.get("class") or "")
    path = sample.get("full_img_path") or sample.get("image")
    if not (layout and cls and path):
        return None
    name = annos_entry_name(layout, str(meta.get("defect_type") or ""), path)
    idx = _index(str(annos_root), layout, cls)
    entry = idx.get(name)
    if entry is None and layout == "visa":
        entry = idx.get(f"anomaly-{Path(path).stem.zfill(3)}.png")
    if entry is None:
        return None
    with Image.open(path) as im:
        size = im.size
    boxes = boxes_px_from_entry(entry, size)
    return boxes or None


def apply_annos_gt(samples: list, annos_root: str | Path | None = None) -> dict:
    """Replace mask-derived component boxes with ANNOS human boxes when present.

    Normals stay empty. Anomalies without an ANNOS entry keep mask GT.
    """
    root = Path(annos_root) if annos_root else DEFAULT_ANNOS_ROOT
    stats = dict(replaced=0, fallback_mask=0, normal=0, empty_annos=0)
    if not root.is_dir():
        raise FileNotFoundError(f"ANNOS root not found: {root}")
    for sample in samples:
        meta = sample.setdefault("metadata", {})
        if not meta.get("anomaly"):
            meta.setdefault("gt_source", "none")
            stats["normal"] += 1
            continue
        boxes = lookup_annos_boxes(sample, root)
        if not boxes:
            meta.setdefault("gt_source", "mask")
            stats["fallback_mask"] += 1
            continue
        union = union_boxes(boxes)
        path = sample.get("full_img_path") or sample.get("image")
        with Image.open(path) as im:
            ow, oh = im.size
        meta["component_bboxes"] = boxes
        meta["bbox"] = union
        meta["num_components"] = len(boxes)
        if union is not None:
            meta["union_area_fraction"] = float(
                (union[2] - union[0]) * (union[3] - union[1]) / max(ow * oh, 1)
            )
        meta["gt_source"] = "annos"
        stats["replaced"] += 1
    return stats


def apply_gt_source(samples: list, cfg: dict, split: str = "") -> list:
    data_cfg = cfg.get("data") or {}
    source = str(data_cfg.get("gt_source") or "mask").lower()
    if source in ("mask", "", "none"):
        return samples
    if source != "annos":
        raise ValueError(f"unknown data.gt_source={source!r} (use mask or annos)")
    root = data_cfg.get("annos_root") or str(DEFAULT_ANNOS_ROOT)
    stats = apply_annos_gt(samples, root)
    tag = f" {split}" if split else ""
    print(
        f"[gt]{tag} source=annos replaced={stats['replaced']} "
        f"fallback_mask={stats['fallback_mask']} normal={stats['normal']} root={root}",
        flush=True,
    )
    return samples
