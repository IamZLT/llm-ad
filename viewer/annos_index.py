"""Map ANNOS paper boxes onto local MVTec / VisA images."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from data.annos_gt import PAPER_CANVAS, paper_canvas_size, paper_to_px

ROOT = Path(__file__).resolve().parents[1]
ANNOS = ROOT / "datasets" / "ANNOS"
DATASETS = ROOT / "datasets"

SOURCE_MAP = {
    "mvtec": {
        "annos": "mvtec",
        "images": DATASETS / "mvtec_anomaly_detection",
        "layout": "mvtec",
        "label": "MVTec AD",
    },
    "visa": {
        "annos": "visa",
        "images": DATASETS / "VisA",
        "layout": "visa",
        "label": "VisA",
    },
}

_IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".JPG", ".PNG", ".JPEG")


def parse_annos_name(name: str) -> tuple[str, str, str]:
    """`broken_large-000.png` -> (defect, `000.png`, `000`)."""
    stem_full = Path(name).name
    if "-" not in stem_full:
        stem = Path(stem_full).stem
        return "unknown", stem_full, stem
    defect, rest = stem_full.rsplit("-", 1)
    return defect, rest, Path(rest).stem


def _first_existing(cands) -> Path | None:
    for p in cands:
        if p.is_file():
            return p
    return None


def resolve_test_image(source: str, cls: str, name: str) -> Path | None:
    if source not in SOURCE_MAP:
        return None
    meta = SOURCE_MAP[source]
    root = Path(meta["images"])
    defect, fname, stem = parse_annos_name(name)
    if meta["layout"] == "mvtec":
        return _first_existing(
            [root / cls / "test" / defect / fname, root / cls / "test" / defect / f"{stem}.png"]
        )
    # VisA anomaly files are 000.JPG (ANNOS uses anomaly-000.png).
    folder = root / cls / "Data" / "Images" / "Anomaly"
    cands = [folder / f"{stem}{ext}" for ext in _IMG_EXT]
    cands.append(folder / fname)
    return _first_existing(cands)


def resolve_mask(source: str, cls: str, name: str) -> Path | None:
    if source not in SOURCE_MAP:
        return None
    meta = SOURCE_MAP[source]
    root = Path(meta["images"])
    defect, fname, stem = parse_annos_name(name)
    if meta["layout"] == "mvtec":
        gt = root / cls / "ground_truth" / defect
        return _first_existing(
            [gt / f"{stem}_mask.png", gt / f"{stem}.png", gt / fname]
        )
    gt = root / cls / "Data" / "Masks" / "Anomaly"
    return _first_existing([gt / f"{stem}.png", gt / f"{stem}_mask.png"] + [gt / f"{stem}{e}" for e in _IMG_EXT])


def resolve_ref(source: str, cls: str, rel: str) -> Path | None:
    meta = SOURCE_MAP[source]
    root = Path(meta["images"])
    rel = rel.lstrip("/")
    # reference.json paths already include the class prefix.
    direct = root / rel
    if direct.is_file():
        return direct
    parts = Path(rel).parts
    if parts and parts[0] == cls:
        p = root / Path(*parts[1:])
        if p.is_file():
            return p
    return _first_existing([root / cls / rel, root / rel])


@lru_cache(maxsize=64)
def load_json(path: str):
    p = Path(path)
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def _annos_root(source: str) -> Path:
    if source in SOURCE_MAP:
        return ANNOS / SOURCE_MAP[source]["annos"]
    return ANNOS / source


def list_sources() -> list[dict]:
    out = []
    for key, meta in SOURCE_MAP.items():
        annos_dir = ANNOS / meta["annos"]
        img_root = Path(meta["images"])
        classes = sorted(p.name for p in annos_dir.iterdir() if p.is_dir()) if annos_dir.is_dir() else []
        out.append(
            dict(
                id=key,
                label=meta["label"],
                has_annos=annos_dir.is_dir(),
                has_images=img_root.is_dir(),
                n_classes=len(classes),
                classes=classes,
            )
        )
    if ANNOS.is_dir():
        known = {m["annos"] for m in SOURCE_MAP.values()}
        for p in sorted(ANNOS.iterdir()):
            if not p.is_dir() or p.name in known:
                continue
            classes = sorted(c.name for c in p.iterdir() if c.is_dir() and (c / "bbox_annos.json").exists())
            out.append(
                dict(
                    id=p.name,
                    label=p.name,
                    has_annos=True,
                    has_images=False,
                    n_classes=len(classes),
                    classes=classes,
                )
            )
    return out


def list_classes(source: str) -> list[str]:
    annos_dir = _annos_root(source)
    if not annos_dir.is_dir():
        return []
    return sorted(p.name for p in annos_dir.iterdir() if p.is_dir() and (p / "bbox_annos.json").exists())


def list_local_classes(source: str) -> list[str]:
    if source not in SOURCE_MAP:
        return []
    root = Path(SOURCE_MAP[source]["images"])
    if not root.is_dir():
        return []
    skip = {"LICENSE-DATASET", "Data"}
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and p.name not in skip and not p.name.startswith(".")
    )


def _annos_entries(source: str, cls: str) -> list[dict]:
    path = _annos_root(source) / cls / "bbox_annos.json"
    data = load_json(str(path)) or []
    return data if isinstance(data, list) else []


def _text_index(source: str, cls: str) -> dict:
    path = _annos_root(source) / cls / "text_annos.json"
    data = load_json(str(path)) or []
    if not isinstance(data, list):
        return {}
    return {e.get("name"): e for e in data if isinstance(e, dict) and e.get("name")}


def _ref_index(source: str, cls: str) -> dict:
    path = _annos_root(source) / cls / "reference.json"
    data = load_json(str(path)) or {}
    return data if isinstance(data, dict) else {}


def list_samples(source: str, cls: str) -> list[dict]:
    rows = []
    for e in _annos_entries(source, cls):
        name = e.get("name")
        if not name:
            continue
        defect, _, stem = parse_annos_name(name)
        img = resolve_test_image(source, cls, name)
        rows.append(
            dict(
                name=name,
                defect=defect,
                stem=stem,
                n_boxes=len(e.get("bboxes") or []),
                has_image=img is not None,
                labels=[b.get("label") or "" for b in (e.get("bboxes") or [])],
            )
        )
    return rows


def list_local_samples(source: str, cls: str) -> list[dict]:
    """All test images on disk, annotated when ANNOS has a matching name."""
    if source not in SOURCE_MAP:
        return []
    meta = SOURCE_MAP[source]
    root = Path(meta["images"])
    annos = {e["name"]: e for e in _annos_entries(source, cls)}
    rows = []
    if meta["layout"] == "mvtec":
        test = root / cls / "test"
        if not test.is_dir():
            return []
        for defect_dir in sorted(test.iterdir()):
            if not defect_dir.is_dir():
                continue
            defect = defect_dir.name
            for img in sorted(defect_dir.iterdir()):
                if img.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                    continue
                stem = img.stem
                name = f"{defect}-{stem}.png"
                e = annos.get(name)
                rows.append(
                    dict(
                        name=name,
                        defect=defect,
                        stem=stem,
                        n_boxes=len((e or {}).get("bboxes") or []),
                        has_image=True,
                        has_paper=e is not None,
                        labels=[b.get("label") or "" for b in ((e or {}).get("bboxes") or [])],
                    )
                )
        return rows
    images_root = root / cls / "Data" / "Images"
    for defect, folder_name in (("anomaly", "Anomaly"), ("normal", "Normal")):
        folder = images_root / folder_name
        if not folder.is_dir():
            continue
        for img in sorted(folder.iterdir()):
            if img.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                continue
            stem = img.stem
            name = f"{defect}-{stem}.png"
            e = annos.get(name) or annos.get(f"anomaly-{stem.zfill(3)}.png")
            rows.append(
                dict(
                    name=name,
                    defect=defect,
                    stem=stem,
                    n_boxes=len((e or {}).get("bboxes") or []),
                    has_image=True,
                    has_paper=e is not None,
                    labels=[b.get("label") or "" for b in ((e or {}).get("bboxes") or [])],
                )
            )
    return rows


def _default_refs(source: str, cls: str) -> list[Path]:
    meta = SOURCE_MAP[source]
    root = Path(meta["images"])
    if meta["layout"] == "mvtec":
        d = root / cls / "train" / "good"
    else:
        d = root / cls / "Data" / "Images" / "Normal"
    if not d.is_dir():
        return []
    files = [p for p in sorted(d.iterdir()) if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
    return files[:24]


def _paper_boxes(entry: dict, size) -> list[dict]:
    paper = []
    rw, rh = entry.get("resized_width"), entry.get("resized_height")
    for b in entry.get("bboxes") or []:
        box = [float(v) for v in b["bbox_2d"]]
        paper.append(
            dict(
                bbox_448=box,
                bbox_px=paper_to_px(box, size, resized_width=rw, resized_height=rh),
                label=b.get("label") or "",
            )
        )
    return paper


def sample_detail(source: str, cls: str, name: str) -> dict:
    from PIL import Image as PILImage

    from data.scan import extract_targets_from_mask

    entries = {e["name"]: e for e in _annos_entries(source, cls)}
    entry = entries.get(name) or {}
    text_e = _text_index(source, cls).get(name) or {}
    defect, fname, stem = parse_annos_name(name)
    img = None
    if source in SOURCE_MAP:
        img = resolve_test_image(source, cls, name)
        if img is None:
            meta = SOURCE_MAP[source]
            root = Path(meta["images"])
            if meta["layout"] == "mvtec":
                img = _first_existing(
                    [root / cls / "test" / defect / fname, root / cls / "test" / defect / f"{stem}.png"]
                )
            else:
                folder = "Normal" if defect.lower() in ("normal", "good", "ok") else "Anomaly"
                base = root / cls / "Data" / "Images" / folder
                img = _first_existing([base / f"{stem}{ext}" for ext in _IMG_EXT] + [base / fname])
    if img is None or not img.is_file():
        if source not in SOURCE_MAP:
            size = [int(entry.get("resized_width") or PAPER_CANVAS), int(entry.get("resized_height") or PAPER_CANVAS)]
            paper = _paper_boxes(entry, size)
            return dict(
                source=source,
                class_name=cls,
                name=name,
                defect=defect,
                stem=stem,
                image=None,
                mask=None,
                size=list(size),
                paper=paper,
                gt_boxes_px=[],
                refs=[],
                paper_text=text_e.get("text") or "",
                paper_label=(paper[0]["label"] if paper else ""),
                canvas=PAPER_CANVAS,
                canvas_size=list(paper_canvas_size(size, entry.get("resized_width"), entry.get("resized_height"))),
                is_anomaly=defect.lower() not in ("good", "ok", "normal"),
                has_image=False,
            )
        raise FileNotFoundError(f"image not found for {source}/{cls}/{name}")

    with PILImage.open(img) as im:
        size = im.size
    paper = _paper_boxes(entry, size)

    mask = resolve_mask(source, cls, name)
    gt = []
    if mask is not None:
        targets = extract_targets_from_mask(str(mask), min_contour_area=5, merge_kernel_ratio=0.01)
        if targets:
            gt = [list(map(float, bb)) for bb in targets.get("component_bboxes") or []]

    refs = []
    ref_map = _ref_index(source, cls)
    key_candidates = []
    if SOURCE_MAP[source]["layout"] == "mvtec":
        key_candidates.append(f"{cls}/test/{defect}/{stem}.png")
        key_candidates.append(f"{cls}/test/{defect}/{fname}")
    else:
        key_candidates.append(f"{cls}/Data/Images/Anomaly/{stem}.JPG")
        key_candidates.append(f"{cls}/Data/Images/Anomaly/{stem}.jpg")
        key_candidates.append(f"{cls}/Data/Images/Anomaly/{stem}.png")
    rels = []
    for k in key_candidates:
        if k in ref_map:
            rels = list(ref_map[k])
            break
    if not rels and text_e.get("ref-name"):
        rn = str(text_e["ref-name"])
        if SOURCE_MAP[source]["layout"] == "visa":
            rels = [f"{cls}/Data/Images/Normal/{Path(rn).stem}.JPG"]
        else:
            rels = [f"{cls}/train/good/{rn}"]
    for rel in rels[:10]:
        p = resolve_ref(source, cls, rel)
        if p is not None:
            refs.append(str(p))
    if not refs:
        refs = [str(p) for p in _default_refs(source, cls)[:16]]

    return dict(
        source=source,
        class_name=cls,
        name=name,
        defect=defect,
        stem=stem,
        image=str(img),
        mask=str(mask) if mask else None,
        size=list(size),
        paper=paper,
        gt_boxes_px=gt,
        refs=refs,
        paper_text=text_e.get("text") or "",
        paper_label=(paper[0]["label"] if paper else ""),
        canvas=PAPER_CANVAS,
        canvas_size=list(paper_canvas_size(size, entry.get("resized_width"), entry.get("resized_height"))),
        is_anomaly=defect.lower() not in ("good", "ok", "normal"),
        has_image=True,
    )
