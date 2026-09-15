"""ANNOS / MMR-AD boxes as optional detection GT."""
from __future__ import annotations

import json

from PIL import Image

from data.annos_gt import apply_annos_gt, paper_to_px, union_boxes


def test_short_edge_448_maps_full_canvas_to_original():
    # 1500×1000 → canvas 672×448; the full canvas must cover the whole image.
    px = paper_to_px([0, 0, 672, 448], (1500, 1000))
    assert [round(v) for v in px] == [0, 0, 1500, 1000]


def test_square_image_matches_plain_448_scale():
    px = paper_to_px([112, 112, 336, 336], (900, 900))
    assert [round(v) for v in px] == [225, 225, 675, 675]


def test_apply_annos_gt_replaces_mask_boxes(tmp_path):
    img = tmp_path / "000.png"
    Image.new("RGB", (1500, 1000), (0, 0, 0)).save(img)
    cls_dir = tmp_path / "visa" / "candle"
    cls_dir.mkdir(parents=True)
    (cls_dir / "bbox_annos.json").write_text(json.dumps([
        dict(name="anomaly-000.png", bboxes=[dict(bbox_2d=[0, 0, 100, 50], label="x")]),
    ]))
    sample = dict(
        image=str(img),
        full_img_path=str(img),
        metadata={
            "class": "candle",
            "anomaly": True,
            "defect_type": "anomaly",
            "layout": "visa",
            "component_bboxes": [[10, 10, 20, 20], [30, 30, 40, 40]],
            "bbox": [10, 10, 40, 40],
            "num_components": 2,
        },
    )
    stats = apply_annos_gt([sample], tmp_path)
    assert stats["replaced"] == 1
    assert sample["metadata"]["gt_source"] == "annos"
    assert sample["metadata"]["num_components"] == 1
    box = sample["metadata"]["component_bboxes"][0]
    # short=1000 → scale 448/1000; x_px = 100 * 1000/448
    assert abs(box[2] - 100 * 1000 / 448) < 1e-6
    assert abs(box[3] - 50 * 1000 / 448) < 1e-6
    assert union_boxes(sample["metadata"]["component_bboxes"]) == sample["metadata"]["bbox"]


def test_apply_annos_gt_keeps_mask_when_missing(tmp_path):
    img = tmp_path / "999.png"
    Image.new("RGB", (64, 64), (0, 0, 0)).save(img)
    (tmp_path / "visa" / "candle").mkdir(parents=True)
    (tmp_path / "visa" / "candle" / "bbox_annos.json").write_text("[]")
    sample = dict(
        image=str(img),
        full_img_path=str(img),
        metadata={
            "class": "candle",
            "anomaly": True,
            "defect_type": "anomaly",
            "layout": "visa",
            "component_bboxes": [[1, 1, 8, 8]],
            "num_components": 1,
            "bbox": [1, 1, 8, 8],
        },
    )
    stats = apply_annos_gt([sample], tmp_path)
    assert stats["fallback_mask"] == 1
    assert sample["metadata"]["component_bboxes"] == [[1, 1, 8, 8]]
    assert sample["metadata"]["gt_source"] == "mask"
