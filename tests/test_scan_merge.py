"""Tests for AD-FM-style fragment merging in data.scan.extract_targets_from_mask."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from data.scan import extract_targets_from_mask


def _write_mask(path, blobs, size=(200, 200)):
    arr = np.zeros(size, dtype=np.uint8)
    for (y1, x1, y2, x2) in blobs:
        arr[y1:y2, x1:x2] = 255
    Image.fromarray(arr).save(path)
    return str(path)


def test_no_merge_keeps_nearby_components_separate(tmp_path):
    # 8px gap between two 10x10 blobs
    p = _write_mask(tmp_path / 'm.png', [(50, 50, 60, 60), (50, 68, 60, 78)])
    out = extract_targets_from_mask(p, min_contour_area=5, merge_kernel_ratio=0.0)
    assert out['num_components'] == 2
    assert len(out['component_bboxes']) == 2


def test_merge_ratio_merges_nearby_components(tmp_path):
    p = _write_mask(tmp_path / 'm.png', [(50, 50, 60, 60), (50, 68, 60, 78)])
    # min(H,W)=200, ratio=0.05 -> k=11, bridges the 8px gap
    out = extract_targets_from_mask(p, min_contour_area=5, merge_kernel_ratio=0.05)
    assert out['num_components'] == 1
    box = out['component_bboxes'][0]
    # merged box covers both blobs (and is padded by dilation)
    assert box[0] <= 50 and box[1] <= 50 and box[2] >= 78 and box[3] >= 60


def test_small_kernel_does_not_merge(tmp_path):
    p = _write_mask(tmp_path / 'm.png', [(50, 50, 60, 60), (50, 68, 60, 78)])
    # ratio=0.01 -> k=3, bridges at most ~2px; 8px gap survives
    out = extract_targets_from_mask(p, min_contour_area=5, merge_kernel_ratio=0.01)
    assert out['num_components'] == 2


def test_distant_components_never_merge(tmp_path):
    p = _write_mask(tmp_path / 'm.png', [(20, 20, 30, 30), (150, 150, 160, 160)])
    out = extract_targets_from_mask(p, min_contour_area=5, merge_kernel_ratio=0.05)
    assert out['num_components'] == 2


def test_denoise_before_dilation(tmp_path):
    # 2x2 speck (area 4 < min_contour_area=5) far from the real defect
    p = _write_mask(tmp_path / 'm.png', [(100, 100, 110, 110), (10, 10, 12, 12)])
    out = extract_targets_from_mask(p, min_contour_area=5, merge_kernel_ratio=0.05)
    assert out['num_components'] == 1
    box = out['component_bboxes'][0]
    # the surviving box is around the real defect, not the amplified speck
    assert box[0] >= 95 and box[1] >= 95


def test_union_box_always_from_raw_mask(tmp_path):
    p = _write_mask(tmp_path / 'm.png', [(50, 50, 60, 60), (50, 68, 60, 78)])
    raw = extract_targets_from_mask(p, min_contour_area=5, merge_kernel_ratio=0.0)
    merged = extract_targets_from_mask(p, min_contour_area=5, merge_kernel_ratio=0.05)
    assert merged['bbox'] == raw['bbox'] == [50, 50, 78, 60]


def test_empty_and_missing_mask(tmp_path):
    assert extract_targets_from_mask(str(tmp_path / 'nope.png')) is None
    p = _write_mask(tmp_path / 'empty.png', [])
    assert extract_targets_from_mask(p, merge_kernel_ratio=0.05) is None
