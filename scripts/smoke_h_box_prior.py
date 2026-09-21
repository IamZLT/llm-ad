"""Smoke test for the H-Box geometry-token prior (no full LLM load).

Verifies, in isolation:
  1. region_proposals: H connected-components -> candidate bboxes.
  2. geometry_from_proposals: bboxes -> [K, 11] geometry matrix (norm + pad).
  3. HBoxProjector + compute_g_proj: geometry -> hidden-size token, linear.
  4. _HBoxScatterEmbedding: box slots receive g_proj, others untouched.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from outcome.inputs import region_proposals
from models.h_box_prior import (GEOM_DIM, HBoxProjector, _HBoxScatterEmbedding,
                                compute_g_proj, geometry_from_proposals)


def _check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        sys.exit(1)


def test_region_proposals_cc():
    h = np.zeros((16, 16), dtype=float)
    h[2:6, 2:7] = 0.9    # top-left blob (rows 2..5, cols 2..6)
    h[10:14, 9:15] = 0.7  # bottom-right blob
    cfg = dict(relative_threshold=0.7, raw_threshold=None, max_candidates=3,
               min_cells=1, box_mode='full')
    proposals, masks, mode = region_proposals(torch.from_numpy(h), cfg)
    _check("2 connected components", len(proposals) == 2)
    _check("masks shape [R,H,W]", masks.shape == (2, 16, 16))
    b0 = proposals[0]['bbox_2d']  # top-left, highest peak
    _check("bbox x1=125", abs(b0[0] - 125.0) < 0.01)
    _check("bbox y1=125", abs(b0[1] - 125.0) < 0.01)
    _check("bbox x2=437.5", abs(b0[2] - 437.5) < 0.01)
    _check("bbox y2=375", abs(b0[3] - 375.0) < 0.01)
    _check("peak_2d present", len(proposals[0]['peak_2d']) == 2)


def test_flat_h_yields_no_proposals():
    cfg = dict(relative_threshold=0.7, raw_threshold=None, max_candidates=3,
               min_cells=1, box_mode='full')
    proposals, masks, _ = region_proposals(torch.zeros((8, 8)), cfg)
    _check("flat H -> 0 proposals", len(proposals) == 0 and masks.shape[0] == 0)


def test_geometry_integration():
    h = np.zeros((16, 16), dtype=float)
    h[2:6, 2:7] = 0.9
    h[10:14, 9:15] = 0.7
    cfg = dict(relative_threshold=0.7, raw_threshold=None, max_candidates=3,
               min_cells=1, box_mode='full')
    proposals, _, _ = region_proposals(torch.from_numpy(h), cfg)
    K = 3
    g = geometry_from_proposals(proposals, K, h_min=0.0, h_max=1.0)
    _check("geometry [K, 11]", g.shape == (K, GEOM_DIM))
    _check("row0 valid", g[0, -1] == 1.0)
    _check("row2 empty", g[2, -1] == 0.0 and float(g[2].sum()) == 0.0)
    # projected via a real HBoxProjector
    proj = HBoxProjector(geom_dim=GEOM_DIM, hidden_size=64)
    out = compute_g_proj(proj, torch.from_numpy(g))
    _check("g_proj [K, hidden]", out.shape == (K, 64))


def test_scatter():
    base = torch.nn.Embedding(10, 4)
    g_proj = torch.randn(3, 4)
    wrapper = _HBoxScatterEmbedding(base, 7, g_proj)
    ids = torch.tensor([[7, 7, 7, 1]])
    emb = wrapper(ids)
    _check("box slots == g_proj", torch.allclose(emb[0, :3], g_proj))
    _check("other slot untouched", torch.allclose(emb[0, 3], base(torch.tensor([1]))[0]))


if __name__ == '__main__':
    test_region_proposals_cc()
    test_flat_h_yields_no_proposals()
    test_geometry_integration()
    test_scatter()
    print("\nAll H-Box geometry-token smoke tests passed.")
