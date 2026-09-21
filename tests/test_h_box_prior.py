import torch
import torch.nn as nn

from models.h_box_prior import (GEOM_DIM, HBoxProjector, _HBoxScatterEmbedding,
                                compute_g_proj, geometry_from_proposals)


def test_geometry_from_proposals_normalizes_and_pads():
    proposals = [
        dict(bbox_2d=[100.0, 200.0, 500.0, 600.0], raw_peak=0.8),
        dict(bbox_2d=[0.0, 0.0, 100.0, 100.0], raw_peak=0.5),
    ]
    K = 3
    g = geometry_from_proposals(proposals, K, h_min=0.0, h_max=1.0)
    assert g.shape == (K, GEOM_DIM)
    x1, y1, x2, y2, cx, cy, w, h, area, peak, valid = [float(v) for v in g[0]]
    assert abs(x1 - 0.1) < 1e-6 and abs(y1 - 0.2) < 1e-6
    assert abs(x2 - 0.5) < 1e-6 and abs(y2 - 0.6) < 1e-6
    assert abs(cx - 0.3) < 1e-6 and abs(cy - 0.4) < 1e-6
    assert abs(w - 0.4) < 1e-6 and abs(h - 0.4) < 1e-6
    assert abs(area - 0.16) < 1e-6
    assert abs(peak - 0.8) < 1e-6
    assert valid == 1.0
    # The padded third row is all-zero with is_valid=0 (an explicit "no box" slot).
    assert float(g[2].sum()) == 0.0
    assert g[2, -1] == 0.0


def test_geometry_peak_normalization():
    proposals = [dict(bbox_2d=[0.0, 0.0, 100.0, 100.0], raw_peak=7.5)]
    g = geometry_from_proposals(proposals, 1, h_min=5.0, h_max=10.0)
    # (7.5 - 5.0) / (10.0 - 5.0) = 0.5
    assert abs(float(g[0, 9]) - 0.5) < 1e-6


def test_projector_shape_and_dtype():
    proj = HBoxProjector(geom_dim=GEOM_DIM, hidden_size=32)
    g = torch.randn(3, GEOM_DIM)
    out = proj(g)
    assert out.shape == (3, 32)
    assert out.dtype == g.dtype


class _FakeEmb(nn.Module):
    def __init__(self, vocab=10, dim=4):
        super().__init__()
        self.w = nn.Embedding(vocab, dim)

    def forward(self, input_ids):
        return self.w(input_ids)


def test_scatter_replaces_box_slots_only():
    base = _FakeEmb()
    g_proj = torch.randn(3, 4)
    box_id = 7
    wrapper = _HBoxScatterEmbedding(base, box_id, g_proj)
    input_ids = torch.tensor([[box_id, box_id, box_id, 1, 2]])
    embeds = wrapper(input_ids)
    assert torch.allclose(embeds[0, :3], g_proj)
    assert torch.allclose(embeds[0, 3], base.w(torch.tensor([1]))[0])
    assert torch.allclose(embeds[0, 4], base.w(torch.tensor([2]))[0])


def test_scatter_tiles_across_batch():
    base = _FakeEmb()
    g_proj = torch.randn(3, 4)
    box_id = 7
    wrapper = _HBoxScatterEmbedding(base, box_id, g_proj)
    input_ids = torch.tensor([[box_id] * 3 + [1], [box_id] * 3 + [2]])
    embeds = wrapper(input_ids)
    assert torch.allclose(embeds[0, :3], g_proj)
    assert torch.allclose(embeds[1, :3], g_proj)
    assert torch.allclose(embeds[0, 3], base.w(torch.tensor([1]))[0])
    assert torch.allclose(embeds[1, 3], base.w(torch.tensor([2]))[0])


def test_compute_g_proj_moves_device_and_keeps_shape():
    proj = HBoxProjector(geom_dim=GEOM_DIM, hidden_size=16)
    geom = torch.randn(3, GEOM_DIM)
    out = compute_g_proj(proj, geom)
    assert out.shape == (3, 16)
