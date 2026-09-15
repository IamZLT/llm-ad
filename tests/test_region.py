"""Unit tests for the H-guided region contrast token stack (no real model needed)."""
import numpy as np
import pytest
import torch

from models.anomaly_prior import AnomalyPrior
from models.region_adapter import RegionAdapter
from models.region_injection import (
    _RegionScatterEmbedding,
    load_region_adapter,
    save_region_adapter,
)
from outcome.inputs import extract_region_cells, region_proposals


# --------------------------------------------------------------------------- #
# _nn_match
# --------------------------------------------------------------------------- #
def test_nn_match_global_returns_distance_features_and_coords():
    # ref patches: two orthogonal vectors, plus the test patch equal to ref[1].
    ref = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    test = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    out = AnomalyPrior._nn_match(None, test, ref, (2, 1), (2, 1), 0)
    assert out['distance'].shape == (2, 1)
    assert out['matched_ref_features'].shape == (2, 2)
    assert out['match_coordinates'].shape == (2, 2)
    # test[0] == ref[1] -> match coordinate (y=1, x=0)
    assert out['match_coordinates'][0].tolist() == [1, 0]
    # test[1] == ref[0] -> match coordinate (y=0, x=0)
    assert out['match_coordinates'][1].tolist() == [0, 0]
    assert torch.allclose(out['matched_ref_features'][0], ref[1])


def test_nn_match_refuses_to_guess_grid_size():
    ref = torch.zeros(3, 4)
    test = torch.zeros(5, 4)
    with pytest.raises(ValueError):
        AnomalyPrior._nn_match(None, test, ref, (2, 2), (2, 2), 0)


# --------------------------------------------------------------------------- #
# region_proposals masks
# --------------------------------------------------------------------------- #
def test_region_proposals_masks_track_connected_cells():
    h = np.zeros((4, 4))
    h[0, 0] = 1.0
    h[3, 3] = 0.9
    meta, masks, mode = region_proposals(h, {'max_candidates': 2})
    assert len(meta) == 2
    assert masks.shape == (2, 4, 4)
    assert bool(masks[0][0, 0])
    assert bool(masks[1][3, 3])


# --------------------------------------------------------------------------- #
# extract_region_cells
# --------------------------------------------------------------------------- #
def test_extract_region_cells_empty_returns_single_invalid_cell():
    D = 4
    test_f = torch.zeros(4, D)
    ref_f = torch.zeros(4, D)
    hmap = torch.zeros(2, 2)
    out = extract_region_cells(np.zeros((0, 2, 2), dtype=bool), test_f, ref_f, hmap, {})
    assert out['test'].shape == (1, 1, D)
    assert out['valid'].tolist() == [[False]]


def test_extract_region_cells_full_region_splits_into_2x2():
    Ht, Wt, D = 2, 2, 3
    test_f = torch.arange(Ht * Wt * D, dtype=torch.float32).reshape(Ht * Wt, D)
    ref_f = torch.zeros_like(test_f)
    hmap = torch.tensor([[0.2, 0.8], [0.9, 0.3]])
    masks = np.ones((1, Ht, Wt), dtype=bool)
    out = extract_region_cells(masks, test_f, ref_f, hmap, {'token_mode': 'grid'})
    assert out['test'].shape == (1, 4, D)
    assert out['geom'].shape == (1, 4, 5)
    assert out['hstat'].shape == (1, 4, 2)
    assert out['valid'].tolist() == [[True, True, True, True]]


def test_extract_region_cells_peak_mode_one_token_at_argmax():
    Ht, Wt, D = 2, 2, 3
    test_f = torch.arange(Ht * Wt * D, dtype=torch.float32).reshape(Ht * Wt, D)
    ref_f = torch.zeros_like(test_f)
    hmap = torch.tensor([[0.2, 0.8], [0.9, 0.3]])
    masks = np.ones((1, Ht, Wt), dtype=bool)
    out = extract_region_cells(masks, test_f, ref_f, hmap, {'token_mode': 'peak'})
    assert out['test'].shape == (1, 1, D)
    # peak is (1,0) -> flattened index 2
    assert torch.allclose(out['test'][0, 0], test_f[2])
    assert out['valid'].tolist() == [[True]]
    assert out['owner'].tolist() == [[0]]


def test_extract_region_cells_hybrid_peak_then_grid():
    Ht, Wt, D = 2, 2, 3
    test_f = torch.arange(Ht * Wt * D, dtype=torch.float32).reshape(Ht * Wt, D)
    ref_f = torch.zeros_like(test_f)
    hmap = torch.tensor([[0.2, 0.8], [0.9, 0.3]])
    masks = np.ones((1, Ht, Wt), dtype=bool)
    out = extract_region_cells(masks, test_f, ref_f, hmap, {'token_mode': 'hybrid'})
    assert out['test'].shape == (1, 5, D)
    assert torch.allclose(out['test'][0, 0], test_f[2])
    assert out['owner'].tolist() == [[0, 0, 0, 0, 0]]
    grid = extract_region_cells(masks, test_f, ref_f, hmap, {'token_mode': 'grid'})
    assert torch.allclose(out['test'][0, 1:], grid['test'][0])


# --------------------------------------------------------------------------- #
# RegionAdapter
# --------------------------------------------------------------------------- #
def test_region_adapter_shape_and_empty_mask():
    B, n, D = 1, 3, 8
    adapter = RegionAdapter(feature_dim=D, hidden_size=16, intermediate_dim=8)
    raw = dict(test=torch.randn(B, n, D), ref=torch.randn(B, n, D),
               geom=torch.randn(B, n, 5), hstat=torch.randn(B, n, 2),
               valid=torch.ones(B, n, dtype=torch.bool))
    out = adapter(raw)
    assert out.shape == (B, n, 16)
    raw['valid'] = torch.zeros(B, n, dtype=torch.bool)
    out_empty = adapter(raw)
    assert torch.allclose(out_empty[0, 0], adapter.empty)


def test_region_adapter_save_load_roundtrip(tmp_path):
    adapter = RegionAdapter(feature_dim=8, hidden_size=16, intermediate_dim=12)
    path = tmp_path / 'region_adapter.pt'
    save_region_adapter(adapter, path)
    loaded = load_region_adapter(RegionAdapter, path, feature_dim=8, hidden_size=16)
    raw = dict(test=torch.randn(1, 2, 8), ref=torch.randn(1, 2, 8),
               geom=torch.randn(1, 2, 5), hstat=torch.randn(1, 2, 2),
               valid=torch.ones(1, 2, dtype=torch.bool))
    assert torch.allclose(adapter(raw), loaded(raw))


# --------------------------------------------------------------------------- #
# H coverage diagnostic helper
# --------------------------------------------------------------------------- #
def test_box_union_coverage():
    from outcome.protocol import box_union_coverage
    gt = [100, 100, 200, 200]
    assert box_union_coverage(gt, [[100, 100, 200, 200]]) == 1.0
    assert box_union_coverage(gt, [[100, 100, 150, 200]]) == 0.5
    assert box_union_coverage(gt, [[100, 100, 150, 200], [150, 100, 200, 200]]) == 1.0
    assert box_union_coverage(gt, [[0, 0, 50, 50]]) == 0.0
    assert box_union_coverage(None, [[0, 0, 10, 10]]) is None


# --------------------------------------------------------------------------- #
# region injection scatter
# --------------------------------------------------------------------------- #
class _Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(10, 4)

    def forward(self, input_ids):
        return self.emb(input_ids)


def test_region_scatter_embedding_replaces_placeholder_positions():
    base = _Emb()
    region_embeds = torch.tensor([[1.0, 1, 1, 1], [2.0, 2, 2, 2]])
    wrapper = _RegionScatterEmbedding(base, region_token_id=9, region_embeds=region_embeds)
    ids = torch.tensor([[1, 9, 2, 9]])
    embeds = wrapper(ids)
    assert embeds.shape == (1, 4, 4)
    assert torch.allclose(embeds[0, 1], region_embeds[0])
    assert torch.allclose(embeds[0, 3], region_embeds[1])
    assert torch.allclose(embeds[0, 0], base.emb(torch.tensor(1)))
    assert torch.allclose(embeds[0, 2], base.emb(torch.tensor(2)))


# --------------------------------------------------------------------------- #
# SFT target construction
# --------------------------------------------------------------------------- #
def test_box_where_bins():
    from train_region_sft import _box_where, _where_join

    assert _box_where([50, 50, 150, 150]) == 'in the upper left'
    assert _box_where([800, 800, 950, 950]) == 'in the lower right'
    assert _box_where([400, 400, 600, 600]) == 'near the center'
    assert _box_where([50, 50, 950, 950]) == 'across a large area'
    assert _where_join([[50, 50, 150, 150], [800, 800, 950, 950]]) == (
        'in the upper left and in the lower right')


def test_sft_target_is_parseable_with_gt_category_and_bbox():
    from train_region_sft import build_sft_target
    from outcome.protocol import parse_output, score_output

    anomaly_meta = dict(is_anomaly=True, orig_size=[500, 500], class_name='bottle',
                        gt_box_px=[100, 150, 200, 250], defect_type='scratch')
    target = build_sft_target(anomaly_meta)
    parsed = parse_output(target)
    assert parsed['task_valid'] and parsed['is_anomaly'] is True
    assert parsed['bbox_2d'] == [200.0, 300.0, 400.0, 500.0]
    assert parsed['candidate_bbox_2d'] == [200.0, 300.0, 400.0, 500.0]
    assert score_output(parsed, anomaly_meta)['task'] == 1.0
    # description summarizes the reasoning chain, not a bare label; the coarse folder
    # defect_type must NOT leak into the text (it is never given in the prompt)
    assert 'bottle' in parsed['description']
    assert 'cannot explain' in parsed['description']
    understand = parsed['tags']['understand']
    compare = parsed['tags']['compare']
    assert understand.startswith('Image 1') and 'Image 2' in understand
    assert 'coordinates' in understand
    assert 'true defect' in compare and 'normal variation' in compare
    assert 'on the left' not in understand and 'on the left' not in compare
    assert 'on the left' in parsed['tags']['ground']
    assert 'on the left' in parsed['description']
    assert 'on the left' not in parsed['verify_evidence']
    assert 'scratch' not in target
    assert parsed['verify_action'] == 'keep'

    normal_meta = dict(is_anomaly=False, orig_size=[500, 500], class_name='bottle',
                       gt_box_px=None, defect_type='good')
    parsed_normal = parse_output(build_sft_target(normal_meta))
    assert parsed_normal['task_valid'] and parsed_normal['is_anomaly'] is False
    assert parsed_normal['bbox_2d'] is None
    assert parsed_normal['verify_action'] == 'none'
    assert 'consistent' in parsed_normal['description']
    assert 'good' not in parsed_normal['description']
    assert parsed_normal['tags']['understand'].startswith('Image 1')


def test_sft_target_multibox_parseable_and_descriptive():
    from train_region_sft import build_sft_target
    from outcome.protocol_multibox import parse_output as parse_mb, score_output as score_mb

    meta = dict(is_anomaly=True, orig_size=[1000, 1000], class_name='metal_nut',
                defect_type='broken_teeth', gt_box_px=[100, 100, 500, 500],
                component_bboxes=[[100, 100, 200, 200], [300, 300, 500, 500]])
    target = build_sft_target(meta, multibox=True)
    parsed = parse_mb(target)
    assert parsed['task_valid'] and parsed['is_anomaly'] is True
    assert parsed['bboxes_2d'] == [[100.0, 100.0, 200.0, 200.0], [300.0, 300.0, 500.0, 500.0]]
    # no H hints in meta -> empty candidate + discover verb (ground reads out hints,
    # it no longer copies the final GT boxes)
    assert parsed['candidate_bboxes_2d'] == []
    assert parsed['verify_action'] == 'discover'
    assert parsed['protocol_strict']
    assert score_mb(parsed, meta)['task'] > 0.0
    assert 'metal nut' in parsed['description']
    assert 'cannot explain' in parsed['description']
    assert 'broken' not in target and 'teeth' not in target
    assert '2 regions' in target
    assert parsed['tags']['understand'].startswith('Image 1')
    assert 'upper left' not in parsed['tags']['compare']
    assert 'upper left' in parsed['description']
    assert 'upper left' not in parsed['verify_evidence']

    normal = dict(is_anomaly=False, orig_size=[1000, 1000], class_name='metal_nut',
                  defect_type='good', gt_box_px=None, component_bboxes=None)
    parsed_n = parse_mb(build_sft_target(normal, multibox=True))
    assert parsed_n['protocol_strict'] and parsed_n['is_anomaly'] is False
    assert parsed_n['bboxes_2d'] == [] and parsed_n['verify_action'] == 'none'


def _mb_meta(cands, **kw):
    meta = dict(is_anomaly=True, orig_size=[1000, 1000], class_name='metal_nut',
                defect_type='broken_teeth', gt_box_px=[100, 100, 500, 500],
                component_bboxes=[[100, 100, 200, 200], [300, 300, 500, 500]],
                prior_candidates=[dict(bbox_2d=b) for b in cands])
    meta.update(kw)
    return meta


def test_sft_target_multibox_ground_reads_out_h_hints():
    from train_region_sft import build_sft_target
    from outcome.protocol_multibox import parse_output as parse_mb

    # candidates exactly on both GT components -> keep
    meta = _mb_meta([[100, 100, 200, 200], [300, 300, 500, 500]])
    parsed = parse_mb(build_sft_target(meta, multibox=True))
    assert parsed['candidate_bboxes_2d'] == [[100.0, 100.0, 200.0, 200.0], [300.0, 300.0, 500.0, 500.0]]
    assert parsed['verify_action'] == 'keep'
    assert parsed['protocol_strict']
    assert 'upper left' in parsed['tags']['ground']

    # loose candidates hitting both components below the tight bar -> refine
    meta = _mb_meta([[110, 110, 150, 150], [320, 320, 400, 400]])
    parsed = parse_mb(build_sft_target(meta, multibox=True))
    assert parsed['verify_action'] == 'refine'
    assert 'tightened' in parsed['verify_evidence']

    # point-like candidates strictly inside the GT comps (IoU < 0.1) still count
    # as hits via the center rule -> refine, NOT discover
    meta = _mb_meta([[137, 137, 163, 163], [390, 390, 410, 410]])
    parsed = parse_mb(build_sft_target(meta, multibox=True))
    assert parsed['verify_action'] == 'refine'
    assert 'tightened' in parsed['verify_evidence']

    # one spurious candidate on top of tight ones -> refine + dropped
    meta = _mb_meta([[100, 100, 200, 200], [300, 300, 500, 500], [700, 700, 760, 760]])
    parsed = parse_mb(build_sft_target(meta, multibox=True))
    assert parsed['verify_action'] == 'refine'
    assert 'dropped' in parsed['verify_evidence']

    # second component not covered by any candidate -> discover
    meta = _mb_meta([[100, 100, 200, 200]])
    parsed = parse_mb(build_sft_target(meta, multibox=True))
    assert parsed['verify_action'] == 'discover'
    assert 'beyond the marked candidates' in parsed['verify_evidence']


def test_sft_target_multibox_normal_localize_then_reject():
    from train_region_sft import build_sft_target
    from outcome.protocol_multibox import parse_output as parse_mb, score_output as score_mb

    normal = dict(is_anomaly=False, orig_size=[1000, 1000], class_name='metal_nut',
                  defect_type='good', gt_box_px=None, component_bboxes=None,
                  prior_candidates=[dict(bbox_2d=[110, 110, 150, 150]), dict(bbox_2d=[320, 320, 400, 400])])
    target = build_sft_target(normal, multibox=True)
    parsed = parse_mb(target)
    assert parsed['protocol_strict'] and parsed['is_anomaly'] is False
    # the normal target now localizes the H hints and rejects them in verify
    assert parsed['candidate_bboxes_2d'] == [[110.0, 110.0, 150.0, 150.0], [320.0, 320.0, 400.0, 400.0]]
    assert parsed['verify_action'] == 'reject'
    assert parsed['bboxes_2d'] == []
    # and the reward scheme pays the focus bonus for this trajectory
    assert score_mb(parsed, normal)['task'] == 1.0 + 0.2 * 0.5


def test_build_labels_masks_prompt_and_supervises_whole_target():
    from train_region_sft import build_labels
    prompt_ids = [0, 0, 0]
    target_ids = [11, 12, 13, 14]
    labels = build_labels(prompt_ids, target_ids)
    assert labels[:len(prompt_ids)] == [-100] * len(prompt_ids)
    assert labels[len(prompt_ids):] == target_ids


def test_default_batch_size_scales_with_arch():
    from train_region_sft import default_batch_size
    assert default_batch_size('Qwen3.5-2B') == 8
    assert default_batch_size('Qwen3.5-4B') == 4
    assert default_batch_size('Qwen3.5-9B') == 2


def test_shard_indices_is_multiple_of_batch_and_rank_disjoint():
    from train_region_sft import shard_indices
    device = torch.device('cpu')
    a = shard_indices(21, epoch=1, seed=42, rank=0, world=2, batch_size=4, device=device)
    b = shard_indices(21, epoch=1, seed=42, rank=1, world=2, batch_size=4, device=device)
    assert len(a) == len(b)
    assert len(a) % 4 == 0
    assert set(a).isdisjoint(set(b))
