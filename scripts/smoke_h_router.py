"""Smoke test for the H-aware Adaptive Evidence Router (no full model load).

Verifies, in isolation:
  1. build_router_condition (real/shuffled/flat) semantics.
  2. extract_routed_region_cells shapes + role diversity + budget <= max_cells.
  3. RegionAdapter: use_hstat=false ignores hstat; role branch is zero-init parity.
  4. load_region_adapter tolerates the newly-added role_embedding (old checkpoint).
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from outcome.inputs import (build_router_condition, extract_routed_region_cells,
                            _fps_partition, _dilate_mask, _normalize_h)
from models.region_adapter import RegionAdapter
from models.region_injection import load_region_adapter


def _check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        sys.exit(1)


def test_build_router_condition():
    h = torch.rand(6, 8)
    real = build_router_condition(h, 'real', 'k')
    _check("router real is identity", torch.equal(real, h))
    flat = build_router_condition(h, 'flat', 'k')
    _check("router flat is constant mean", torch.allclose(flat, h.mean(), atol=1e-6))
    shuf = build_router_condition(h, 'shuffled', 'k')
    _check("router shuffled permutes", sorted(shuf.flatten().tolist()) == sorted(h.flatten().tolist()))
    _check("router shuffled != real", not torch.equal(shuf, h))


def test_fps_partition():
    cells = [(y, x) for y in range(6) for x in range(6)]
    clusters = _fps_partition(cells, 4)
    _check("fps returns <= K clusters", len(clusters) <= 4)
    all_cells = [c for cl in clusters for c in cl]
    _check("fps covers all cells", sorted(all_cells) == sorted(cells))
    # seed pinning
    cl = _fps_partition(cells, 1, first=(5, 5))
    _check("fps first-seed pinned", (5, 5) in cl[0])


def test_dilate_normalize():
    m = np.zeros((5, 5), dtype=bool)
    m[2, 2] = True
    d = _dilate_mask(m, 1)
    _check("dilate r=1 -> 3x3", d.sum() == 9)
    h = _normalize_h(np.array([[1.0, 2.0], [3.0, 4.0]]))
    _check("normalize H -> [0,1]", abs(h.max() - 1.0) < 1e-5 and abs(h.min()) < 1e-5)


def test_router_cells():
    torch.manual_seed(0)
    Ht, Wt, D = 10, 12, 16
    test_f = torch.rand(Ht * Wt, D)
    ref_f = torch.rand(Ht * Wt, D)
    hmap = torch.rand(Ht, Wt)
    # two proposals: a blob at top-left, another at bottom-right
    masks = np.zeros((2, Ht, Wt), dtype=bool)
    masks[0, 1:4, 1:5] = True
    masks[1, 6:9, 7:11] = True
    cfg = dict(
        max_cells=15, geometry_dim=5, hstat_dim=2, num_roles=3,
        router=dict(min_tokens_per_candidate=3, context_radius_min=1,
                    context_radius_max=4, core_fraction=0.5),
    )
    raw, meta = extract_routed_region_cells(masks, hmap, test_f, ref_f, cfg)
    n = raw['valid'].shape[1]
    _check("router n_tokens <= max_cells", n <= 15)
    _check("router test shape [1,n,D]", raw['test'].shape == (1, n, D))
    _check("router role dtype long", raw['role'].dtype == torch.long)
    _check("router role shape [1,n]", raw['role'].shape == (1, n))
    roles = set(raw['role'][0].tolist())
    _check("router has core+extent+context roles", roles >= {0, 1, 2})
    _check("router owner all valid", (raw['owner'][0] >= 0).all())
    _check("router geom 5 dims", raw['geom'].shape[-1] == 5)
    _check("router meta len == 2", len(meta) == 2)
    # budgets >= min_per and roles consistent
    for m in meta:
        _check(f"budget >= 3 (cand {m['candidate']})", m['budget'] >= 3)
    # flat H -> all uncertainty ~1 -> radius == r_max
    flat = torch.full_like(hmap, float(hmap.mean()))
    _, meta_flat = extract_routed_region_cells(masks, flat, test_f, ref_f, cfg)
    _check("flat H -> max context radius", all(m['radius'] == 4 for m in meta_flat))


def test_adapter_use_hstat_and_role():
    B, n, D, hidden, inter = 1, 5, 16, 32, 8
    torch.manual_seed(1)
    raw = dict(
        test=torch.randn(B, n, D), ref=torch.randn(B, n, D),
        geom=torch.randn(B, n, 5), hstat=torch.randn(B, n, 2),
        valid=torch.ones(B, n, dtype=torch.bool),
    )
    # use_hstat=False: output must not depend on hstat at all
    a_no = RegionAdapter(feature_dim=D, hidden_size=hidden, intermediate_dim=inter,
                         num_roles=3, use_hstat=False)
    a_no.role_embedding.weight.data.zero_()
    out1 = a_no(raw)
    raw2 = dict(raw)
    raw2['hstat'] = torch.randn(B, n, 2) * 100
    out2 = a_no(raw2)
    _check("use_hstat=False ignores hstat", torch.allclose(out1, out2, atol=1e-6))
    # role branch zero-init: role=0 contributes nothing vs. absent role
    a_yes = RegionAdapter(feature_dim=D, hidden_size=hidden, intermediate_dim=inter,
                          num_roles=3, use_hstat=True)
    a_yes.role_embedding.weight.data.zero_()
    out_norole = a_yes(raw)
    raw_role = dict(raw)
    raw_role['role'] = torch.zeros(B, n, dtype=torch.long)
    out_role0 = a_yes(raw_role)
    _check("zero-init role branch == no role", torch.allclose(out_norole, out_role0, atol=1e-6))
    # nonzero role embedding must change output (sanity that branch is wired)
    a_yes.role_embedding.weight.data.normal_()
    out_role0b = a_yes(raw_role)
    _check("trained role embedding changes output", not torch.allclose(out_role0, out_role0b, atol=1e-3))
    # config round-trip
    cfg = a_no.config_dict
    _check("config_dict has num_roles/use_hstat", cfg['num_roles'] == 3 and cfg['use_hstat'] is False)


def test_load_old_checkpoint():
    D, hidden, inter = 16, 32, 8
    old = RegionAdapter(feature_dim=D, hidden_size=hidden, intermediate_dim=inter,
                        num_roles=3, use_hstat=True)
    old_sd = {k: v for k, v in old.state_dict().items() if k != 'role_embedding.weight'}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'region_adapter.pt')
        torch.save(dict(state_dict=old_sd, config=old.config_dict), p)
        loaded = load_region_adapter(RegionAdapter, p, D, hidden)
    _check("load old checkpoint tolerates missing role_embedding",
           'role_embedding.weight' in loaded.state_dict()
           and loaded.role_embedding.weight.abs().sum() == 0.0)


if __name__ == '__main__':
    test_build_router_condition()
    test_fps_partition()
    test_dilate_normalize()
    test_router_cells()
    test_adapter_use_hstat_and_role()
    test_load_old_checkpoint()
    print("\nAll router smoke tests passed.")
