"""Shared full-image + optional high-res H / original-crop zoom input construction."""
from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.prior_dataset import PriorCollator, PriorCoTDataset, apply_chat_template_safe
from models.anomaly_prior import fuse_maps, heatmap_to_pil, softmax_fuse_maps, unpack_merge_order
from models.qwen35 import qwen_vision_factor
from models.h_vpt import control_token_id_of, feat_token_id_of
from models.h_box_prior import GEOM_DIM, box_token_id_of, geometry_from_proposals
from models.region_injection import REGION_TOKEN, region_token_id_of
from outcome.protocol import render_prompt, validate_gt
from utils.common import smart_resize


@torch.no_grad()
def encode_pair_canonical(prior, pixels, grid):
    """Get H features and merger cache from ONE official joint vision forward.

    Hooks observe selected pre-merger block outputs without changing computation.
    This avoids model-/shape-dependent BF16 differences from manual per-image ViT.
    """
    if grid.shape != (2, 3) or not bool((grid[:, 0] == 1).all()):
        raise ValueError('canonical pair encoder expects two still images')
    captured = {}
    handles = []
    def hook(index):
        def capture(module, args, output):
            captured[index] = (output[0] if isinstance(output, tuple) else output).detach()
        return capture
    prior.visual.eval()
    try:
        for index in prior.block_indices:
            handles.append(prior.visual.blocks[index].register_forward_hook(hook(index)))
        merged = prior.visual(pixels.to(dtype=prior.visual.dtype), grid_thw=grid).pooler_output
    finally:
        for handle in handles:
            handle.remove()
    counts = [int(v) for v in grid.prod(-1)]
    hw_r, hw_t = tuple(int(v) for v in grid[0, 1:]), tuple(int(v) for v in grid[1, 1:])
    region_idx = int(getattr(prior, 'region_feature_index', prior.block_indices[-1]))
    match_fn = getattr(prior, '_nn_match', None)
    test_features = matched_ref_features = match_coordinates = None
    maps = []
    for index in prior.block_indices:
        ref, test = torch.split(captured[index], counts, dim=0)
        ref = unpack_merge_order(ref, *hw_r, prior.spatial_merge_size)
        test = unpack_merge_order(test, *hw_t, prior.spatial_merge_size)
        if match_fn is not None:
            match = match_fn(test, ref, hw_t, hw_r, prior.neighborhood_radius)
            maps.append(match['distance'])
            if index == region_idx:
                test_features = test.detach()
                matched_ref_features = match['matched_ref_features'].detach()
                # [Ht*Wt, 2] (y, x) reference-grid coordinate each test patch matched.
                match_coordinates = match['match_coordinates'].detach()
        else:
            maps.append(prior._nn_map(test, ref, hw_t, hw_r, prior.neighborhood_radius))
    stack = torch.stack(maps)
    hmap = (
        fuse_maps(stack, getattr(prior, "fusion_mode", "softmax"),
                  prior.temperature, getattr(prior, "normalize_layers", False))
        if len(maps) > 1 else stack[0]
    )
    return dict(patch_map=hmap, merged_embeddings=merged.detach(),
                test_features=test_features, matched_ref_features=matched_ref_features,
                match_coordinates=match_coordinates,
                test_grid_hw=hw_t)


@torch.no_grad()
def encode_visual_merged(prior, pixels, grid):
    """Merger tokens only; no H hooks. Used for global 448 views and zoom crops."""
    prior.visual.eval()
    merged = prior.visual(pixels.to(dtype=prior.visual.dtype), grid_thw=grid).pooler_output
    return merged.detach()


@contextmanager
def pixel_budget(processor, max_size: int, min_pixels: int = 256 * 256):
    """Temporarily raise the official image processor pixel cap (needed for H@768)."""
    img = getattr(processor, 'image_processor', None)
    if img is None:
        yield
        return
    old_min = getattr(img, 'min_pixels', None)
    old_max = getattr(img, 'max_pixels', None)
    old_size = getattr(img, 'size', None)
    cap = int(max_size) * int(max_size)
    floor = min(int(min_pixels), cap)
    img.min_pixels = floor
    img.max_pixels = cap
    if isinstance(old_size, dict):
        img.size = dict(old_size)
        img.size['shortest_edge'] = floor
        img.size['longest_edge'] = cap
    try:
        yield
    finally:
        if old_min is not None:
            img.min_pixels = old_min
        if old_max is not None:
            img.max_pixels = old_max
        if isinstance(old_size, dict):
            img.size = old_size


def crop_original_by_box1000(image: Image.Image, box, expand: float = 1.5, min_side: int = 32):
    """Crop ``image`` (original pixels) around a [0,1000] box, expanded by ``expand``.

    Returns a RGB PIL crop, or None if the box collapses after clamping.
    """
    if image is None or box is None or len(box) != 4:
        return None
    w, h = image.size
    if w < 2 or h < 2:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    px1, py1 = x1 / 1000.0 * w, y1 / 1000.0 * h
    px2, py2 = x2 / 1000.0 * w, y2 / 1000.0 * h
    bw, bh = max(px2 - px1, 1.0), max(py2 - py1, 1.0)
    cx, cy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
    scale = max(float(expand), 1.0)
    nw, nh = max(bw * scale, float(min_side)), max(bh * scale, float(min_side))
    left = int(round(cx - nw / 2.0))
    top = int(round(cy - nh / 2.0))
    right = int(round(cx + nw / 2.0))
    bottom = int(round(cy + nh / 2.0))
    left = max(0, min(w - 2, left))
    top = max(0, min(h - 2, top))
    right = max(left + 2, min(w, right))
    bottom = max(top + 2, min(h, bottom))
    if right - left < 2 or bottom - top < 2:
        return None
    return image.convert('RGB').crop((left, top, right, bottom))


def crop_original_by_center1000(image: Image.Image, cx: float, cy: float,
                                crop_fraction: float = 0.20, min_side: int = 32):
    """Crop ``image`` (original pixels) around a [0,1000] center point.

    The window is a square of ``crop_fraction`` of the image's *smaller* side, so
    a 10x10px defect at 20% becomes a ~90-200px crop before resizing. Returns None
    if the crop collapses after clamping.
    """
    if image is None:
        return None
    w, h = image.size
    if w < 2 or h < 2:
        return None
    frac = max(0.0, min(float(crop_fraction), 1.0))
    side = max(frac * float(min(w, h)), float(min_side))
    px, py = float(cx) / 1000.0 * w, float(cy) / 1000.0 * h
    left = int(round(px - side / 2.0))
    top = int(round(py - side / 2.0))
    right = int(round(px + side / 2.0))
    bottom = int(round(py + side / 2.0))
    left = max(0, min(w - 2, left))
    top = max(0, min(h - 2, top))
    right = max(left + 2, min(w, right))
    bottom = max(top + 2, min(h, bottom))
    if right - left < 2 or bottom - top < 2:
        return None
    return image.convert('RGB').crop((left, top, right, bottom))


def zoom_prompt_suffix(n_crops: int, paired: bool = False, reference_mode: str = 'matched') -> str:
    """Tell the model extra images are full-image-coordinate zooms of Image 2.

    ``reference_mode`` is 'matched' (reference crop at the *most similar* normal
    location) or 'same' (reference crop at the identical [0,1000] coordinate as
    the inspection crop). The suffix always stresses that a zoomed region is a
    hypothesis to inspect, not an anomaly label.
    """
    n = int(n_crops)
    if n <= 0:
        return ''
    ref_desc = 'matched normal reference' if reference_mode == 'matched' else 'normal reference'
    if paired:
        if n == 1:
            span = ('Image 3 is a magnified inspection crop and Image 4 is the '
                    f'{ref_desc} crop at the same location')
        else:
            span = (f'For each of the {n} marked candidates, a pair of magnified crops '
                    f'follows: first the inspection crop, then its {ref_desc} crop')
    elif n == 1:
        span = 'Image 3 is a magnified crop of Image 2 at the first marked candidate'
    else:
        span = (f'Images 3-{2 + n} are magnified crops of Image 2 at the marked '
                f'candidate locations, in that order')
    return (
        f'\n{span}. A zoomed region is only a suspicious hypothesis and may still be '
        'normal: use these magnified views in [confirm] and <answer> to keep, refine, '
        'or reject the hypothesis. All boxes stay Image 2 FULL IMAGE integers in '
        '[0,1000]; do not write crop-local coordinates.'
    )


def format_region_hints(proposals, n_tokens: int, owners=None, expose_peak: bool = True) -> str:
    """One ``<|region|>`` per token, grouped by H region.

    With ``expose_peak=True`` the first token of each region is ``hK:<|region|>@[x,y]``
    (peak on Image 2) and later tokens are bare ``<|region|>``. With ``expose_peak=False``
    every token is a bare ``<|region|>`` so H coordinates never enter the text.
    """
    n = max(1, int(n_tokens))
    if not proposals or not expose_peak:
        return ' '.join([REGION_TOKEN] * n)
    if owners is None:
        owners = list(range(min(n, len(proposals)))) + [-1] * max(0, n - len(proposals))
    else:
        owners = [int(v) for v in list(owners)[:n]]
        owners.extend([-1] * (n - len(owners)))
    parts = []
    i = 0
    while i < n:
        own = owners[i]
        prop = proposals[own] if 0 <= own < len(proposals) else None
        peak = (prop or {}).get('peak_2d') or []
        if prop is not None and len(peak) == 2:
            x, y = int(round(float(peak[0]))), int(round(float(peak[1])))
            hid = prop.get('id') or f'h{own + 1}'
            parts.append(f'{hid}:{REGION_TOKEN}@[{x},{y}]')
        else:
            parts.append(REGION_TOKEN)
        i += 1
        while i < n and owners[i] == own and own >= 0:
            parts.append(REGION_TOKEN)
            i += 1
    return ' '.join(parts)


def _peak_core_cells(cells, scores, radius: int, keep: float):
    """Keep the peak cell plus nearby cells that still carry most of the peak H."""
    peak_idx = int(np.argmax(scores))
    py, px = int(cells[peak_idx][0]), int(cells[peak_idx][1])
    peak = float(scores[peak_idx])
    radius = max(0, int(radius))
    floor = float(peak) * max(0.0, min(float(keep), 1.0))
    core = []
    for (cy, cx), score in zip(cells, scores):
        if max(abs(int(cy) - py), abs(int(cx) - px)) <= radius and float(score) >= floor:
            core.append((int(cy), int(cx)))
    if not core:
        core = [(py, px)]
    return core, (py, px)


def region_proposals(hmap, cfg):
    """Four-connected patch regions; bbox uses cell EDGES (including singleton).

    ``box_mode=full`` (default) keeps the original connected-component bbox.
    ``box_mode=peak_core`` shrinks each component to the peak cell plus a small
    high-H neighbourhood; A/B'd as too small for IoU.

    Returns ``(proposal_meta, proposal_masks, mode)``:
      - ``proposal_meta`` : list of dicts (id/bbox_2d/peak_2d/raw_peak/raw_mean/area_fraction)
      - ``proposal_masks`` : bool ndarray ``[R, Ht, Wt]`` (one mask per candidate region)
      - ``mode``          : threshold mode string for logging
    """
    arr = hmap.detach().float().cpu().numpy() if torch.is_tensor(hmap) else np.asarray(hmap, dtype=float)
    if arr.ndim != 2 or not arr.size or not np.isfinite(arr).all():
        raise ValueError('H must be a finite nonempty 2D patch map')
    h, w = arr.shape
    threshold = cfg.get('raw_threshold')
    mode = 'absolute_raw_uncalibrated' if threshold is not None else 'relative_uncalibrated'
    if threshold is None:
        if float(arr.max()-arr.min()) < 1e-8:
            return [], np.zeros((0, h, w), dtype=bool), mode
        fraction = float(cfg.get('relative_threshold', 0.7))
        if not 0 < fraction <= 1:
            raise ValueError('relative_threshold must be in (0,1]')
        threshold = float(arr.min()+fraction*(arr.max()-arr.min()))
    threshold = float(threshold)
    if not math.isfinite(threshold):
        raise ValueError('nonfinite H threshold')
    mask = arr >= threshold
    seen = np.zeros_like(mask)
    regions = []
    region_masks = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        queue = [(int(y), int(x))]
        seen[y, x] = True
        cells = []
        while queue:
            cy, cx = queue.pop()
            cells.append((cy, cx))
            for ny, nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny,nx] and not seen[ny,nx]:
                    seen[ny,nx] = True
                    queue.append((ny,nx))
        if len(cells) < int(cfg.get('min_cells', 1)):
            continue
        yy, xx = np.array(cells).T
        scores = arr[yy, xx]
        box_mode = str(cfg.get('box_mode', 'full'))
        if box_mode == 'peak_core':
            core, (py, px) = _peak_core_cells(
                cells, scores,
                radius=int(cfg.get('peak_radius', 1)),
                keep=float(cfg.get('peak_keep', 0.85)))
            yy, xx = np.array(core).T
            core_scores = arr[yy, xx]
            peak_y, peak_x = py, px
            used = core
            used_scores = core_scores
        elif box_mode == 'full':
            peak_idx = int(scores.argmax())
            peak_y, peak_x = int(yy[peak_idx]), int(xx[peak_idx])
            used, used_scores = cells, scores
        else:
            raise ValueError(f'unknown prior.box_mode: {box_mode}')
        regions.append(dict(bbox_2d=[round(float(xx.min())/w*1000, 3), round(float(yy.min())/h*1000, 3),
                                    round(float(xx.max()+1)/w*1000, 3), round(float(yy.max()+1)/h*1000, 3)],
                            peak_2d=[round((float(peak_x)+.5)/w*1000, 3),
                                     round((float(peak_y)+.5)/h*1000, 3)],
                            raw_peak=round(float(used_scores.max()), 6), raw_mean=round(float(used_scores.mean()), 6),
                            area_fraction=round(len(used)/(h*w), 6)))
        region_masks.append(_cells_to_mask(h, w, used))
    order = sorted(range(len(regions)),
                   key=lambda i: (-regions[i]['raw_peak'], -regions[i]['raw_mean'], regions[i]['bbox_2d']))
    order = order[:max(0, int(cfg.get('max_candidates', 3)))]
    regions = [regions[i] for i in order]
    region_masks = [region_masks[i] for i in order]
    for i, region in enumerate(regions):
        region['id'] = f'h{i+1}'
    masks_arr = np.stack(region_masks, axis=0) if region_masks else np.zeros((0, h, w), dtype=bool)
    return regions, masks_arr, mode


def _cells_to_mask(h, w, cells):
    """Build a dense [H, W] bool mask from a list of (y, x) cell indices."""
    out = np.zeros((h, w), dtype=bool)
    if not cells:
        return out
    yy, xx = np.array(cells).T
    out[yy, xx] = True
    return out


def _extract_h_patches(hmap, geoms, patch_size):
    """Sample a ``patch_size x patch_size`` patch from sample-normalized H per token.

    ``geoms`` is a list of ``[gcx, gcy, gw, gh, r/R]`` where ``gcx/gcy`` are cell
    centers in ``[0,1]`` (the same convention ``extract_region_cells`` writes into
    ``geom``). H is first normalized per-sample to ``[0,1]`` so the patch encodes
    the *shape* of the anomaly prior around the region, not its absolute magnitude.
    Bilinear sampling handles fractional grid-token centers and clamps at borders.

    Returns ``[N, 1, P, P]`` float32 (N == len(geoms)). ``patch_size <= 0`` returns
    an empty tensor so callers can gate the feature off without shape mismatch.
    """
    import torch.nn.functional as F

    P = max(0, int(patch_size))
    n = len(geoms)
    if P == 0 or n == 0:
        return torch.zeros(n, 1, 0, 0, dtype=torch.float32, device=hmap.device)

    h = hmap.detach().float()
    Ht, Wt = int(h.shape[0]), int(h.shape[1])
    if Ht < 2 or Wt < 2:
        return torch.zeros(n, 1, P, P, dtype=torch.float32, device=hmap.device)

    # sample-wise normalization -> [0,1]
    hmin = h.min()
    denom = (h.max() - hmin) + 1e-6
    h = (h - hmin) / denom
    h = h.reshape(1, 1, Ht, Wt)

    half = (P - 1) / 2.0
    off = torch.arange(P, device=hmap.device, dtype=torch.float32) - half
    patches = []
    for g in geoms:
        gcx, gcy = float(g[0]), float(g[1])
        # cell-center in element-index space (element k occupies [k, k+1))
        cx = gcx * Wt - 0.5
        cy = gcy * Ht - 0.5
        xs = cx + off
        ys = cy + off
        # grid_sample(align_corners=False): normalized = 2*(element + 0.5)/size - 1
        xn = 2.0 * (xs + 0.5) / Wt - 1.0
        yn = 2.0 * (ys + 0.5) / Ht - 1.0
        gy, gx = torch.meshgrid(yn, xn, indexing='ij')
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # [1, P, P, 2] (x, y)
        out = F.grid_sample(h, grid, mode='bilinear', align_corners=False,
                            padding_mode='border')
        patches.append(out[0, 0])  # [P, P]
    return torch.stack(patches, dim=0).unsqueeze(1)  # [N, 1, P, P]


def extract_region_cells(proposal_masks, test_features, matched_ref_features, hmap, cfg,
                         patch_size: int = 0):
    """Turn candidate region masks into contrast features for the region adapter.

    ``token_mode=hybrid`` (default): peak cell first, then a 2x2 split of the
    region bbox so the LLM sees both the hottest difference and its extent.
    ``token_mode=peak``: one token per region at its highest-H cell.
    ``token_mode=grid``: legacy 2x2 split only.

    ``patch_size > 0`` additionally returns ``hpatch`` ([1,n,1,P,P]) — a PxP
    sample-normalized H patch around each token's ``geom`` center, the extra input
    for ``HPriorAdapter`` (kept separate from the main RegionAdapter).

    Returns dict(test=[1,n,D], ref=[1,n,D], geom=[1,n,G], hstat=[1,n,H],
    hpatch=[1,n,1,P,P], valid=[1,n], owner=[1,n]). Always returns at least one
    cell: a single invalid "empty" cell when there are no candidate regions.
    """
    dev = test_features.device
    Ht, Wt = int(hmap.shape[0]), int(hmap.shape[1])
    D = int(test_features.shape[-1])
    test_f = test_features.float().reshape(Ht, Wt, D)
    ref_f = matched_ref_features.float().reshape(Ht, Wt, D)
    h = hmap.detach().float().reshape(Ht, Wt)
    R = int(proposal_masks.shape[0])
    max_cells = max(1, int(cfg.get('max_cells', 12)))
    token_mode = str(cfg.get('token_mode', 'hybrid'))
    if token_mode not in ('hybrid', 'peak', 'grid'):
        raise ValueError(f'unknown region.token_mode: {token_mode}')
    pmask = torch.from_numpy(np.asarray(proposal_masks, dtype=bool)).to(dev)
    out_test, out_ref, out_geom, out_hstat, out_owner = [], [], [], [], []

    def _peak(r, ys, xs):
        local = h[ys, xs]
        pick = int(local.argmax())
        y, x = int(ys[pick]), int(xs[pick])
        out_test.append(test_f[y, x])
        out_ref.append(ref_f[y, x])
        out_geom.append([(x + 0.5) / Wt, (y + 0.5) / Ht, 1.0 / Wt, 1.0 / Ht, r / max(R, 1)])
        out_hstat.append([float(h[y, x]), float(local.mean())])
        out_owner.append(r)

    def _grid(r, ys, xs):
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        height, width = y1 - y0 + 1, x1 - x0 + 1
        y_edges = [y0, y0 + height // 2, y1 + 1] if height >= 2 else [y0, y1 + 1]
        x_edges = [x0, x0 + width // 2, x1 + 1] if width >= 2 else [x0, x1 + 1]
        for i in range(len(y_edges) - 1):
            for j in range(len(x_edges) - 1):
                sy0, sy1, sx0, sx1 = y_edges[i], y_edges[i + 1], x_edges[j], x_edges[j + 1]
                sub_mask = pmask[r, sy0:sy1, sx0:sx1].reshape(-1)
                if not sub_mask.any():
                    continue
                sub_test = test_f[sy0:sy1, sx0:sx1].reshape(-1, D)[sub_mask]
                sub_ref = ref_f[sy0:sy1, sx0:sx1].reshape(-1, D)[sub_mask]
                sub_h = h[sy0:sy1, sx0:sx1].reshape(-1)[sub_mask]
                w = sub_h - sub_h.min() + 1e-6
                w = w / w.sum()
                out_test.append((sub_test * w[:, None]).sum(0))
                out_ref.append((sub_ref * w[:, None]).sum(0))
                gcx = ((sx0 + sx1 - 1) / 2.0 + 0.5) / Wt
                gcy = ((sy0 + sy1 - 1) / 2.0 + 0.5) / Ht
                gw = (sx1 - sx0) / Wt
                gh = (sy1 - sy0) / Ht
                out_geom.append([gcx, gcy, gw, gh, r / max(R, 1)])
                out_hstat.append([float(sub_h.max()), float(sub_h.mean())])
                out_owner.append(r)

    for r in range(R):
        ys, xs = torch.nonzero(pmask[r], as_tuple=True)
        if ys.numel() == 0:
            continue
        if token_mode in ('peak', 'hybrid'):
            _peak(r, ys, xs)
        if token_mode in ('grid', 'hybrid'):
            _grid(r, ys, xs)
    P = max(1, int(patch_size))
    if not out_test:
        return dict(test=torch.zeros(1, 1, D, device=dev),
                    ref=torch.zeros(1, 1, D, device=dev),
                    geom=torch.zeros(1, 1, int(cfg.get('geometry_dim', 5)), device=dev),
                    hstat=torch.zeros(1, 1, int(cfg.get('hstat_dim', 2)), device=dev),
                    hpatch=torch.zeros(1, 1, 1, P, P, device=dev),
                    role=torch.zeros(1, 1, device=dev, dtype=torch.long),
                    valid=torch.zeros(1, 1, dtype=torch.bool, device=dev),
                    owner=torch.full((1, 1), -1, device=dev, dtype=torch.long))
    n = min(len(out_test), max_cells)
    hpatch = _extract_h_patches(h, out_geom[:n], patch_size)
    return dict(test=torch.stack(out_test[:n]).unsqueeze(0),
                ref=torch.stack(out_ref[:n]).unsqueeze(0),
                geom=torch.tensor(out_geom[:n], device=dev, dtype=torch.float32).unsqueeze(0),
                hstat=torch.tensor(out_hstat[:n], device=dev, dtype=torch.float32).unsqueeze(0),
                hpatch=hpatch.unsqueeze(0),  # [1, n, 1, P, P]
                role=torch.zeros(1, n, device=dev, dtype=torch.long),
                valid=torch.ones(1, n, dtype=torch.bool, device=dev),
                owner=torch.tensor(out_owner[:n], device=dev, dtype=torch.long).unsqueeze(0))


# ---------------------------------------------------------------------------
# H-aware Adaptive Evidence Router (token_mode=routed)
# ---------------------------------------------------------------------------
# H is NOT injected into the LLM. Instead it allocates the region-evidence budget:
# for each real-H proposal it selects Core / Extent / Context sub-regions, decides
# how broadly to search the surrounding context (radius from H confidence), and how
# many tokens each role gets — all from cached block24 test/matched-ref features
# (no extra ViT forward). The router's H input is a separate ablation variable
# (real | shuffled | flat) while proposals always come from the real H.

CORE, EXTENT, CONTEXT = 0, 1, 2


def build_router_condition(hmap_real, condition: str, seed_key: str) -> torch.Tensor:
    """Router-side H map for the causal ablation (proposals stay real-H)."""
    condition = str(condition)
    if condition == 'real':
        return hmap_real
    if condition == 'shuffled':
        s = int(hashlib.sha256(seed_key.encode()).hexdigest()[:8], 16)
        perm = torch.randperm(hmap_real.numel(), generator=torch.Generator().manual_seed(s))
        return hmap_real.flatten()[perm.to(hmap_real.device)].reshape_as(hmap_real)
    if condition == 'flat':
        return torch.full_like(hmap_real, float(hmap_real.mean()))
    raise ValueError(f'unknown router.condition: {condition}')


def _normalize_h(h: np.ndarray) -> np.ndarray:
    h = h.astype(np.float32)
    denom = float(h.max() - h.min()) + 1e-6
    return (h - float(h.min())) / denom


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation by a square (2r+1)x(2r+1) structuring element."""
    radius = max(0, int(radius))
    if radius == 0:
        return mask.copy()
    h, w = mask.shape
    out = mask.copy()
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return out
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ny = np.clip(ys + dy, 0, h - 1)
            nx = np.clip(xs + dx, 0, w - 1)
            out[ny, nx] = True
    return out


def _mask_cells(mask: np.ndarray) -> list:
    return [(int(y), int(x)) for y, x in zip(*np.nonzero(mask))]


def _fps_partition(cells, n_tokens: int, first=None) -> list:
    """Farthest-point partition of ``cells`` into ``n_tokens`` clusters.

    ``first`` optionally pins the first seed (a (y, x) cell, e.g. the H peak).
    Returns a list of ``K`` lists of (y, x) cells (K <= n_tokens, K <= len(cells)).
    """
    if not cells:
        return []
    K = min(max(1, int(n_tokens)), len(cells))
    pts = np.array(cells, dtype=float)
    if first is not None:
        d = (pts[:, 0] - first[0]) ** 2 + (pts[:, 1] - first[1]) ** 2
        s0 = int(np.argmin(d))
    else:
        s0 = 0
    seeds = [s0]
    dist = np.full(len(cells), np.inf)

    def _update(s):
        d = np.sqrt((pts[:, 0] - pts[s, 0]) ** 2 + (pts[:, 1] - pts[s, 1]) ** 2)
        np.minimum(dist, d, out=dist)

    _update(s0)
    for _ in range(1, K):
        s = int(np.argmax(dist))
        seeds.append(s)
        _update(s)
    clusters = [[] for _ in range(K)]
    sd = np.sqrt((pts[:, 0][:, None] - pts[seeds, 0][None, :]) ** 2 +
                 (pts[:, 1][:, None] - pts[seeds, 1][None, :]) ** 2)
    assign = np.argmin(sd, axis=1)
    for i in range(len(cells)):
        clusters[int(assign[i])].append((int(pts[i, 0]), int(pts[i, 1])))
    return clusters


def _pool_cells(test_f, ref_f, h_norm, cells, weights=None):
    """Pool a cluster of cells into (test_vec, ref_vec, geom, hstat)."""
    ys = np.array([c[0] for c in cells], dtype=int)
    xs = np.array([c[1] for c in cells], dtype=int)
    t = test_f[ys, xs]  # [K, D]
    r = ref_f[ys, xs]
    hh = h_norm[ys, xs]
    if weights is None:
        w = np.full(len(cells), 1.0 / max(1, len(cells)), dtype=np.float32)
    else:
        w = np.asarray(weights, dtype=np.float32)
        w = w / (w.sum() + 1e-6)
    tvec = (t * w[:, None]).sum(0)
    rvec = (r * w[:, None]).sum(0)
    Ht, Wt = test_f.shape[0], test_f.shape[1]
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    geom = [(x0 + x1) / 2.0 / Wt + 0.5 / Wt,
            (y0 + y1) / 2.0 / Ht + 0.5 / Ht,
            (x1 - x0 + 1) / Wt,
            (y1 - y0 + 1) / Ht,
            0.0]  # owner fraction patched later
    hstat = [float(hh.max()), float(hh.mean())]
    return tvec, rvec, geom, hstat


def extract_routed_region_cells(proposal_masks, router_hmap, test_features,
                                matched_ref_features, cfg):
    """H-aware adaptive evidence router: Core / Extent / Context with dynamic budget.

    ``proposal_masks`` (bool [R, Ht, Wt]) always come from the real H. ``router_hmap``
    is the ablation variable (real/shuffled/flat) and drives confidence, context
    radius, and the Core threshold — i.e. only H's *second-stage routing* role.

    Returns ``(region_raw, router_meta)`` where region_raw matches the shape of
    ``extract_region_cells`` but adds a ``role`` field ([1, n] long, 0/1/2), and
    ``router_meta`` records per-candidate confidence/radius/budget for logging.
    """
    dev = test_features.device
    Ht, Wt = int(router_hmap.shape[0]), int(router_hmap.shape[1])
    D = int(test_features.shape[-1])
    test_f = test_features.float().reshape(Ht, Wt, D).cpu().numpy()
    ref_f = matched_ref_features.float().reshape(Ht, Wt, D).cpu().numpy()
    rc = dict(cfg)
    router = dict(rc.get('router') or {})
    max_cells = max(1, int(rc.get('max_cells', 15)))
    min_per = max(1, int(router.get('min_tokens_per_candidate', 3)))
    r_min = int(router.get('context_radius_min', 1))
    r_max = int(router.get('context_radius_max', 4))
    beta = float(router.get('core_fraction', 0.5))
    if r_max < r_min:
        r_max = r_min

    h = router_hmap.detach().float().cpu().numpy().reshape(Ht, Wt)
    h_norm = _normalize_h(h)
    R = int(proposal_masks.shape[0])
    pmask = np.asarray(proposal_masks, dtype=bool)

    out_test, out_ref, out_geom, out_hstat, out_role, out_owner = [], [], [], [], [], []
    router_meta = []

    # ---- per-candidate confidence / radius / budget ----
    metas = []
    for i in range(R):
        M = pmask[i]
        if not M.any():
            continue
        p_i = float(h_norm[M].max())
        ring = _dilate_mask(M, 1) & ~M
        b_i = float(h_norm[ring].mean()) if ring.any() else 0.0
        c_i = float(np.clip(p_i - b_i, 0.0, 1.0))
        u_i = 1.0 - c_i
        radius = int(round(r_min + u_i * (r_max - r_min)))
        # budget weight q_i = peak * sqrt(area) (area in cells)
        area = float(M.sum())
        q_i = float(p_i * np.sqrt(area + 1e-6))
        metas.append(dict(i=i, p=p_i, b=b_i, conf=c_i, unc=u_i, radius=radius,
                          area=area, q=q_i))
    if not metas:
        return dict(test=torch.zeros(1, 1, D, device=dev),
                    ref=torch.zeros(1, 1, D, device=dev),
                    geom=torch.zeros(1, 1, int(rc.get('geometry_dim', 5)), device=dev),
                    hstat=torch.zeros(1, 1, int(rc.get('hstat_dim', 2)), device=dev),
                    role=torch.zeros(1, 1, device=dev, dtype=torch.long),
                    valid=torch.zeros(1, 1, dtype=torch.bool, device=dev),
                    owner=torch.full((1, 1), -1, device=dev, dtype=torch.long)), []

    # ---- allocate total budget across candidates (largest remainder) ----
    R_used = len(metas)
    q_sum = float(sum(m['q'] for m in metas)) + 1e-6
    B_extra = max_cells - R_used * min_per
    if B_extra < 0:
        B_extra = 0
    exact = []
    for m in metas:
        w = m['q'] / q_sum
        exact.append(dict(**m, w=w, budget=min_per + w * B_extra))
    # floor + largest remainder
    floors = [int(np.floor(e['budget'])) for e in exact]
    remainders = [(e['budget'] - f, idx) for idx, (e, f) in enumerate(zip(exact, floors))]
    rem_budget = max_cells - sum(floors)
    if rem_budget > 0:
        remainders.sort(key=lambda t: -t[0])
        for _, idx in remainders[:rem_budget]:
            floors[idx] += 1
    # enforce min_per floor
    for idx in range(len(floors)):
        floors[idx] = max(min_per, floors[idx])
    # final cap to max_cells (if min_per*R > max_cells)
    while sum(floors) > max_cells:
        idx = max(range(len(floors)), key=lambda j: floors[j])
        if floors[idx] <= min_per:
            break
        floors[idx] -= 1
    for e, budget in zip(exact, floors):
        e['budget'] = budget

    # ---- emit Core / Extent / Context tokens ----
    for e in exact:
        i = e['i']
        M = pmask[i]
        ys, xs = np.nonzero(M)
        cells = list(zip(ys.tolist(), xs.tolist()))
        local = h_norm[M]
        thr = float(local.mean()) + beta * (float(local.max()) - float(local.mean()))
        core_mask = M & (h_norm >= thr)
        if not core_mask.any():
            # fall back to the peak cell
            pk = int(np.argmax(local))
            core_mask = _cells_to_mask(Ht, Wt, [cells[pk]])
        extent_mask = M & ~core_mask
        if not extent_mask.any():
            extent_mask = M.copy()
        context_mask = _dilate_mask(M, e['radius']) & ~M

        budget = e['budget']
        n_core = 1
        E = max(0, budget - min_per)  # extra beyond 1 core + 1 extent + 1 context
        n_context = 1 + int(round(e['unc'] * E))
        n_extent = max(1, budget - n_core - n_context)

        # Core: single cluster at H peak, H-weighted pooling.
        core_cells = _mask_cells(core_mask)
        pk_cell = max(core_cells, key=lambda c: h_norm[c[0], c[1]])
        tvec, rvec, geom, hstat = _pool_cells(test_f, ref_f, h_norm, core_cells,
                                              weights=[h_norm[c] for c in core_cells])
        out_test.append(torch.from_numpy(tvec))
        out_ref.append(torch.from_numpy(rvec))
        out_geom.append(geom)
        out_hstat.append(hstat)
        out_role.append(CORE)
        out_owner.append(i)

        # Extent: n_extent clusters, seed at H peak, H-weighted pooling.
        extent_cells = _mask_cells(extent_mask)
        clusters = _fps_partition(extent_cells, n_extent, first=pk_cell)
        for cl in clusters:
            if not cl:
                continue
            tvec, rvec, geom, hstat = _pool_cells(test_f, ref_f, h_norm, cl,
                                                  weights=[h_norm[c] for c in cl])
            out_test.append(torch.from_numpy(tvec))
            out_ref.append(torch.from_numpy(rvec))
            out_geom.append(geom)
            out_hstat.append(hstat)
            out_role.append(EXTENT)
            out_owner.append(i)

        # Context: n_context clusters, uniform pooling, seed near proposal centroid.
        context_cells = _mask_cells(context_mask)
        if context_cells:
            cy = float(np.mean([c[0] for c in cells]))
            cx = float(np.mean([c[1] for c in cells]))
            clusters = _fps_partition(context_cells, n_context, first=(cy, cx))
        else:
            clusters = []
        for cl in clusters:
            if not cl:
                continue
            tvec, rvec, geom, hstat = _pool_cells(test_f, ref_f, h_norm, cl, weights=None)
            out_test.append(torch.from_numpy(tvec))
            out_ref.append(torch.from_numpy(rvec))
            out_geom.append(geom)
            out_hstat.append(hstat)
            out_role.append(CONTEXT)
            out_owner.append(i)

        router_meta.append(dict(
            candidate=i, confidence=round(float(e['conf']), 4),
            uncertainty=round(float(e['unc']), 4), radius=int(e['radius']),
            budget=int(budget), n_core=n_core, n_extent=n_extent, n_context=n_context))

    n = min(len(out_test), max_cells)
    if n == 0:
        return dict(test=torch.zeros(1, 1, D, device=dev),
                    ref=torch.zeros(1, 1, D, device=dev),
                    geom=torch.zeros(1, 1, int(rc.get('geometry_dim', 5)), device=dev),
                    hstat=torch.zeros(1, 1, int(rc.get('hstat_dim', 2)), device=dev),
                    role=torch.zeros(1, 1, device=dev, dtype=torch.long),
                    valid=torch.zeros(1, 1, dtype=torch.bool, device=dev),
                    owner=torch.full((1, 1), -1, device=dev, dtype=torch.long)), router_meta

    geoms = torch.tensor(out_geom[:n], device=dev, dtype=torch.float32)  # [n, 5]
    # 5th geometry slot = owner fraction (r / R), matching extract_region_cells.
    owner_frac = torch.tensor(
        [o / max(R, 1) for o in out_owner[:n]], device=dev, dtype=torch.float32)
    geoms = torch.cat([geoms[:, :4], owner_frac.unsqueeze(1)], dim=1)

    region_raw = dict(
        test=torch.stack(out_test[:n]).unsqueeze(0).to(dev),
        ref=torch.stack(out_ref[:n]).unsqueeze(0).to(dev),
        geom=geoms.unsqueeze(0),
        hstat=torch.tensor(out_hstat[:n], device=dev, dtype=torch.float32).unsqueeze(0),
        role=torch.tensor(out_role[:n], device=dev, dtype=torch.long).unsqueeze(0),
        valid=torch.ones(1, n, dtype=torch.bool, device=dev),
        owner=torch.tensor(out_owner[:n], device=dev, dtype=torch.long).unsqueeze(0),
    )
    return region_raw, router_meta


class OutcomeDataset(PriorCoTDataset):
    def __getitem__(self, index):
        item = self._load_pair(self.samples[index])
        if Path(item['image_path']).resolve() == Path(item['ref_path']).resolve():
            raise ValueError('normal reference must not be the inspection image')
        getattr(self, 'validate_gt_fn', validate_gt)(item)
        return item


class OutcomeCollator(PriorCollator):
    def _align_pair_at(self, ref: Image.Image, test: Image.Image, max_size: int):
        """Resize the inspection image to ``max_size`` budget; match the reference canvas."""
        visual = getattr(self.prior, 'visual', None)
        factor = qwen_vision_factor(self.processor, visual)
        cap = int(max_size) * int(max_size)
        floor = min(256 * 256, cap)
        test_rs, _, _ = smart_resize(
            test, max_size=int(max_size), factor=factor, min_pixels=floor, max_pixels=cap)
        ref_rs = ref.resize(test_rs.size, Image.Resampling.BICUBIC)
        return ref_rs, test_rs

    def _resize_one(self, image: Image.Image, max_size: int) -> Image.Image:
        visual = getattr(self.prior, 'visual', None)
        factor = qwen_vision_factor(self.processor, visual)
        cap = int(max_size) * int(max_size)
        floor = min(256 * 256, cap)
        out, _, _ = smart_resize(
            image, max_size=int(max_size), factor=factor, min_pixels=floor, max_pixels=cap)
        return out

    def _zoom_cfg(self) -> dict:
        return dict(self.cfg.get('outcome', {}).get('zoom') or {})

    def _zoom_peaks(self, hmap, zcfg) -> list:
        """H local-max + NMS → top-K peaks in [0,1000], independent of CC proposals."""
        from models.vision_cache import topk_spatial_points

        max_crops = max(0, int(zcfg.get('max_crops', 3)))
        nms_radius = int(zcfg.get('nms_radius', 2))
        return topk_spatial_points(hmap, k=max_crops, nms_radius=nms_radius)

    def _matched_ref_crop(self, ref_original, qx, qy, hmap, match_coordinates, crop_fraction):
        """Crop the normal reference at the location each test peak *matched*.

        ``match_coordinates`` is [Ht*Wt, 2] (y, x) on the reference grid; index it
        by the test grid cell of the peak [0,1000] point, then crop the reference
        around the matched coordinate.
        """
        if ref_original is None or match_coordinates is None:
            return None
        arr = hmap.detach().float().cpu().numpy() if torch.is_tensor(hmap) else np.asarray(hmap, dtype=float)
        ht, wt = int(arr.shape[0]), int(arr.shape[1])
        if ht <= 0 or wt <= 0:
            return None
        mc = match_coordinates.detach().float().cpu().numpy()
        if mc.ndim != 2 or int(mc.shape[0]) != ht * wt:
            return None
        tx = max(0, min(wt - 1, int(round(float(qx) / 1000.0 * wt))))
        ty = max(0, min(ht - 1, int(round(float(qy) / 1000.0 * ht))))
        flat = ty * wt + tx
        ry = float(mc[flat, 0])
        rx = float(mc[flat, 1])
        rqx = (rx + 0.5) / float(wt) * 1000.0
        rqy = (ry + 0.5) / float(ht) * 1000.0
        return crop_original_by_center1000(ref_original, rqx, rqy, crop_fraction)

    def _gt_oracle_boxes(self, item) -> list:
        """GT component boxes in [0,1000] (eval-only oracle source)."""
        comps = list(item.get('component_bboxes') or [])
        if not comps and item.get('gt_box_px'):
            comps = [item['gt_box_px']]
        orig = item.get('orig_size')
        if not orig or not comps:
            return []
        w, h = float(orig[0]), float(orig[1])
        return [[b[0] / w * 1000.0, b[1] / h * 1000.0, b[2] / w * 1000.0, b[3] / h * 1000.0]
                for b in comps if len(b) == 4]

    def _build_zoom_crops(self, test_original, ref_original, proposals, hmap,
                          match_coordinates, oracle_boxes) -> tuple:
        """Build zoom crop entries: each entry is [test_crop] or [test_crop, ref_crop].

        ``source`` picks the proposal used for the test crop:
          - 'cc'    : existing connected-component bbox (expand)
          - 'peaks' : H local-max + NMS peak, fixed ``crop_fraction`` window
        ``include_reference`` appends a reference crop. ``reference_mode`` chooses
        where that reference crop is taken:
          - 'matched' : the location each test peak *matched* (needs match coords)
          - 'same'    : the identical [0,1000] coordinate as the test crop
        ``oracle_boxes`` (non-empty) overrides source with GT boxes (eval-only).
        """
        zcfg = self._zoom_cfg()
        expand = float(zcfg.get('expand', 1.5))
        crop_size = int(zcfg.get('crop_image_size', 448))
        max_crops = max(0, int(zcfg.get('max_crops', 3)))
        source = str(zcfg.get('source', 'cc'))
        include_ref = bool(zcfg.get('include_reference', False))
        reference_mode = str(zcfg.get('reference_mode', 'matched'))
        crop_fraction = float(zcfg.get('crop_fraction', 0.20))
        entries = []

        def _entry(center, box=None):
            if box is not None:
                raw = crop_original_by_box1000(test_original, box, expand=expand)
            else:
                raw = crop_original_by_center1000(test_original, center[0], center[1], crop_fraction)
            if raw is None:
                return
            imgs = [self._resize_one(raw, crop_size)]
            if include_ref:
                if reference_mode == 'matched':
                    ref_raw = self._matched_ref_crop(ref_original, center[0], center[1],
                                                    hmap, match_coordinates, crop_fraction)
                else:
                    ref_raw = crop_original_by_center1000(ref_original, center[0], center[1],
                                                          crop_fraction)
                if ref_raw is not None:
                    imgs.append(self._resize_one(ref_raw, crop_size))
            entries.append(imgs)

        if oracle_boxes:
            for box in oracle_boxes[:max_crops]:
                cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
                _entry([cx, cy], box=box)
        elif source == 'peaks':
            for qx, qy in self._zoom_peaks(hmap, zcfg):
                _entry([qx, qy])
        else:
            for prop in proposals[:max_crops]:
                peak = prop.get('peak_2d') or [(prop.get('bbox_2d') or [500, 500])[0],
                                               (prop.get('bbox_2d') or [500, 500])[1]]
                _entry([float(peak[0]), float(peak[1])], box=prop.get('bbox_2d'))
        return entries, include_ref

    def __call__(self, batch):
        if len(batch) != 1:
            raise ValueError('OutcomeCollator expects one sample; group sampling happens downstream')
        item = batch[0]
        device = self._device()
        global_size = int((self.cfg.get('data') or {}).get('max_image_size', 448))
        pcfg = self.cfg.get('outcome', {}).get('prior', {})
        h_size = global_size
        if pcfg.get('h_image_size'):
            h_size = int(pcfg['h_image_size'])

        ref_g, test_g = self._align_pair_at(item['ref'], item['test'], global_size)
        initial = self._concat_image_tensors(ref_g, test_g)

        if h_size != global_size:
            with pixel_budget(self.processor, h_size):
                ref_h, test_h = self._align_pair_at(item['ref'], item['test'], h_size)
                h_in = self._concat_image_tensors(ref_h, test_h)
            vis_h = encode_pair_canonical(
                self.prior, h_in['pixel_values'].to(device), h_in['image_grid_thw'].to(device))
            vis_g_merged = encode_visual_merged(
                self.prior, initial['pixel_values'].to(device), initial['image_grid_thw'].to(device))
        else:
            vis_h = encode_pair_canonical(
                self.prior, initial['pixel_values'].to(device), initial['image_grid_thw'].to(device))
            vis_g_merged = vis_h['merged_embeddings'].detach()

        hmap_real = vis_h['patch_map']
        # H-VPT cross-attn map: the ONLY H channel into the LLM. ``condition`` is
        # real/shuffled/zero (causal ablation). Proposals / region tokens / hstat /
        # peak text are all removed — H reaches the LLM only through the VPT-style
        # control-token cross-attention.
        h_vpt_cfg = self.cfg.get('outcome', {}).get('h_vpt', {}) or {}
        h_vpt_map = None
        if bool(h_vpt_cfg.get('enabled', False)):
            cond = str(h_vpt_cfg.get('condition', 'real'))
            if cond not in ('real', 'shuffled', 'zero'):
                raise ValueError(f'unknown h_vpt.condition: {cond}')
            if cond == 'real':
                h_vpt_map = hmap_real
            elif cond == 'shuffled':
                key = f"hvpt:{self.cfg['training']['seed']}:{item['image_path']}:{item['ref_path']}"
                s = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
                perm = torch.randperm(hmap_real.numel(), generator=torch.Generator().manual_seed(s))
                h_vpt_map = hmap_real.flatten()[perm.to(hmap_real.device)].reshape_as(hmap_real)
            else:
                h_vpt_map = torch.zeros_like(hmap_real)

        # H-Box geometry-token prior: H connected components -> candidate boxes ->
        # geometry vectors injected as STATIC tokens in [localize]. This is the
        # "search prior" channel (extent signal), parallel to h_vpt's dense features.
        h_box_cfg = self.cfg.get('outcome', {}).get('h_box_prior', {}) or {}
        h_box_geom = None
        box_proposals = []
        if bool(h_box_cfg.get('enabled', False)):
            cond = str(h_box_cfg.get('condition', 'real'))
            if cond not in ('real', 'shuffled', 'zero'):
                raise ValueError(f'unknown h_box_prior.condition: {cond}')
            k_boxes = int(h_box_cfg.get('max_boxes', pcfg.get('max_candidates', 3)))
            box_proposals, _, _ = region_proposals(hmap_real, pcfg)
            h_min = float(hmap_real.min())
            h_max = float(hmap_real.max())
            if cond == 'zero':
                geom = np.zeros((k_boxes, GEOM_DIM), dtype=np.float32)
            else:
                geom = geometry_from_proposals(box_proposals, k_boxes, h_min, h_max)
                if cond == 'shuffled':
                    key = f"hbox:{self.cfg['training']['seed']}:{item['image_path']}:{item['ref_path']}"
                    s = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
                    geom = geom[np.random.default_rng(s).permutation(k_boxes)]
            h_box_geom = torch.from_numpy(geom)  # [K, GEOM_DIM]

        images = [ref_g, test_g]
        caches = [vis_g_merged]
        grids = [initial['image_grid_thw']]
        text = render_prompt(self.cfg, item['class_name'], region_tokens='')

        user = dict(role='user', content=[dict(type='image', image=im) for im in images]+[dict(type='text', text=text)])
        enable_thinking = bool((self.cfg.get('prompt') or {}).get('enable_thinking', False))
        rendered = apply_chat_template_safe(self.processor, [user], True, enable_thinking)
        full = self.processor(text=[rendered], images=images, return_tensors='pt', truncation=False)
        length = full['input_ids'].shape[-1]
        maximum = int(self.cfg['training']['max_length'])
        if length > maximum:
            raise ValueError(f'prompt has {length} tokens > {maximum}; increase max_length or reduce image budget; refusing silent truncation')
        expected_grid = torch.cat(grids).cpu()
        if not torch.equal(full['image_grid_thw'].reshape(-1, 3).cpu(), expected_grid):
            raise RuntimeError('processor geometry changed between vision cache and prompt construction')
        full['image_embeds'] = torch.cat(caches)
        full['prompt_len'] = torch.tensor([length])
        full['control_token_id'] = int(control_token_id_of(self.processor))
        full['feat_token_id'] = int(feat_token_id_of(self.processor))
        full['n_feat_tokens'] = int(h_vpt_cfg.get('n_tokens', 64))
        if h_vpt_map is not None:
            full['h_map'] = h_vpt_map.detach().cpu()
        full['box_token_id'] = int(box_token_id_of(self.processor))
        full['n_box_tokens'] = int(h_box_cfg.get('max_boxes', 3)) if h_box_geom is not None else 0
        if h_box_geom is not None:
            full['h_box_geom'] = h_box_geom
        full['_meta'] = [{key: item.get(key) for key in ('orig_size','gt_box_px','is_anomaly','image_path','ref_path','class_name','defect_type','component_bboxes','num_components','mask_area_fraction','union_area_fraction','full_mask_path')}]
        full['_meta'][0].update(h_min=float(hmap_real.min()), h_max=float(hmap_real.max()),
                                image_count=len(images), prompt_tokens=int(length),
                                visual_tokens=int((full['image_grid_thw'].prod(-1)//(self.prior.spatial_merge_size**2)).sum()),
                                h_size=h_size,
                                prior_candidates=box_proposals,
                                prior_condition=str(h_box_cfg.get('condition', 'real')))
        # Visualization payload (heatmap + original images).
        full['_meta'][0].update(
            ref=item['ref'],
            test=item['test'],
            heatmap=heatmap_to_pil(hmap_real, item['test'].size),
        )
        return full
