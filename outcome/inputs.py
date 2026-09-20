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
from models.anomaly_prior import heatmap_to_pil, softmax_fuse_maps, unpack_merge_order
from models.qwen35 import qwen_vision_factor
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
    hmap = softmax_fuse_maps(stack, prior.temperature) if len(maps) > 1 else stack[0]
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


def format_region_hints(proposals, n_tokens: int, owners=None) -> str:
    """One ``<|region|>`` per token, grouped by H region.

    The first token of each region is ``hK:<|region|>@[x,y]`` (peak on Image 2).
    Later tokens of the same region are bare ``<|region|>`` and describe extent.
    """
    n = max(1, int(n_tokens))
    if not proposals:
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
                    valid=torch.zeros(1, 1, dtype=torch.bool, device=dev),
                    owner=torch.full((1, 1), -1, device=dev, dtype=torch.long))
    n = min(len(out_test), max_cells)
    hpatch = _extract_h_patches(h, out_geom[:n], patch_size)
    return dict(test=torch.stack(out_test[:n]).unsqueeze(0),
                ref=torch.stack(out_ref[:n]).unsqueeze(0),
                geom=torch.tensor(out_geom[:n], device=dev, dtype=torch.float32).unsqueeze(0),
                hstat=torch.tensor(out_hstat[:n], device=dev, dtype=torch.float32).unsqueeze(0),
                hpatch=hpatch.unsqueeze(0),  # [1, n, 1, P, P]
                valid=torch.ones(1, n, dtype=torch.bool, device=dev),
                owner=torch.tensor(out_owner[:n], device=dev, dtype=torch.long).unsqueeze(0))


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
        zcfg = self._zoom_cfg()
        zoom_on = bool(zcfg.get('enabled', False))
        global_size = int((self.cfg.get('data') or {}).get('max_image_size', 448))
        # H resolution is decoupled from crop zoom: `prior.h_image_size` (new,
        # explicit) wins; `zoom.h_image_size` remains a legacy alias that only
        # applies when zoom is on (preserves old configs' behavior).
        pcfg = self.cfg.get('outcome', {}).get('prior', {})
        h_size = global_size
        if pcfg.get('h_image_size'):
            h_size = int(pcfg['h_image_size'])
        elif zoom_on and zcfg.get('h_image_size'):
            h_size = int(zcfg['h_image_size'])

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

        pcfg = self.cfg.get('outcome', {}).get('prior', {})
        hmap = vis_h['patch_map']
        hmap_real = hmap  # real, un-shuffled H (kept for the cross-attn memory ablation)
        condition = pcfg.get('condition', 'real')
        if condition == 'shuffled':
            # Fixed per sample and across all members of its group; no GT involved.
            key = f"{self.cfg['training']['seed']}:{item['image_path']}:{item['ref_path']}"
            seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
            perm = torch.randperm(hmap.numel(), generator=torch.Generator().manual_seed(seed))
            hmap = hmap.flatten()[perm.to(hmap.device)].reshape_as(hmap)
        if condition not in ('real', 'none', 'shuffled'):
            raise ValueError(f'unknown H condition: {condition}')
        # H-memory (cross-attn) input: independent of prior.condition so proposals/
        # RegionAdapter always use the real H while only the memory is perturbed.
        h_mem_cfg = self.cfg.get('outcome', {}).get('h_memory', {}) or {}
        h_mem_map = None
        if bool(h_mem_cfg.get('enabled', False)):
            mem_cond = str(h_mem_cfg.get('condition', 'real'))
            if mem_cond not in ('real', 'shuffled', 'zero'):
                raise ValueError(f'unknown h_memory.condition: {mem_cond}')
            if mem_cond == 'real':
                h_mem_map = hmap_real
            elif mem_cond == 'shuffled':
                key = f"hmem:{self.cfg['training']['seed']}:{item['image_path']}:{item['ref_path']}"
                s = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
                perm = torch.randperm(hmap_real.numel(), generator=torch.Generator().manual_seed(s))
                h_mem_map = hmap_real.flatten()[perm.to(hmap_real.device)].reshape_as(hmap_real)
            else:
                h_mem_map = torch.zeros_like(hmap_real)
        proposals, proposal_masks, threshold_mode = region_proposals(hmap, pcfg)
        if condition == 'none':
            proposals = []
            proposal_masks = np.zeros((0, *proposal_masks.shape[1:]), dtype=bool)
        region_cfg = self.cfg.get('outcome', {}).get('region', {}) or {}
        h_prior_cfg = self.cfg.get('outcome', {}).get('h_prior_adapter') or {}
        h_patch_size = int(h_prior_cfg.get('patch_size', 7))
        images = [ref_g, test_g]
        caches = [vis_g_merged]
        grids = [initial['image_grid_thw']]
        if vis_h.get('test_features') is None or vis_h.get('matched_ref_features') is None:
            raise ValueError('region tokens require matched features; check prior.region_feature_index')
        region_raw = extract_region_cells(proposal_masks, vis_h['test_features'],
                                          vis_h['matched_ref_features'], hmap, region_cfg,
                                          patch_size=h_patch_size)
        # HAdapter side-channel ablation: keep proposals/geom/hstat from real H but
        # perturb ONLY the H patch fed to HPriorAdapter. `h_prior_adapter.condition`
        # is independent of `prior.condition` so we can test the side channel's use of
        # H's *spatial structure* without changing proposal quality.
        h_prior_cond = str(h_prior_cfg.get('condition', 'real'))
        if h_prior_cond not in ('real', 'shuffled', 'zero'):
            raise ValueError(f'unknown h_prior_adapter.condition: {h_prior_cond}')
        if h_prior_cond != 'real' and h_patch_size > 0:
            hp_map = vis_h['patch_map']  # the real, un-shuffled H map
            if h_prior_cond == 'shuffled':
                key = f"hprior:{self.cfg['training']['seed']}:{item['image_path']}:{item['ref_path']}"
                s = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
                perm = torch.randperm(hp_map.numel(), generator=torch.Generator().manual_seed(s))
                hp_map = hp_map.flatten()[perm.to(hp_map.device)].reshape_as(hp_map)
            elif h_prior_cond == 'zero':
                hp_map = torch.zeros_like(hp_map)
            geoms = region_raw['geom'][0].tolist()
            region_raw['hpatch'] = _extract_h_patches(hp_map, geoms, h_patch_size).unsqueeze(0)
        region_token_id = region_token_id_of(self.processor)
        n_region = int(region_raw['valid'].shape[1])
        owners = region_raw['owner'][0].tolist() if 'owner' in region_raw else None
        region_tokens = format_region_hints(proposals, n_region, owners)
        text = render_prompt(self.cfg, item['class_name'], region_tokens=region_tokens)

        crops = []
        paired = False
        n_entries = 0
        if zoom_on and condition != 'none':
            oracle_boxes = self._gt_oracle_boxes(item) if bool(zcfg.get('oracle', False)) else []
            entries, paired = self._build_zoom_crops(item['test'], item['ref'], proposals, hmap,
                                                     vis_h.get('match_coordinates'), oracle_boxes)
            n_entries = len(entries)
            crops = [img for entry in entries for img in entry]
            if crops:
                # Encode every crop under the global (LLM) pixel budget so processor grids match.
                crop_encs = [
                    getattr(self.processor, 'image_processor')(images=crop, return_tensors='pt')
                    for crop in crops
                ]
                def _pixels(t):
                    return t.reshape(-1, t.shape[-1]) if t.ndim == 3 else t
                def _grid(t):
                    if t.ndim == 1:
                        t = t.unsqueeze(0)
                    if t.ndim == 3:
                        t = t.reshape(-1, int(t.shape[-1]))
                    return t
                crop_pixels = torch.cat([_pixels(e['pixel_values']) for e in crop_encs], dim=0)
                crop_grid = torch.cat([_grid(e['image_grid_thw']) for e in crop_encs], dim=0)
                crop_merged = encode_visual_merged(
                    self.prior, crop_pixels.to(device), crop_grid.to(device))
                images.extend(crops)
                caches.append(crop_merged)
                grids.append(crop_grid)
                ref_mode = str(zcfg.get('reference_mode', 'matched'))
                text = text + zoom_prompt_suffix(n_entries, paired=paired, reference_mode=ref_mode)

        user = dict(role='user', content=[dict(type='image', image=im) for im in images]+[dict(type='text', text=text)])
        enable_thinking = bool((self.cfg.get('prompt') or {}).get('enable_thinking', False))
        rendered = apply_chat_template_safe(self.processor, [user], True, enable_thinking)
        full = self.processor(text=[rendered], images=images, return_tensors='pt', truncation=False)
        length = full['input_ids'].shape[-1]
        maximum = int(self.cfg['training']['max_length'])
        if length > maximum:
            raise ValueError(f'prompt has {length} tokens > {maximum}; increase max_length or reduce image budget; refusing silent truncation')
        n_region = int(region_raw['valid'].shape[1])
        actual = int((full['input_ids'][0] == region_token_id).sum())
        if actual != n_region:
            raise RuntimeError(f'region tokenization mismatch: expected {n_region} <|region|> tokens, got {actual}')
        expected_grid = torch.cat(grids).cpu()
        if not torch.equal(full['image_grid_thw'].reshape(-1, 3).cpu(), expected_grid):
            raise RuntimeError('processor geometry changed between vision cache and prompt construction')
        full['image_embeds'] = torch.cat(caches)
        full['prompt_len'] = torch.tensor([length])
        full['region_test'] = region_raw['test'].cpu()
        full['region_ref'] = region_raw['ref'].cpu()
        full['region_geom'] = region_raw['geom'].cpu()
        full['region_hstat'] = region_raw['hstat'].cpu()
        full['region_hpatch'] = region_raw['hpatch'].cpu()
        full['region_valid'] = region_raw['valid'].cpu()
        full['region_token_id'] = int(region_token_id)
        if h_mem_map is not None:
            full['h_map'] = h_mem_map.detach().cpu()
        prior_hint_tokens = int(region_raw['valid'].shape[1])
        full['_meta'] = [{key: item.get(key) for key in ('orig_size','gt_box_px','is_anomaly','image_path','ref_path','class_name','defect_type','component_bboxes','num_components','mask_area_fraction','union_area_fraction','full_mask_path')}]
        full['_meta'][0].update(prior_candidates=proposals, prior_condition=condition,
                                prior_threshold_mode=threshold_mode, h_min=float(hmap.min()), h_max=float(hmap.max()),
                                image_count=len(images), prompt_tokens=int(length),
                                visual_tokens=int((full['image_grid_thw'].prod(-1)//(self.prior.spatial_merge_size**2)).sum()),
                                prior_hint_tokens=prior_hint_tokens,
                                zoom_enabled=zoom_on, zoom_h_size=h_size, zoom_n_crops=len(crops))
        # Visualization payload (heatmap + original images + H peaks in 0-1000).
        full['_meta'][0].update(
            ref=item['ref'],
            test=item['test'],
            heatmap=heatmap_to_pil(hmap, item['test'].size),
            prior_points=[p['peak_2d'] for p in proposals],
        )
        return full
