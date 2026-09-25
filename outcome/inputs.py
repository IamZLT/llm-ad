"""Shared full-image input construction with a static H candidate-box prior.

The H (anomaly prior) heatmap is no longer injected dynamically (VPT cross-attn /
H-Box geometry tokens are gone). It is reduced to a *static search hint*: its
connected-component candidate boxes are rendered as text and placed in the prompt
once, and the model verifies them itself in ``[confirm]`` (world-model style).
"""
from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.prior_dataset import PriorCollator, PriorCoTDataset, apply_chat_template_safe
from models.anomaly_prior import fuse_maps, heatmap_to_pil, softmax_fuse_maps, unpack_merge_order
from models.h_box_prior import GEOM_DIM, box_token_id_of, format_h_box_tokens, geometry_from_proposals
from models.h_vpt import CONTROL_TOKEN, FEAT_TOKEN, control_token_id_of, feat_token_id_of
from models.qwen35 import qwen_vision_factor
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
    high-H neighbourhood.

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


def format_h_candidates(proposals, max_candidates: int = None) -> str:
    """Render H connected-component boxes as static search-hint text ([0,1000]).

    These are *fallible* location/extent hints — some may be false alarms — never
    labels. The model uses them to seed ``[localize]`` and then verifies them in
    ``[confirm]`` (predict / observe / error, world-model style).
    """
    boxes = [p.get('bbox_2d') for p in (proposals or [])]
    if max_candidates is not None:
        boxes = boxes[:max(0, int(max_candidates))]
    if not boxes:
        return '[]'
    return json.dumps(boxes)


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

    def __call__(self, batch):
        if len(batch) != 1:
            raise ValueError('OutcomeCollator expects one sample; group sampling happens downstream')
        item = batch[0]
        device = self._device()
        global_size = int((self.cfg.get('data') or {}).get('max_image_size', 448))
        pcfg = self.cfg.get('outcome', {}).get('prior', {})

        ref_g, test_g = self._align_pair_at(item['ref'], item['test'], global_size)
        initial = self._concat_image_tensors(ref_g, test_g)

        # One joint vision forward → H patch map + merger tokens (no VPT/H-Box).
        vis_h = encode_pair_canonical(
            self.prior, initial['pixel_values'].to(device), initial['image_grid_thw'].to(device))
        hmap_real = vis_h['patch_map']
        vis_g_merged = vis_h['merged_embeddings'].detach()

        # Static H prior: connected-component candidate boxes as search-hint text,
        # plus (optionally) H-Box geometry tokens injected as <|h_box|> placeholders.
        proposals, _, _ = region_proposals(hmap_real, pcfg)
        h_candidates = format_h_candidates(proposals, int(pcfg.get('max_candidates', 3)))

        h_box_cfg = self.cfg.get('outcome', {}).get('h_box_prior', {}) or {}
        h_box_geom = None
        h_box_tokens = ''
        if bool(h_box_cfg.get('enabled', False)):
            k_boxes = int(h_box_cfg.get('max_boxes', pcfg.get('max_candidates', 3)))
            h_min = float(hmap_real.min())
            h_max = float(hmap_real.max())
            geom = geometry_from_proposals(proposals, k_boxes, h_min, h_max)
            h_box_geom = torch.from_numpy(geom)  # [K, GEOM_DIM]
            h_box_tokens = format_h_box_tokens(k_boxes)

        # H-VPT dense cross-attention: H patch map -> z_H -> control-token modulated
        # h_H, scattered into <|h_ctrl|> + N <|h_feat|> placeholder slots. This is the
        # dynamic H channel (parallel to the static H-Box geometry tokens).
        h_vpt_cfg = self.cfg.get('outcome', {}).get('h_vpt', {}) or {}
        h_vpt_map = None
        h_vpt_tokens = ''
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
            n_feat = int(h_vpt_cfg.get('n_tokens', 64))
            h_vpt_tokens = CONTROL_TOKEN + FEAT_TOKEN * n_feat

        images = [ref_g, test_g]
        caches = [vis_g_merged]
        grids = [initial['image_grid_thw']]
        text = render_prompt(self.cfg, item['class_name'], h_candidates=h_candidates,
                             h_box_tokens=h_box_tokens, h_vpt_tokens=h_vpt_tokens)

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
        # H-Box geometry-token prior: the [K, GEOM_DIM] raw geometry + the token id so
        # the policy/engine layers can compute g_proj and scatter it into <|h_box|> slots.
        full['box_token_id'] = int(box_token_id_of(self.processor)) if h_box_geom is not None else -1
        full['n_box_tokens'] = int(h_box_cfg.get('max_boxes', 3)) if h_box_geom is not None else 0
        if h_box_geom is not None:
            full['h_box_geom'] = h_box_geom
        # H-VPT dynamic channel: control/feat token ids + the H patch map so the
        # policy/engine can probe h_ctrl and scatter the modulated h_H.
        full['control_token_id'] = int(control_token_id_of(self.processor))
        full['feat_token_id'] = int(feat_token_id_of(self.processor))
        full['n_feat_tokens'] = int(h_vpt_cfg.get('n_tokens', 64))
        if h_vpt_map is not None:
            full['h_map'] = h_vpt_map.detach().cpu()
        full['_meta'] = [{key: item.get(key) for key in ('orig_size','gt_box_px','is_anomaly','image_path','ref_path','class_name','defect_type','component_bboxes','num_components','mask_area_fraction','union_area_fraction','full_mask_path')}]
        full['_meta'][0].update(h_min=float(hmap_real.min()), h_max=float(hmap_real.max()),
                                image_count=len(images), prompt_tokens=int(length),
                                visual_tokens=int((full['image_grid_thw'].prod(-1)//(self.prior.spatial_merge_size**2)).sum()),
                                prior_candidates=proposals,
                                h_candidates_text=h_candidates)
        # Visualization payload (heatmap + original images).
        full['_meta'][0].update(
            ref=item['ref'],
            test=item['test'],
            heatmap=heatmap_to_pil(hmap_real, item['test'].size),
        )
        return full


def _smart_resize_image(image: Image.Image, max_size: int, factor: int,
                        min_pixels: int, max_pixels: int) -> Image.Image:
    resized, _, _ = smart_resize(image, max_size=max_size, factor=factor,
                                 min_pixels=min_pixels, max_pixels=max_pixels)
    return resized


def build_zoom_batch(processor, prior, cfg: dict, ref_img: Image.Image,
                     test_img: Image.Image, crop_img: Image.Image,
                     continuation_text: str, device, crop_min_pixels: int = None,
                     prefill_text: str = '') -> dict:
    """Build a three-image batch (ref + test + crop) for the zoom observation pass.

    The crop is re-encoded at high resolution (``crop_min_pixels`` raised so the
    small window is *upscaled* and its extent detail becomes readable), while ref
    and test keep the same global budget as the first (latent) pass. No H-Box /
    H-VPT injection here: the second pass verifies extent from the visible crop,
    not from the global prior.

    ``prefill_text`` (the stage-1 ``[understand][compare][localize]`` chain) is
    appended to the prompt as already-generated assistant content, so the model
    continues from ``[imagine]`` instead of restarting its think chain. ``prompt_len``
    is set past the prefill, so only the newly generated tokens are scored.
    """
    data = cfg.get('data') or {}
    max_size = int(data.get('max_image_size', 768))
    factor = qwen_vision_factor(processor, getattr(prior, 'visual', None))
    cap = max_size * max_size
    test_rs = _smart_resize_image(test_img, max_size, factor, 256 * 256, cap)
    ref_rs = ref_img.resize(test_rs.size, Image.Resampling.BICUBIC)
    if crop_min_pixels is None:
        crop_min_pixels = max(cap // 2, 256 * 256)
    crop_rs = _smart_resize_image(crop_img, max_size, factor, crop_min_pixels, cap)

    img_proc = getattr(processor, 'image_processor', None)
    if img_proc is None:
        raise RuntimeError('processor.image_processor is required for the zoom pass')
    enc = img_proc(images=[ref_rs, test_rs, crop_rs], return_tensors='pt')

    # Merger tokens only (no H hooks): the second pass re-observes the crop.
    vis = encode_visual_merged(prior, enc['pixel_values'].to(device),
                               enc['image_grid_thw'].to(device))

    user = dict(role='user', content=[
        dict(type='image', image=ref_rs),
        dict(type='image', image=test_rs),
        dict(type='image', image=crop_rs),
        dict(type='text', text=continuation_text),
    ])
    enable_thinking = bool((cfg.get('prompt') or {}).get('enable_thinking', False))
    rendered = apply_chat_template_safe(processor, [user], True, enable_thinking)
    combined = rendered + (prefill_text or '')
    full = processor(text=[combined], images=[ref_rs, test_rs, crop_rs],
                     return_tensors='pt', truncation=False)
    length = int(full['input_ids'].shape[-1])
    maximum = int(cfg['training']['max_length'])
    if length > maximum:
        raise ValueError(f'zoom prompt has {length} tokens > {maximum}; refusing silent truncation')
    full['image_embeds'] = vis
    full['prompt_len'] = torch.tensor([length])
    # No h_box_geom / h_map / control/feat token ids: the zoom pass is H-free.
    full['box_token_id'] = -1
    full['n_box_tokens'] = 0
    full['control_token_id'] = -1
    full['feat_token_id'] = -1
    full['n_feat_tokens'] = 0
    return full


ZOOM_PROMPT_SUFFIX = (
    'Image 3 is a zoomed crop around the candidate box: the red rectangle marks '
    'that candidate box, and the surrounding context reveals whether the box '
    'undershoots, overshoots, or is shifted relative to the true defect boundary. '
    'Use Image 3 to verify the box extent and correct it before finalizing your answer.'
)


def build_zoom_train_batch(processor, prior, cfg: dict, device, base_batch: dict,
                           crop_img: Image.Image) -> dict:
    """Upgrade a 2-image collator batch to a single-pass 3-image (ref+test+crop) batch.

    Unlike ``build_zoom_batch`` (the eval two-stage "continue from [imagine]" pass,
    H-free), this keeps the H fields from ``base_batch`` (``h_box_geom`` / ``h_map`` /
    control-feat-box token ids) so H-Box + H-VPT injection work unchanged in the
    ordinary single-pass rollout/actor forward. Only the vision stack and the prompt
    gain the third zoomed-and-outlined image. The prompt's H hint placeholders are
    re-rendered from the base batch's token counts so the injected tokens still line up.

    Used for both the SFT zoom trajectories (``train_region_sft.py``) and the RL
    zoom-training rollouts (``engine_multibox.py`` scheme B: a fixed shrink-GT
    candidate box supplies the crop, so the whole GRPO group shares one 3-image cache
    and the standard group rollout/optimize path is reused unchanged).
    """
    meta = base_batch['_meta'][0]
    ref_orig = meta['ref']
    test_orig = meta['test']
    class_name = str(meta.get('class_name') or 'object')
    h_candidates = str(meta.get('h_candidates_text') or '')

    data = cfg.get('data') or {}
    max_size = int(data.get('max_image_size', 768))
    factor = qwen_vision_factor(processor, getattr(prior, 'visual', None))
    cap = max_size * max_size
    test_rs, _, _ = smart_resize(test_orig, max_size=max_size, factor=factor,
                                 min_pixels=256 * 256, max_pixels=cap)
    ref_rs = ref_orig.resize(test_rs.size, Image.Resampling.BICUBIC)
    zcfg = (cfg.get('outcome') or {}).get('zoom') or {}
    crop_min_pixels = int(zcfg.get('crop_min_pixels') or max(cap // 2, 256 * 256))
    crop_rs, _, _ = smart_resize(crop_img, max_size=max_size, factor=factor,
                                 min_pixels=crop_min_pixels, max_pixels=cap)

    img_proc = getattr(processor, 'image_processor', None)
    if img_proc is None:
        raise RuntimeError('processor.image_processor is required for the zoom batch')
    enc = img_proc(images=[ref_rs, test_rs, crop_rs], return_tensors='pt')
    vis = encode_visual_merged(prior, enc['pixel_values'].to(device),
                               enc['image_grid_thw'].to(device))

    n_box = int(base_batch.get('n_box_tokens', 0) or 0)
    n_feat = int(base_batch.get('n_feat_tokens', 0) or 0)
    h_box_tokens = format_h_box_tokens(n_box) if n_box > 0 else ''
    h_vpt_tokens = (CONTROL_TOKEN + FEAT_TOKEN * n_feat) if n_feat > 0 else ''
    text = render_prompt(cfg, class_name, h_candidates=h_candidates,
                         h_box_tokens=h_box_tokens, h_vpt_tokens=h_vpt_tokens)
    text = text + '\n' + ZOOM_PROMPT_SUFFIX

    user = dict(role='user', content=[
        dict(type='image', image=ref_rs),
        dict(type='image', image=test_rs),
        dict(type='image', image=crop_rs),
        dict(type='text', text=text),
    ])
    enable_thinking = bool((cfg.get('prompt') or {}).get('enable_thinking', False))
    rendered = apply_chat_template_safe(processor, [user], True, enable_thinking)
    full = processor(text=[rendered], images=[ref_rs, test_rs, crop_rs],
                     return_tensors='pt', truncation=False)
    length = int(full['input_ids'].shape[-1])
    maximum = int(cfg['training']['max_length'])
    if length > maximum:
        raise ValueError(f'zoom prompt has {length} tokens > {maximum}; refusing silent truncation')
    full['image_embeds'] = vis
    full['prompt_len'] = torch.tensor([length])
    for key in ('box_token_id', 'n_box_tokens', 'control_token_id', 'feat_token_id', 'n_feat_tokens'):
        full[key] = base_batch.get(key, -1 if 'token' in key else 0)
    if 'h_box_geom' in base_batch:
        full['h_box_geom'] = base_batch['h_box_geom']
    if 'h_map' in base_batch:
        full['h_map'] = base_batch['h_map']
    # Refresh logging metadata to reflect the 3-image prompt (keep test/ref/heatmap
    # used by visualization).
    merge = int(getattr(getattr(prior, 'visual', None), 'spatial_merge_size', 2) or 2)
    meta_copy = dict(base_batch['_meta'][0])
    meta_copy['prompt_tokens'] = int(length)
    meta_copy['visual_tokens'] = int((full['image_grid_thw'].prod(-1) // (merge * merge)).sum())
    meta_copy['image_count'] = 3
    full['_meta'] = [meta_copy]
    from rl.grpo import move_batch
    return move_batch(full, device)
