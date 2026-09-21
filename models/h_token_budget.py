"""Training-free H-guided selection of already projected image tokens.

H affects input allocation only. No GT, extra embeddings, feature amplification,
or textual hint is used. Selected tokens retain their original raster ordering.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def merged_scores(hmap: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    """Map pre-merger H to the merger grid, preserving local peaks."""
    if hmap.ndim != 2 or not torch.isfinite(hmap).all():
        raise ValueError('expected a finite 2D H map')
    h, w = map(int, grid_hw)
    if h <= 0 or w <= 0:
        raise ValueError('invalid merger grid')
    return F.adaptive_max_pool2d(hmap.float()[None, None], (h, w))[0, 0]


def uniform_indices(hw: tuple[int, int], budget: int) -> torch.Tensor:
    """Stratified raster samples with a guaranteed exact count (CPU indices)."""
    h, w = map(int, hw)
    n = h*w
    if not 1 <= budget <= n:
        raise ValueError('budget must be in [1, number of tokens]')
    # Make roughly square spatial strata. Fill residual slots by farthest coverage.
    rows = min(h, max(1, int(math.sqrt(budget*h/w))))
    cols = min(w, max(1, budget//rows))
    ys = ((torch.arange(rows, dtype=torch.float64)+.5)*h/rows).long()
    xs = ((torch.arange(cols, dtype=torch.float64)+.5)*w/cols).long()
    selected = (ys[:, None]*w+xs).flatten().tolist()[:budget]
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    coords = torch.stack((yy.flatten()/h, xx.flatten()/w), -1)
    dist = torch.cdist(coords, coords[selected]).amin(-1)
    while len(selected) < budget:
        dist[selected] = -1
        idx = int(dist.argmax())
        selected.append(idx)
        dist = torch.minimum(dist, (coords-coords[idx]).square().sum(-1).sqrt())
    return torch.tensor(sorted(selected), dtype=torch.long)


def select_tokens(hmap: torch.Tensor, grid_hw: tuple[int,int], budget: int,
                  mode: str = 'h', global_fraction: float = .5,
                  halo_weight: float = .25) -> tuple[torch.Tensor, dict]:
    h, w = grid_hw
    n = h*w
    budget = min(int(budget), n)
    if budget < 1 or not 0 <= global_fraction <= 1 or not 0 <= halo_weight <= 1:
        raise ValueError('invalid token selection configuration')
    if mode not in ('full', 'uniform', 'h', 'shuffled'):
        raise ValueError(f'unknown selection mode: {mode}')
    if mode == 'full':
        return torch.arange(n), dict(mode=mode, before=n, after=n)
    score = merged_scores(hmap.detach().cpu(), grid_hw)
    if mode == 'shuffled':
        # Reproducible negative control, never uses GT.
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(9321))
        score = score.flatten()[perm].reshape(h,w)
    if mode == 'uniform' or float(score.max()-score.min()) < 1e-8:
        return uniform_indices(grid_hw,budget), dict(mode=mode,before=n,after=budget,flat_fallback=True)
    base_count = max(1, min(budget, math.ceil(budget*global_fraction)))
    selected = uniform_indices(grid_hw,base_count).tolist()
    # Nearby context helps preserve boundaries; the original peak signal dominates.
    halo = F.max_pool2d(score[None,None],3,stride=1,padding=1)[0,0]
    utility = ((1-halo_weight)*score+halo_weight*halo).flatten()
    utility[selected] = -torch.inf
    chosen = torch.argsort(utility,descending=True,stable=True)[:budget-base_count].tolist()
    idx = torch.tensor(sorted(selected+chosen),dtype=torch.long)
    assert idx.unique().numel() == budget
    return idx, dict(mode=mode,before=n,after=budget,global_tokens=base_count,
                     h_min=float(score.min()),h_max=float(score.max()))


@torch.no_grad()
def pack_sparse_pair(model, batch, merged, hmap, mode='h', keep_ratio=.5,
                     global_fraction=.5, halo_weight=.25):
    """Single pair, no padding: retain original 3D RoPE, compact only sequence slots.

    The frozen merger has already run. This API never changes image_grid_thw to
    pretend the sparse tokens form a smaller regular image. Instead, image features
    are embedded explicitly and positions are computed BEFORE token selection.
    """
    ids = batch['input_ids']
    if ids.shape[0] != 1 or not bool(batch['attention_mask'].all()):
        raise ValueError('sparse pair API supports one unpadded sample')
    if not 0 < keep_ratio <= 1:
        raise ValueError('keep_ratio must be in (0,1]')
    core = model.get_base_model() if hasattr(model,'get_base_model') else model
    inner = core.model
    grids = batch['image_grid_thw']
    if grids.shape != (2,3) or not bool((grids[:,0] == 1).all()):
        raise ValueError('expected exactly two still images')
    merge = int(core.config.vision_config.spatial_merge_size)
    counts = (grids.prod(-1)//merge**2).tolist()
    image_positions = (ids[0] == core.config.image_token_id).nonzero().flatten()
    if image_positions.numel() != sum(counts) or merged.shape[0] != sum(counts):
        raise ValueError('image placeholders and original merger features must match')
    types = batch.get('mm_token_type_ids')
    if types is None:
        types = (ids == core.config.image_token_id).long()
    position_ids, _ = inner.get_rope_index(ids, mm_token_type_ids=types,
                        image_grid_thw=grids, attention_mask=batch['attention_mask'])
    embeds = core.get_input_embeddings()(ids)
    embeds[:, image_positions] = merged.to(embeds)
    hw = tuple((grids[1,1:]//merge).tolist())
    keep_test, stats = select_tokens(hmap,hw,max(1,math.ceil(counts[1]*keep_ratio)),
                                     mode,global_fraction,halo_weight)
    test_positions = image_positions[counts[0]:]
    keep = torch.ones(ids.shape[-1],device=ids.device,dtype=torch.bool)
    keep[test_positions] = False
    keep[test_positions[keep_test.to(ids.device)]] = True
    selected = keep.nonzero().flatten()
    stats.update(prompt_before=ids.shape[-1],prompt_after=int(selected.numel()),
                 reference_tokens=counts[0],test_indices=keep_test.tolist(),grid_hw=hw)
    return dict(inputs_embeds=embeds[:,selected],position_ids=position_ids[:,:,selected],
                next_position=int(position_ids.max())+1,sequence_indices=selected,
                selection=stats)


@torch.no_grad()
def greedy_decode_sparse(model, packed, tokenizer, max_new_tokens=192):
    """Decode with explicit original RoPE coordinates and compact cache lengths.

    Avoid implicit generation position reconstruction from the obsolete image grid.
    No sampling differences across the allocation conditions.
    """
    device=packed['inputs_embeds'].device
    n=packed['inputs_embeds'].shape[1]
    out=model(inputs_embeds=packed['inputs_embeds'],position_ids=packed['position_ids'],
              attention_mask=torch.ones(1,n,device=device,dtype=torch.long),
              use_cache=True,logits_to_keep=1)
    ends=getattr(model.generation_config,'eos_token_id',None)
    ends=set(ends if isinstance(ends,list) else [ends])
    ends.add(tokenizer.eos_token_id)
    result=[]
    for step in range(max_new_tokens):
        token=int(out.logits[0,-1].argmax())
        result.append(token)
        if token in ends:break
        if step+1 == max_new_tokens:break
        past=out.past_key_values
        del out
        out=model(input_ids=torch.tensor([[token]],device=device),
                  position_ids=torch.full((3,1,1),packed['next_position']+step,device=device,dtype=torch.long),
                  attention_mask=torch.ones(1,n+step+1,device=device,dtype=torch.long),
                  past_key_values=past,use_cache=True,logits_to_keep=1)
    return dict(text=tokenizer.decode(result,skip_special_tokens=True),ids=result,
                truncated=bool(result and result[-1] not in ends))
