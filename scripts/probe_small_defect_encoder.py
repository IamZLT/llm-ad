#!/usr/bin/env python3
"""Do Qwen ViT features actually differ at small-defect GT vs the matched reference?

Geometry is computed for every small/medium/large eval anomaly. Encoder comparison
(H + cosine(test, matched-ref)) runs on the small set, at 448 and 768, without
touching the LLM.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.anomaly_prior import AnomalyPrior, get_qwen_visual
from models.qwen35 import qwen_vision_factor, setup_model_and_processor
from outcome.inputs import encode_pair_canonical, pixel_budget
from utils.common import smart_resize
from utils.config import load_yaml_config

ALIGN = 32  # patch 16 × merge 2


def mvtec_mask_path(image_path: str) -> Path | None:
    p = Path(image_path)
    # .../class/test/defect/000.png -> .../class/ground_truth/defect/000_mask.png
    parts = list(p.parts)
    try:
        i = parts.index('test')
    except ValueError:
        return None
    parts[i] = 'ground_truth'
    mask = Path(*parts)
    mask = mask.with_name(mask.stem + '_mask' + mask.suffix)
    return mask if mask.exists() else None


def size_bin(frac: float) -> str:
    if frac < 0.02:
        return 'small'
    if frac < 0.1:
        return 'medium'
    return 'large'


def load_mask(image_path: str, orig_size) -> np.ndarray:
    mp = mvtec_mask_path(image_path)
    w, h = int(orig_size[0]), int(orig_size[1])
    if mp is None:
        return np.zeros((h, w), dtype=bool)
    arr = np.array(Image.open(mp).convert('L'))
    if arr.shape[1] != w or arr.shape[0] != h:
        arr = np.array(Image.open(mp).convert('L').resize((w, h), Image.Resampling.NEAREST))
    return arr > 0


def geometry_at(mask: np.ndarray, canvas_wh: tuple[int, int]) -> dict:
    mh, mw = mask.shape
    cw, ch = canvas_wh
    mask_rs = np.array(
        Image.fromarray(mask.astype(np.uint8) * 255).resize((cw, ch), Image.Resampling.NEAREST)
    ) > 0
    cells_w, cells_h = cw // ALIGN, ch // ALIGN
    occ = np.zeros((cells_h, cells_w), dtype=np.float32)
    for y in range(cells_h):
        for x in range(cells_w):
            patch = mask_rs[y * ALIGN:(y + 1) * ALIGN, x * ALIGN:(x + 1) * ALIGN]
            if patch.size:
                occ[y, x] = float(patch.mean())
    hit = occ > 0
    n_hit = int(hit.sum())
    occupancies = occ[hit] if n_hit else np.zeros(0, dtype=np.float32)
    ys, xs = np.where(mask_rs)
    if ys.size:
        bw = int(xs.max() - xs.min() + 1)
        bh = int(ys.max() - ys.min() + 1)
    else:
        bw = bh = 0
    return dict(
        canvas_w=cw, canvas_h=ch, grid=f'{cells_h}x{cells_w}',
        mask_px=int(mask_rs.sum()),
        bbox_w=bw, bbox_h=bh,
        n_cells_hit=n_hit,
        max_occupancy=float(occupancies.max()) if n_hit else 0.0,
        mean_occupancy=float(occupancies.mean()) if n_hit else 0.0,
        n_cells_occ_ge_25=int((occupancies >= 0.25).sum()) if n_hit else 0,
        n_cells_occ_ge_50=int((occupancies >= 0.50).sum()) if n_hit else 0,
        diluted=bool(n_hit > 0 and float(occupancies.max()) < 0.25),
        sub_cell=bool(n_hit == 0 or (n_hit == 1 and float(occupancies.max()) < 0.25)),
        occupancy=occ,
        mask_rs=mask_rs,
    )


def summarize_geom(rows: list[dict], bin_name: str) -> dict:
    sub = [r for r in rows if r['bin'] == bin_name]
    if not sub:
        return {}

    def mean(key, size):
        return float(np.mean([r[size][key] for r in sub]))

    def frac(pred):
        return float(np.mean([pred(r) for r in sub]))

    return dict(
        n=len(sub),
        orig_mask_frac=float(np.mean([r['mask_frac'] for r in sub])),
        orig_min_bbox_side=float(np.mean([min(r['bbox_w'], r['bbox_h']) for r in sub])),
        at448=dict(
            mean_bbox= [mean('bbox_w', 'g448'), mean('bbox_h', 'g448')],
            mean_cells_hit=mean('n_cells_hit', 'g448'),
            mean_max_occ=mean('max_occupancy', 'g448'),
            frac_diluted=frac(lambda r: r['g448']['diluted']),
            frac_sub_cell=frac(lambda r: r['g448']['sub_cell']),
            frac_no_cell=frac(lambda r: r['g448']['n_cells_hit'] == 0),
            frac_ge25=frac(lambda r: r['g448']['n_cells_occ_ge_25'] >= 1),
            frac_ge50=frac(lambda r: r['g448']['n_cells_occ_ge_50'] >= 1),
        ),
        at768=dict(
            mean_bbox=[mean('bbox_w', 'g768'), mean('bbox_h', 'g768')],
            mean_cells_hit=mean('n_cells_hit', 'g768'),
            mean_max_occ=mean('max_occupancy', 'g768'),
            frac_diluted=frac(lambda r: r['g768']['diluted']),
            frac_sub_cell=frac(lambda r: r['g768']['sub_cell']),
            frac_ge25=frac(lambda r: r['g768']['n_cells_occ_ge_25'] >= 1),
            frac_ge50=frac(lambda r: r['g768']['n_cells_occ_ge_50'] >= 1),
        ),
    )


@torch.no_grad()
def encode_pair(prior, processor, ref: Image.Image, test: Image.Image, max_size: int, device):
    factor = qwen_vision_factor(processor, prior.visual)
    cap = max_size * max_size
    floor = min(256 * 256, cap)
    test_rs, _, _ = smart_resize(test, max_size=max_size, factor=factor, min_pixels=floor, max_pixels=cap)
    ref_rs = ref.resize(test_rs.size, Image.Resampling.BICUBIC)
    with pixel_budget(processor, max_size):
        img_proc = processor.image_processor
        ref_enc = img_proc(images=ref_rs, return_tensors='pt')
        test_enc = img_proc(images=test_rs, return_tensors='pt')

    def _pixels(t):
        return t.reshape(-1, t.shape[-1]) if t.ndim == 3 else t

    def _grid(t):
        if t.ndim == 1:
            t = t.unsqueeze(0)
        if t.ndim == 3:
            t = t.reshape(-1, int(t.shape[-1]))
        return t

    pixels = torch.cat([_pixels(ref_enc['pixel_values']), _pixels(test_enc['pixel_values'])], 0)
    grid = torch.cat([_grid(ref_enc['image_grid_thw']), _grid(test_enc['image_grid_thw'])], 0)
    vis = encode_pair_canonical(prior, pixels.to(device), grid.to(device))
    return vis, test_rs.size


def cell_stats(hmap, test_f, ref_f, occ, occ_thr=0.05):
    ht, wt = hmap.shape
    if occ.shape != (ht, wt):
        occ_img = Image.fromarray((occ * 255).astype(np.uint8))
        occ = np.array(occ_img.resize((wt, ht), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    pos = occ >= occ_thr
    neg = ~pos
    h = hmap
    tf = test_f.float().reshape(ht * wt, -1)
    rf = ref_f.float().reshape(ht * wt, -1)
    tf_n = torch.nn.functional.normalize(tf, dim=-1)
    rf_n = torch.nn.functional.normalize(rf, dim=-1)
    cos = (tf_n * rf_n).sum(-1).reshape(ht, wt).cpu().numpy()
    l2 = (tf - rf).norm(dim=-1).reshape(ht, wt).cpu().numpy()

    def take(arr, m):
        v = arr[m]
        if v.size == 0:
            return dict(n=0, mean=None, max=None)
        return dict(n=int(v.size), mean=float(v.mean()), max=float(v.max()))

    h_pos, h_neg = take(h, pos), take(h, neg)
    # How often is the global H peak inside the GT cells?
    peak = np.unravel_index(int(h.argmax()), h.shape)
    peak_in_gt = bool(pos[peak]) if pos.any() else False
    # Rank of best GT cell among all cells (1 = H peak is a GT cell)
    order = np.argsort(-h.reshape(-1))
    if pos.any():
        pos_flat = pos.reshape(-1)
        best_gt_rank = int(np.where(pos_flat[order])[0][0]) + 1
    else:
        best_gt_rank = None
    # Cell-level AUROC of H vs occupancy>thr
    if pos.any() and neg.any():
        scores = np.concatenate([h[pos], h[neg]])
        labels = np.concatenate([np.ones(pos.sum()), np.zeros(neg.sum())])
        # Mann-Whitney AUROC
        ranks = scores.argsort().argsort() + 1
        n_pos, n_neg = int(pos.sum()), int(neg.sum())
        auroc = float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
        # wait, ranks of positives: need ranks of pos samples
        pos_ranks = ranks[labels.astype(bool)] if False else None
        # correct AUROC:
        order_s = np.argsort(scores)
        labels_s = labels[order_s]
        # Wilcoxon-Mann-Whitney
        n1 = n_pos
        rank_sum = (np.arange(1, len(scores) + 1)[labels_s.astype(bool)]).sum()
        # labels_s after argsort is increasing score; ranks 1=lowest
        # AUROC = (rank_sum_pos - n1(n1+1)/2) / (n1*n2) with rank 1 = lowest
        auroc = float((rank_sum - n1 * (n1 + 1) / 2) / (n1 * n_neg))
    else:
        auroc = None
    sep = None
    if h_pos['mean'] is not None and h_neg['mean'] is not None and h_neg['mean'] > 1e-8:
        sep = h_pos['mean'] / h_neg['mean']
    return dict(
        h_gt=h_pos, h_bg=h_neg,
        cos_gt=take(cos, pos), cos_bg=take(cos, neg),
        l2_gt=take(l2, pos), l2_bg=take(l2, neg),
        peak_in_gt=peak_in_gt, best_gt_rank=best_gt_rank, auroc=auroc,
        h_sep=sep, n_pos=int(pos.sum()), n_neg=int(neg.sum()),
    )


def mean_or_none(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/qwen35_2b_outcome_multibox.yaml')
    ap.add_argument('--jsonl', default='outputs/train/qwen35_2b_outcome_multibox/train_20260913_053232_599794/test_000200.jsonl')
    ap.add_argument('--zoom-jsonl', default='outputs/train/qwen35_2b_outcome_multibox/train_zoom_h768_a400/test_000200.jsonl')
    ap.add_argument('--out', default='outputs/eval/small_defect_encoder_probe.json')
    ap.add_argument('--max-small-encode', type=int, default=63)
    ap.add_argument('--skip-encoder', action='store_true')
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.jsonl).read_text().splitlines() if l.strip()]
    anomalies = [r for r in rows if r.get('is_anomaly')]
    print(f'eval anomalies: {len(anomalies)}', flush=True)

    geom_rows = []
    for r in anomalies:
        img = Image.open(r['image_path']).convert('RGB')
        mask = load_mask(r['image_path'], img.size)
        frac = float(mask.mean())
        ys, xs = np.where(mask)
        bw = int(xs.max() - xs.min() + 1) if xs.size else 0
        bh = int(ys.max() - ys.min() + 1) if ys.size else 0
        # canvases used by the model
        t448, _, _ = smart_resize(img, max_size=448, factor=ALIGN, min_pixels=256 * 256, max_pixels=448 * 448)
        t768, _, _ = smart_resize(img, max_size=768, factor=ALIGN, min_pixels=256 * 256, max_pixels=768 * 768)
        g448 = geometry_at(mask, t448.size)
        g768 = geometry_at(mask, t768.size)
        # drop bulky arrays from stored geom
        g448_s = {k: v for k, v in g448.items() if k not in ('occupancy', 'mask_rs')}
        g768_s = {k: v for k, v in g768.items() if k not in ('occupancy', 'mask_rs')}
        geom_rows.append(dict(
            class_name=r['class_name'], image_path=r['image_path'], ref_path=r['ref_path'],
            mask_frac=frac, bin=size_bin(frac), bbox_w=bw, bbox_h=bh,
            orig_wh=list(img.size),
            mask_iou=r.get('mask_iou'), iou_h_bestk=r.get('iou_h_bestk'),
            g448=g448_s, g768=g768_s,
            _occ448=g448['occupancy'], _occ768=g768['occupancy'],
        ))

    geom_summary = {b: summarize_geom(geom_rows, b) for b in ('small', 'medium', 'large')}
    print(json.dumps(geom_summary, indent=2), flush=True)

    zoom_h = {}
    zp = Path(args.zoom_jsonl)
    if zp.exists():
        for line in zp.read_text().splitlines():
            z = json.loads(line)
            if z.get('is_anomaly'):
                zoom_h[z['image_path']] = z.get('iou_h_bestk')

    # encoder
    enc = {}
    if not args.skip_encoder:
        cfg = load_yaml_config(args.config)
        cfg['outcome']['zoom'] = dict(enabled=False)
        device = torch.device('cpu')
        print(f'loading vision on {device}', flush=True)
        model, processor = setup_model_and_processor(cfg, for_inference=True, freeze_vision=True)
        model = model.to(device).eval()
        prior = AnomalyPrior.from_qwen(model, cfg)
        # drop LLM weights from RAM after grabbing visual
        visual = get_qwen_visual(model)
        prior.visual = visual
        del model
        small = [g for g in geom_rows if g['bin'] == 'small'][: args.max_small_encode]
        # also a few medium as control
        medium = [g for g in geom_rows if g['bin'] == 'medium'][:12]
        targets = small + medium
        per = []
        for i, g in enumerate(targets):
            test = Image.open(g['image_path']).convert('RGB')
            ref = Image.open(g['ref_path']).convert('RGB')
            rec = dict(class_name=g['class_name'], image_path=g['image_path'], bin=g['bin'],
                       mask_frac=g['mask_frac'], bbox_w=g['bbox_w'], bbox_h=g['bbox_h'])
            for size, occ in ((448, g['_occ448']), (768, g['_occ768'])):
                vis, canvas = encode_pair(prior, processor, ref, test, size, device)
                h = vis['patch_map'].float().cpu().numpy()
                st = cell_stats(h, vis['test_features'], vis['matched_ref_features'], occ)
                rec[f's{size}'] = dict(
                    canvas=list(canvas),
                    grid=list(h.shape),
                    **{k: v for k, v in st.items()},
                )
                print(
                    f"[{i+1}/{len(targets)}] {g['class_name']} {g['bin']} @{size} "
                    f"Hgt={st['h_gt']['mean']} Hbg={st['h_bg']['mean']} "
                    f"cos_gt={st['cos_gt']['mean']} cos_bg={st['cos_bg']['mean']} "
                    f"peak_in_gt={st['peak_in_gt']} auroc={st['auroc']}",
                    flush=True,
                )
            per.append(rec)

        def pool(items, prefix, key_path):
            vals = []
            for it in items:
                cur = it.get(prefix)
                if not cur:
                    continue
                for k in key_path.split('.'):
                    cur = cur.get(k) if isinstance(cur, dict) else None
                    if cur is None:
                        break
                if cur is not None:
                    vals.append(float(cur))
            return mean_or_none(vals)

        enc = {}
        for b in ('small', 'medium'):
            items = [p for p in per if p['bin'] == b]
            enc[b] = {}
            for size in (448, 768):
                p = f's{size}'
                enc[b][size] = dict(
                    n=len(items),
                    h_gt=pool(items, p, 'h_gt.mean'),
                    h_bg=pool(items, p, 'h_bg.mean'),
                    h_sep=pool(items, p, 'h_sep'),
                    cos_gt=pool(items, p, 'cos_gt.mean'),
                    cos_bg=pool(items, p, 'cos_bg.mean'),
                    l2_gt=pool(items, p, 'l2_gt.mean'),
                    l2_bg=pool(items, p, 'l2_bg.mean'),
                    peak_in_gt=pool(items, p, 'peak_in_gt'),
                    auroc=pool(items, p, 'auroc'),
                    best_gt_rank=pool(items, p, 'best_gt_rank'),
                )

    # attach zoom H vs baseline H for small
    small_paths = [g['image_path'] for g in geom_rows if g['bin'] == 'small']
    base_h = [g['iou_h_bestk'] for g in geom_rows if g['bin'] == 'small' and g['iou_h_bestk'] is not None]
    z_h = [zoom_h[p] for p in small_paths if p in zoom_h and zoom_h[p] is not None]
    h_cmp = dict(
        small_n=len(small_paths),
        baseline_h_bestk=mean_or_none(base_h),
        zoom_h_bestk=mean_or_none(z_h),
        baseline_h_ge03=float(np.mean([x >= 0.3 for x in base_h])) if base_h else None,
        zoom_h_ge03=float(np.mean([x >= 0.3 for x in z_h])) if z_h else None,
    )

    slim_cases = []
    for g in geom_rows:
        if g['bin'] != 'small':
            continue
        slim_cases.append(dict(
            class_name=g['class_name'],
            image=Path(g['image_path']).as_posix().split('/')[-4:],
            mask_frac=round(g['mask_frac'], 5),
            orig_bbox=[g['bbox_w'], g['bbox_h']],
            cells448=g['g448']['n_cells_hit'],
            max_occ448=round(g['g448']['max_occupancy'], 3),
            cells768=g['g768']['n_cells_hit'],
            max_occ768=round(g['g768']['max_occupancy'], 3),
            mask_iou=g['mask_iou'],
            iou_h=g['iou_h_bestk'],
        ))

    out = dict(geom=geom_summary, encoder=enc, h_box_iou=h_cmp, small_cases=slim_cases)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # don't dump occupancy
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    print(f'wrote {out_path}', flush=True)


if __name__ == '__main__':
    main()
