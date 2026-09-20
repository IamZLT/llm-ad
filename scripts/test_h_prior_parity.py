#!/usr/bin/env python3
"""No-training verification of the H-Region Fusion PoC (HPriorAdapter).

Three checks, in order:
  1. baseline parity  : h_prior.enabled=false -> logits identical to the region-only path.
  2. alpha=0 parity   : h_prior.enabled=true, alpha=0 -> logits identical to baseline
                        (tanh(0)=0, so the gated residual is exactly identity).
  3. H-patch visual   : render the full H map + region centers and each 7x7 sampled
                        patch, to verify the coordinate mapping (no flip/off-by-half).

Usage:
  python scripts/test_h_prior_parity.py [config] [out_png_dir]
"""
import os, sys
os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image, ImageDraw

from utils.config import load_yaml_config
from utils.common import set_seed
from data.prior_dataset import build_train_ref_pool
from data.scan import load_prior_split, split_holdout_by_class
from outcome.inputs_multibox import OutcomeMultiboxCollator, OutcomeMultiboxDataset
from train_region_sft import load_sft_model, pack_sft_batch
from models.qwen35 import unwrap_model
from models.region_injection import bind_region_injection
from models.vision_cache import bind_cached_image_features
from rl.grpo import forward_with_vision


def forward_logits(model, packed):
    adapter = getattr(unwrap_model(model), 'region_adapter', None)
    with bind_cached_image_features(model, packed['image_embeds']), \
            bind_region_injection(model, adapter, packed['region_raw'], int(packed['region_token_id'])):
        out = forward_with_vision(model, packed['gen_in'], packed['input_ids'], packed['attention_mask'])
    return out.logits


def render_patches(batch, packed, out_path):
    """Draw the H map + region centers, then each 7x7 H patch next to its center."""
    meta = batch['_meta'][0]
    test_img = meta['test'].convert('RGB')
    heat = meta['heatmap'].convert('RGB')
    # overlay heatmap on test image (semi-transparent)
    overlay = Image.blend(test_img, heat, 0.45)
    d = ImageDraw.Draw(overlay)
    W, H = test_img.size
    for cand in meta.get('prior_candidates', []):
        x1, y1, x2, y2 = [int(v / 1000 * (W if i % 2 == 0 else H)) for i, v in enumerate(cand['bbox_2d'])]
        d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        px, py = cand['peak_2d']
        d.ellipse([px / 1000 * W - 4, py / 1000 * H - 4, px / 1000 * W + 4, py / 1000 * H + 4],
                  fill=(0, 255, 0))

    raw = packed['region_raw']
    hpatch = raw['hpatch'][0].cpu()          # [n, 1, P, P]
    geom = raw['geom'][0].cpu()              # [n, 5]
    hstat = raw['hstat'][0].cpu()            # [n, 2]
    valid = raw['valid'][0].cpu().bool()     # [n]
    n = int(hpatch.shape[0])
    P = int(hpatch.shape[-1])

    # tile of patches
    tile = 64
    pad = 8
    cols = max(1, n)
    canvas = Image.new('L', (cols * (tile + pad) + pad, tile + 2 * pad + 12), 20)
    dd = ImageDraw.Draw(canvas)
    for i in range(n):
        p = hpatch[i, 0]                     # [P, P]
        p = (p - p.min()) / (p.max() - p.min() + 1e-6)
        img = Image.fromarray((p.numpy() * 255).astype('uint8')).resize((tile, tile), Image.NEAREST)
        canvas.paste(img, (pad + i * (tile + pad), pad + 12))
        gcx, gcy = float(geom[i, 0]), float(geom[i, 1])
        dd.text((pad + i * (tile + pad), pad), f'v={int(valid[i])} ({gcx:.2f},{gcy:.2f})', fill=255)

    # side-by-side: overview + patches
    W0, H0 = overlay.size
    out = Image.new('RGB', (W0 + canvas.width + 20, max(H0, canvas.height)), (0, 0, 0))
    out.paste(overlay, (0, 0))
    out.paste(canvas.convert('RGB'), (W0 + 20, 0))
    out.save(out_path)
    print(f'  saved viz -> {out_path}  (n_tokens={n}, P={P}, hstat={hstat.tolist()})')


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/qwen35_2b_annos_probe_v2_fsm_h_prior_sft.yaml'
    out_dir = sys.argv[2] if len(sys.argv) > 2 else 'outputs/eval/h_prior_poc'
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml_config(cfg_path)
    set_seed(int(cfg['training']['seed']))
    device = torch.device('cuda', 0)

    hcfg = (cfg.get('outcome') or {}).get('h_prior_adapter') or {}
    print(f"[config] h_prior_adapter.enabled={hcfg.get('enabled', False)} "
          f"patch_size={hcfg.get('patch_size', 7)} init_gate={hcfg.get('init_gate', 0.0)}")

    model, processor, prior = load_sft_model(
        cfg, device, init_sft=cfg['outcome']['sft_adapter'],
        freeze_region=True, freeze_lora=True)
    tokenizer = getattr(processor, 'tokenizer', processor)
    model.eval()

    h_adapter = getattr(unwrap_model(model), 'h_prior_adapter', None)
    print(f"[model] h_prior_adapter mounted={h_adapter is not None}; alpha={float(h_adapter.alpha) if h_adapter is not None else None}")

    train, test = load_prior_split(cfg)
    train, dev = split_holdout_by_class(train, float(cfg['data']['holdout_ratio']), seed=int(cfg['training']['seed']))
    pool = build_train_ref_pool(train)
    dataset = OutcomeMultiboxDataset(train, cfg, processor, 'train', pool)
    collator = OutcomeMultiboxCollator(processor, prior, cfg)

    anom = norm = None
    for i in range(len(dataset)):
        s = dataset[i]
        if s.get('is_anomaly') and anom is None:
            anom = (i, s)
        elif not s.get('is_anomaly') and norm is None:
            norm = (i, s)
        if anom and norm:
            break
    print(f"[data] anomaly idx={anom[0]} normal idx={norm[0]}")

    # ---- parity checks on the anomaly sample (has real H candidates) ----
    batch = collator([anom[1]])
    packed, _ = pack_sft_batch(collator, device, [anom[1]], tokenizer,
                               multibox=True, thinking=True, answer_only=False)
    rr = packed['region_raw']
    print(f"[shape] region_test={tuple(rr['test'].shape)} hpatch={tuple(rr['hpatch'].shape)} "
          f"geom={tuple(rr['geom'].shape)} valid={tuple(rr['valid'].shape)}")

    core = unwrap_model(model)
    with torch.no_grad():
        # Test 1: baseline (h_prior adapter detached)
        saved = core.h_prior_adapter
        core.h_prior_adapter = None
        logits_base = forward_logits(model, packed)
        # Test 2: enabled + alpha=0
        core.h_prior_adapter = saved
        logits_new = forward_logits(model, packed)
    core.h_prior_adapter = saved

    diff = (logits_base.float() - logits_new.float()).abs().max().item()
    print(f"[parity] logits shape={tuple(logits_base.shape)} "
          f"max_abs_diff(base vs alpha=0)={diff:.3e}")
    print(f"[parity] alpha=0 exact identity: {'PASS' if diff == 0.0 else 'FAIL'}")

    # ---- visualization ----
    render_patches(batch, packed, out_dir / 'anomaly_hpatches.png')

    # also render a normal sample (often a false-positive candidate)
    nb = collator([norm[1]])
    npk, _ = pack_sft_batch(collator, device, [norm[1]], tokenizer,
                            multibox=True, thinking=True, answer_only=False)
    render_patches(nb, npk, out_dir / 'normal_hpatches.png')

    print('DONE')


if __name__ == '__main__':
    main()
