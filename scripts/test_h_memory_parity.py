#!/usr/bin/env python3
"""No-training verification of the Spatial H-Memory cross-attention PoC.

Checks, in order:
  1. collator emits ``h_map`` (real) and the memory encoder builds Z_H [1,K,d].
  2. gate=0 parity : h_memory mounted but every cross-attn gate zero -> logits identical
                     to the baseline (h_memory unmounted). tanh(0)=0 => exact identity.
  3. gate>0        : restoring the init gate changes logits (the residual is live).
  4. memory causal : real/shuffled/zero produce DIFFERENT memories (so the ablation is
                     well-posed) while PE keeps them same-shaped.

Usage:
  python scripts/test_h_memory_parity.py [config] [out_dir]
"""
import os, sys
os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from utils.config import load_yaml_config
from utils.common import set_seed
from data.prior_dataset import build_train_ref_pool
from data.scan import load_prior_split, split_holdout_by_class
from outcome.inputs_multibox import OutcomeMultiboxCollator, OutcomeMultiboxDataset
from train_region_sft import load_sft_model, pack_sft_batch
from models.qwen35 import unwrap_model
from models.h_memory import bind_h_cross_attn, build_h_channels
from models.region_injection import bind_region_injection
from models.vision_cache import bind_cached_image_features
from rl.grpo import forward_with_vision


def _forward(model, packed, h_mem=None):
    adapter = getattr(unwrap_model(model), 'region_adapter', None)
    with bind_cached_image_features(model, packed['image_embeds']), \
            bind_region_injection(model, adapter, packed['region_raw'], int(packed['region_token_id'])), \
            bind_h_cross_attn(model, h_mem, packed.get('h_maps')):
        out = forward_with_vision(model, packed['gen_in'], packed['input_ids'], packed['attention_mask'])
    return out.logits


def _set_gates(h_mem, value):
    for ca in h_mem.cross_attns.values():
        with torch.no_grad():
            ca.gate.fill_(value)


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/qwen35_2b_annos_probe_v2_fsm_h_mem_sft.yaml'
    cfg = load_yaml_config(cfg_path)
    set_seed(int(cfg['training']['seed']))
    device = torch.device('cuda', 0)

    hmc = (cfg.get('outcome') or {}).get('h_memory') or {}
    print(f"[config] h_memory.enabled={hmc.get('enabled')} layers={hmc.get('layers')} "
          f"K={hmc.get('memory_size')} init_gate={hmc.get('init_gate')} "
          f"condition={hmc.get('condition')}")

    model, processor, prior = load_sft_model(
        cfg, device, init_sft=cfg['outcome']['sft_adapter'],
        freeze_region=True, freeze_lora=True)
    tokenizer = getattr(processor, 'tokenizer', processor)
    model.eval()
    core = unwrap_model(model)

    h_mem = getattr(core, 'h_memory', None)
    assert h_mem is not None, "h_memory not mounted (config enabled?)"
    print(f"[model] h_memory mounted; trainable params={sum(p.numel() for p in h_mem.parameters())}")

    train, test = load_prior_split(cfg)
    train, dev = split_holdout_by_class(train, float(cfg['data']['holdout_ratio']), seed=int(cfg['training']['seed']))
    pool = build_train_ref_pool(train)
    dataset = OutcomeMultiboxDataset(train, cfg, processor, 'train', pool)
    collator = OutcomeMultiboxCollator(processor, prior, cfg)

    sample = None
    for i in range(len(dataset)):
        s = dataset[i]
        if s.get('is_anomaly'):
            sample = s
            break
    assert sample is not None, "no anomaly sample found"

    batch = collator([sample])
    print(f"[collator] h_map present={('h_map' in batch)} shape={tuple(batch.get('h_map', torch.zeros(0)).shape)}")
    packed, _ = pack_sft_batch(collator, device, [sample], tokenizer,
                               multibox=True, thinking=True, answer_only=False,
                               localize_source=str((cfg.get('outcome') or {}).get('sft', {}).get('localize_target', 'h')))
    print(f"[packed] h_maps={len(packed.get('h_maps', []))} x {tuple(packed['h_maps'][0].shape)}")

    # ---- memory build shape ----
    hmap = packed['h_maps'][0]
    z = h_mem.build_memory(hmap.to(device=h_mem.encoder.proj[0].weight.device,
                                   dtype=h_mem.encoder.proj[0].weight.dtype))
    print(f"[memory] Z_H shape={tuple(z.shape)} (expect [1, 64, 256])")

    # ---- causal well-posedness ----
    h_shuf = hmap.flatten()[torch.randperm(hmap.numel())].reshape_as(hmap)
    h_zero = torch.zeros_like(hmap)
    zs = [h_mem.build_memory(m.to(device=z.device, dtype=z.dtype)) for m in (hmap, h_shuf, h_zero)]
    d_real_shuf = (zs[0] - zs[1]).abs().max().item()
    d_real_zero = (zs[0] - zs[2]).abs().max().item()
    print(f"[memory] real-vs-shuffled max|dZ|={d_real_shuf:.4f}  real-vs-zero max|dZ|={d_real_zero:.4f} "
          f"(both should be > 0)")

    saved = h_mem
    with torch.no_grad():
        # Test 1: baseline (h_memory unmounted)
        core.h_memory = None
        logits_base = _forward(model, packed, h_mem=None)

        # Test 2: mounted + all gates = 0
        core.h_memory = saved
        _set_gates(h_mem, 0.0)
        logits_g0 = _forward(model, packed, h_mem=saved)

        # Test 3: mounted + init gates restored
        _set_gates(h_mem, float(hmc.get('init_gate', 1e-3)))
        logits_live = _forward(model, packed, h_mem=saved)
    core.h_memory = saved

    d_g0 = (logits_base.float() - logits_g0.float()).abs().max().item()
    d_live = (logits_base.float() - logits_live.float()).abs().max().item()
    print(f"[parity] base vs gate=0 max_abs_diff={d_g0:.3e}  -> {'PASS' if d_g0 == 0.0 else 'FAIL'}")
    print(f"[parity] base vs gate={hmc.get('init_gate')} max_abs_diff={d_live:.3e}  -> "
          f"{'live (differs)' if d_live > 0 else 'no-op (unexpected)'}")

    # gate magnitude sanity: check the logits do not blow up with a large gate
    _set_gates(h_mem, 1.0)
    with torch.no_grad():
        logits_big = _forward(model, packed, h_mem=saved)
    _set_gates(h_mem, float(hmc.get('init_gate', 1e-3)))
    print(f"[sanity] gate=1.0 logits finite={torch.isfinite(logits_big).all().item()} "
          f"max|d| vs base={((logits_big.float()-logits_base.float()).abs().max()).item():.3f}")

    print('DONE')


if __name__ == '__main__':
    main()
