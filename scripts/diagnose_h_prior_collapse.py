#!/usr/bin/env python3
"""Diagnose WHY the HPriorAdapter collapsed the [localize] candidate boxes.

Hypothesis to test: the gated residual ``tanh(alpha) * HPriorAdapter(Hpatch)`` is
large relative to the RegionAdapter output ``R_i``, so even alpha~0.11 perturbs the
region embeddings hard enough to over-anchor the frozen LoRA onto H.

Measures, on a few real train samples, for the TRAINED h_prior adapter:
    ||delta||_2        (per token)
    ||R_i||_2          (per token)
    ratio = ||delta|| / ||R_i||
and the aggregate residual scale ||tanh(alpha) * delta|| / ||R_i||.

Run:
    python scripts/diagnose_h_prior_collapse.py [config] [n_samples]
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
from models.qwen35 import unwrap_model
from models.h_prior_adapter import HPriorAdapter


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/qwen35_2b_annos_probe_v2_fsm_h_prior_sft.yaml'
    n_batch = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    ckpt = sys.argv[3] if len(sys.argv) > 3 else 'outputs/train/region_sft_annos_think_fsm_h_prior_v1/h_prior_adapter.pt'

    cfg = load_yaml_config(cfg_path)
    set_seed(int(cfg['training']['seed']))
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)

    # Reuse the SFT loader to get a full model with region adapter mounted; then
    # load the trained HPriorAdapter separately and probe its delta magnitude.
    from train_region_sft import load_sft_model, pack_sft_batch
    model, processor, prior = load_sft_model(
        cfg, device, init_sft=cfg['outcome']['sft_adapter'],
        freeze_region=True, freeze_lora=True)
    model.eval()

    hidden = int(model.config.text_config.hidden_size)
    ck = torch.load(ckpt, map_location='cpu')
    hp = HPriorAdapter(**{k: ck['config'][k] for k in ('hidden_size', 'patch_size', 'intermediate_dim', 'init_gate')})
    hp.load_state_dict(ck['state_dict'])
    dtype = next(model.parameters()).dtype
    hp.to(device=device, dtype=dtype).eval()

    train, _ = load_prior_split(cfg)
    train, dev = split_holdout_by_class(train, float(cfg['data']['holdout_ratio']),
                                        seed=int(cfg['training']['seed']))
    pool = build_train_ref_pool(train)
    dataset = OutcomeMultiboxDataset(train, cfg, processor, 'train', pool)
    collator = OutcomeMultiboxCollator(processor, prior, cfg)

    samples = []
    n_a = n_batch // 2
    for i in range(len(dataset)):
        s = dataset[i]
        if s.get('is_anomaly'):
            if sum(1 for x in samples if x.get('is_anomaly')) < n_a:
                samples.append(s)
        else:
            if sum(1 for x in samples if not x.get('is_anomaly')) < n_batch - n_a:
                samples.append(s)
        if len(samples) >= n_batch:
            break

    tokenizer = getattr(processor, 'tokenizer', processor)
    alpha = float(hp.alpha.detach())
    print(f'alpha={alpha:.4f}  tanh(alpha)={torch.tanh(torch.tensor(alpha)).item():.4f}')
    print(f'{"sample":<6}{"n_tok":>6}{"||R_i||":>10}{"||delta||":>10}{"ratio":>9}{"|res|/|R|":>10}')
    print('-' * 56)

    all_ratio = []
    all_res = []
    for idx, s in enumerate(samples):
        packed, _ = pack_sft_batch(collator, device, [s], tokenizer,
                                   multibox=True, thinking=True, answer_only=False)
        region_raw = packed['region_raw']
        adapter = getattr(unwrap_model(model), 'region_adapter', None)
        ref = next(adapter.parameters(), None)
        if ref is not None:
            dd, dt = ref.device, ref.dtype
            region_raw = {k: (v.to(device=dd, dtype=dt) if v.is_floating_point() else v.to(device=dd))
                          for k, v in region_raw.items()}
        with torch.no_grad():
            R = adapter(region_raw)  # [1, N, hidden]
            hpatch = region_raw['hpatch'].to(dtype=dt)
            R_in = R[0].unsqueeze(0)  # [1, N, hidden]
            # recompute delta = HPriorAdapter(Hpatch) before the gate
            B, N = hpatch.shape[:2]
            x = hpatch.reshape(B * N, 1, hp.patch_size, hp.patch_size)
            x = hp.encoder(x).flatten(1)
            delta = hp.proj(x).reshape(B, N, hidden).to(dt)
            # valid mask
            valid = region_raw.get('valid')
            if valid is not None:
                mask = valid.unsqueeze(-1).to(delta.dtype)
                delta = delta * mask
        rn = R[0].norm(dim=-1)          # [N]
        dn = delta[0].norm(dim=-1)      # [N]
        ratio = dn / (rn + 1e-6)
        res_scale = (torch.tanh(torch.tensor(alpha)) * dn) / (rn + 1e-6)
        m_r = rn.mean().item()
        m_d = dn.mean().item()
        m_ratio = ratio.mean().item()
        m_res = res_scale.mean().item()
        all_ratio.append(m_ratio)
        all_res.append(m_res)
        print(f'{idx:<6}{N:>6}{m_r:>10.3f}{m_d:>10.3f}{m_ratio:>9.3f}{m_res:>10.3f}')

    print('-' * 56)
    print(f'mean ratio ||delta||/||R|| = {sum(all_ratio)/len(all_ratio):.3f}')
    print(f'mean residual scale |tanh(alpha)*delta|/|R| = {sum(all_res)/len(all_res):.3f}')
    print()
    print('interpretation: ratio >> 1 => delta dominates R even with a small gate;')
    print('a ratio near 0.1-0.3 with tanh~0.11 gives ~1-3% perturbation (benign).')
    print('a residual scale > 0.3 is the likely collapse cause.')


if __name__ == '__main__':
    main()
