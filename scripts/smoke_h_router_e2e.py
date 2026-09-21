"""End-to-end router smoke test: collate a real batch in routed mode and run one
teacher-forced forward/backward to confirm region_role flows and role_embedding
receives a finite gradient (no integration breakage)."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from data.prior_dataset import build_train_ref_pool
from data.scan import load_prior_split, split_holdout_by_class
from models.qwen35 import unwrap_model, force_vision_eval
from outcome.inputs import OutcomeCollator, OutcomeDataset
from outcome.inputs_multibox import OutcomeMultiboxCollator, OutcomeMultiboxDataset
from outcome.thinking import thinking_enabled
from train_region_sft import load_sft_model, pack_sft_batch, batch_loss
from utils.common import set_seed
from utils.config import load_yaml_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--init-sft', default=None)
    ap.add_argument('--samples', type=int, default=2)
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    set_seed(int(cfg['training']['seed']))
    device = torch.device('cuda', 0)

    model, processor, prior = load_sft_model(cfg, device, init_sft=args.init_sft)
    tokenizer = getattr(processor, 'tokenizer', processor)
    thinking = thinking_enabled(cfg)
    reasoning_mode = str((cfg.get('outcome') or {}).get('reasoning_mode', 'fsm'))
    answer_only = reasoning_mode in ('loop', 'direct')
    localize_source = str((cfg.get('outcome') or {}).get('sft', {}).get('localize_target', 'h'))
    multibox = (cfg.get('outcome', {}).get('version') == 'outcome-multibox-v1')

    train, test = load_prior_split(cfg)
    train, dev = split_holdout_by_class(train, float(cfg['data']['holdout_ratio']),
                                        seed=int(cfg['training']['seed']))
    pool = build_train_ref_pool(train)
    cls = OutcomeMultiboxDataset if multibox else OutcomeDataset
    collator_cls = OutcomeMultiboxCollator if multibox else OutcomeCollator
    dataset = cls(train, cfg, processor, 'train', pool)
    dataset.samples = dataset.samples[:max(1, int(args.samples))]
    collator = collator_cls(processor, prior, cfg)

    model.train()
    force_vision_eval(model)
    samples = [dataset[i] for i in range(len(dataset))]
    packed, n_sup = pack_sft_batch(collator, device, samples, tokenizer, multibox,
                                   thinking, answer_only, localize_source)

    rr = packed['region_raw']
    n = rr['valid'].shape[1]
    print(f"[e2e] region tokens={n} role shape={tuple(rr['role'].shape)} "
          f"role unique={sorted(set(rr['role'][0].tolist()))} "
          f"n_sup={n_sup} seq_lens={packed['seq_lens']}")
    assert 'role' in rr, "region_raw missing role"
    assert rr['role'].dtype == torch.long, "role must be long"
    assert rr['role'].shape == (1, n), "role shape mismatch"

    loss = batch_loss(model, packed, backward_scale=1.0)
    print(f"[e2e] loss={float(loss.detach()):.4f}")

    core = unwrap_model(model)
    adapter = core.region_adapter
    g = adapter.role_embedding.weight.grad
    if g is None:
        print("[e2e] role_embedding grad is None (bad)")
        sys.exit(1)
    print(f"[e2e] role_embedding grad: finite={bool(torch.isfinite(g).all())} "
          f"norm={float(g.norm()):.6f} nonzero={bool((g.abs() > 0).any())}")
    assert torch.isfinite(g).all(), "role_embedding grad non-finite"
    assert (g.abs() > 0).any(), "role_embedding grad is zero"
    print("[e2e] PASS")


if __name__ == '__main__':
    main()
