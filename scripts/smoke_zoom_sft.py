#!/usr/bin/env python3
"""Smoke-test the Zoom-SFT training path on one anomaly + one normal sample.

Loads the FSM checkpoint (LoRA + frozen region adapter), builds the multibox
collator with zoom enabled (top-1 paired same-coordinate crop), packs a batch,
and runs one forward+backward to catch any shape errors before the full run.
"""
import os, sys, json
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
from train_region_sft import load_sft_model, pack_sft_batch, batch_loss

def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/qwen35_2b_annos_probe_v2_fsm_zoom_sft.yaml'
    cfg = load_yaml_config(cfg_path)
    seed = int(cfg['training']['seed'])
    set_seed(seed)
    device = torch.device('cuda', 0)

    model, processor, prior = load_sft_model(
        cfg, device, init_sft=cfg['outcome']['sft_adapter'], freeze_region=True)
    tokenizer = getattr(processor, 'tokenizer', processor)
    model.eval()

    train, test = load_prior_split(cfg)
    train, dev = split_holdout_by_class(train, float(cfg['data']['holdout_ratio']), seed=seed)
    pool = build_train_ref_pool(train)
    dataset = OutcomeMultiboxDataset(train, cfg, processor, 'train', pool)
    collator = OutcomeMultiboxCollator(processor, prior, cfg)

    # pick one anomaly and one normal sample (dataset[i] returns the transformed sample)
    anom = norm = None
    for i in range(len(dataset)):
        s = dataset[i]
        if s.get('is_anomaly') and anom is None:
            anom = (i, s)
        elif not s.get('is_anomaly') and norm is None:
            norm = (i, s)
        if anom and norm:
            break
    print(f'dataset size={len(dataset)}  anomaly idx={anom[0] if anom else None}  '
          f'normal idx={norm[0] if norm else None}')
    print(f'anomaly sample={anom[1]["image_path"] if anom else None}')
    print(f'normal sample={norm[1]["image_path"] if norm else None}')

    for tag, (idx, sample) in [('anomaly', anom), ('normal', norm)]:
        if sample is None:
            continue
        batch = collator([sample])
        meta = batch['_meta'][0]
        print(f'\n[{tag}] image_count={meta["image_count"]} zoom_n_crops={meta["zoom_n_crops"]} '
              f'zoom_h_size={meta["zoom_h_size"]} is_anomaly={meta["is_anomaly"]}')
        packed, n_sup = pack_sft_batch(collator, device, [sample], tokenizer,
                                       multibox=True, thinking=True, answer_only=False)
        print(f'  packed image_embeds={tuple(packed["image_embeds"].shape)} '
              f'region_test={tuple(packed["region_raw"]["test"].shape)} seq_len={packed["seq_lens"][0]} '
              f'supervised_tokens={n_sup}')
        with torch.no_grad():
            loss = batch_loss(model, packed)
        print(f'  loss={float(loss):.4f}')
        # print the target tail to eyeball the zoom prompt binding
        print(f'  target tail: ...{packed["targets"][0][-260:]}')

    print('\nSMOKE OK')

if __name__ == '__main__':
    main()
