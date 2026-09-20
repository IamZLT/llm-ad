#!/usr/bin/env python3
"""Quick timing probe: where does per-sample cost go in the representation diag?

Times, for a handful of small-anomaly samples at H@768:
    (a) _load_pair + align + tokenize
    (b) extract_all (2x vision forward + NN matching)
    (c) metrics_for_map (incl. 9x threshold scan)
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from utils.config import load_yaml_config
from utils.common import set_seed
from outcome.inputs import pixel_budget
from outcome.engine_multibox import datasets, load_model
from scripts.diagnose_rep_resolution import extract_all, metrics_for_map

def main():
    cfg = load_yaml_config('configs/qwen35_2b_annos_probe_v2_fsm_rl.yaml')
    cfg.setdefault('outcome', {})['sft_adapter'] = 'outputs/train/region_sft_annos_think_fsm'
    cfg['outcome'].setdefault('prior', {})['h_image_size'] = 768
    cfg['outcome'].setdefault('zoom', {})['enabled'] = False
    device = torch.device('cuda:0')
    set_seed(int(cfg['training']['seed']))

    t0 = time.time()
    model, processor, prior = load_model(cfg, None, fresh_lora=False)
    print(f'load_model: {time.time()-t0:.1f}s', flush=True)

    from outcome.inputs_multibox import OutcomeMultiboxCollator
    collator = OutcomeMultiboxCollator(processor, prior, cfg)
    prior_cfg = dict(cfg['outcome'].get('prior', {}))

    t0 = time.time()
    _, _, test_set = datasets(cfg, processor)
    print(f'datasets(): {time.time()-t0:.1f}s', flush=True)

    small = []
    t0 = time.time()
    for i in range(len(test_set)):
        item = test_set[i]
        if not item.get('is_anomaly'):
            continue
        comps = list(item.get('component_bboxes') or []) or ([item['gt_box_px']] if item.get('gt_box_px') else [])
        if not comps:
            continue
        frac = item.get('mask_area_fraction')
        if frac is None:
            gt = item['gt_box_px']; w, h = item['orig_size']
            frac = (gt[2]-gt[0])*(gt[3]-gt[1])/(w*h) if gt else 0.0
        if frac < 0.02:
            small.append(item)
    small.sort(key=lambda x: x['image_path'])
    print(f'collect small ({len(small)}): {time.time()-t0:.1f}s', flush=True)

    for si in range(3):
        item = small[si]
        t0 = time.time()
        with pixel_budget(processor, 768):
            ref_h, test_h = collator._align_pair_at(item['ref'], item['test'], 768)
            h_in = collator._concat_image_tensors(ref_h, test_h)
        pv = h_in['pixel_values'].to(device); grid = h_in['image_grid_thw'].to(device)
        counts = [int(v) for v in grid.prod(-1)]
        ref_pv, test_pv = torch.split(pv, counts, dim=0)
        ta = time.time() - t0
        t0 = time.time()
        with torch.no_grad():
            rep = extract_all(prior, ref_pv, grid[0:1], test_pv, grid[1:2])
        tb = time.time() - t0
        t0 = time.time()
        comps = list(item.get('component_bboxes') or []) or ([item['gt_box_px']] if item.get('gt_box_px') else [])
        m = metrics_for_map(rep['fused'], prior_cfg, comps, item['orig_size'], rep['hw_t'])
        tc = time.time() - t0
        print(f'sample{si}: align/tok={ta:.1f}s  extract_all={tb:.1f}s  metrics={tc:.1f}s', flush=True)


if __name__ == '__main__':
    main()
