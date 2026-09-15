#!/usr/bin/env bash
# ANNOS-box GT probe: 2-epoch SFT → MVTec-200 eval → short RL (~400 local steps).
set -euo pipefail
cd /root/data-fs/bljiao/anomaly_llm
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
PY=/root/miniconda3/envs/zlt/bin/python
PROBE=configs/qwen35_2b_annos_probe.yaml
BASE=configs/qwen35_2b_outcome_multibox.yaml
SFT=outputs/train/region_sft_annos_probe
RL=outputs/train/qwen35_2b_outcome_multibox/train_annos_rl_a400
HYBRID=outputs/train/qwen35_2b_outcome_multibox/train_hybrid_rl3/checkpoint-1000
LOG=outputs/train/annos_gt_probe.log
mkdir -p outputs/train
exec > >(tee -a "$LOG") 2>&1

echo "===== $(date -Is) ANNOS GT probe start ====="

echo "===== 1) SFT 2 epochs (ANNOS boxes) ====="
CUDA_VISIBLE_DEVICES=0,1 "$PY" train_region_sft.py \
  --config "$PROBE" \
  --output-dir "$SFT" \
  --epochs 2 --lr 1e-4 --num-gpu 2 --save-steps 100

if [[ ! -f "$SFT/adapter_config.json" ]]; then
  echo "SFT adapter missing at $SFT" >&2
  exit 1
fi

echo "===== 2) Eval new SFT on MVTec-200 (ANNOS GT) ====="
CUDA_VISIBLE_DEVICES=0 "$PY" train_outcome_multibox.py \
  --config "$PROBE" --mode eval --split test --eval-limit 200 \
  --output-dir "${SFT}/eval_mvtec200_annos"

echo "===== 3) Baseline: hybrid checkpoint-1000 on MVTec-200 (ANNOS GT, old SFT merge) ====="
if [[ -f "$HYBRID/adapter_config.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "$PY" train_outcome_multibox.py \
    --config "$BASE" --mode eval --split test --eval-limit 200 \
    --adapter "$HYBRID" \
    --output-dir outputs/train/qwen35_2b_outcome_multibox/eval_hybrid1000_annos_gt
else
  echo "skip hybrid baseline, missing $HYBRID"
fi

echo "===== 4) RL ~400 local steps (2 GPU, max-attempts 800) ====="
CUDA_VISIBLE_DEVICES=0,1 "$PY" train_outcome_multibox.py \
  --config "$PROBE" --mode train --num-gpu 2 --max-attempts 800 \
  --output-dir "$RL"

echo "===== $(date -Is) ANNOS GT probe done ====="
echo "SFT eval: ${SFT}/eval_mvtec200_annos"
echo "hybrid ANNOS baseline: outputs/train/qwen35_2b_outcome_multibox/eval_hybrid1000_annos_gt"
echo "RL: $RL"
echo "TensorBoard: http://127.0.0.1:5007"
