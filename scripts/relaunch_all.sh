#!/usr/bin/env bash
# Relaunch the full ANNOS probe v2 experiment suite after the code/config updates.
# Phase 1: train the missing loop2/direct SFT adapters (parallel).
# Phase 2: RL queue on two GPUs.
#
# GPU 0: fsm_final_only -> fsm_full -> direct
# GPU 1: loop4 -> loop2
set -uo pipefail
cd /root/data-fs/bljiao/anomaly_llm
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
source /root/miniconda3/etc/profile.d/conda.sh
conda activate zlt
PY=/root/miniconda3/envs/zlt/bin/python

LOG_DIR=outputs/train/relaunch_logs
mkdir -p "$LOG_DIR"
ORCH="$LOG_DIR/orchestrator.log"
echo "===== $(date -Is) relaunch_all start =====" | tee -a "$ORCH"

# ---- Phase 1: SFT prerequisites (parallel) ----
echo "$(date -Is) Phase1: training loop2 SFT (GPU0) + direct SFT (GPU1)" | tee -a "$ORCH"

CUDA_VISIBLE_DEVICES=0 "$PY" train_region_sft.py \
  --config configs/qwen35_2b_annos_probe_v2_loop2_sft.yaml \
  --output-dir outputs/train/region_sft_loop2 --epochs 1 --num-gpu 1 \
  > "$LOG_DIR/sft_loop2.log" 2>&1 &
SFT_LOOP2=$!

CUDA_VISIBLE_DEVICES=1 "$PY" train_region_sft.py \
  --config configs/qwen35_2b_annos_probe_v2_direct_sft.yaml \
  --output-dir outputs/train/region_sft_direct --epochs 1 --num-gpu 1 \
  > "$LOG_DIR/sft_direct.log" 2>&1 &
SFT_DIRECT=$!

wait $SFT_LOOP2; RC2=$?
wait $SFT_DIRECT; RCD=$?
echo "$(date -Is) Phase1 done: loop2=$RC2 direct=$RCD" | tee -a "$ORCH"

if [ ! -f "outputs/train/region_sft_loop2/adapter_config.json" ] || [ ! -f "outputs/train/region_sft_direct/adapter_config.json" ]; then
  echo "ERROR: one or both SFT adapters missing; aborting before RL." | tee -a "$ORCH"
  echo "  loop2: $(ls outputs/train/region_sft_loop2/adapter_config.json 2>&1)" | tee -a "$ORCH"
  echo "  direct: $(ls outputs/train/region_sft_direct/adapter_config.json 2>&1)" | tee -a "$ORCH"
  exit 1
fi

# ---- Phase 2: RL queue ----
echo "$(date -Is) Phase2: launching RL queue" | tee -a "$ORCH"

run_gpu0() {
  for cfg in qwen35_2b_annos_probe_v2_fsm_final_only_rl qwen35_2b_annos_probe_v2_fsm_rl qwen35_2b_annos_probe_v2_direct_rl; do
    echo "$(date -Is) [GPU0] start $cfg" | tee -a "$ORCH"
    CUDA_VISIBLE_DEVICES=0 "$PY" train_outcome_multibox.py \
      --config "configs/${cfg}.yaml" --mode train --num-gpu 1 \
      >> "$LOG_DIR/rl_${cfg}.log" 2>&1
    echo "$(date -Is) [GPU0] done  $cfg (exit $?)" | tee -a "$ORCH"
  done
}

run_gpu1() {
  for cfg in qwen35_2b_annos_probe_v2_loop4_rl qwen35_2b_annos_probe_v2_loop2_rl; do
    echo "$(date -Is) [GPU1] start $cfg" | tee -a "$ORCH"
    CUDA_VISIBLE_DEVICES=1 "$PY" train_outcome_multibox.py \
      --config "configs/${cfg}.yaml" --mode train --num-gpu 1 \
      >> "$LOG_DIR/rl_${cfg}.log" 2>&1
    echo "$(date -Is) [GPU1] done  $cfg (exit $?)" | tee -a "$ORCH"
  done
}

run_gpu0 &
GPU0_PID=$!
run_gpu1 &
GPU1_PID=$!

wait $GPU0_PID
wait $GPU1_PID
echo "===== $(date -Is) relaunch_all ALL DONE =====" | tee -a "$ORCH"
