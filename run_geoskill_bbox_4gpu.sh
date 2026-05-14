#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/VQA
RL_BBOX="$ROOT/speedup/rl_bbox"
GEOSKILL="$ROOT/speedup/geoskill"
cd "$ROOT"

source .venv/bin/activate 2>/dev/null || true
source /etc/network_turbo 2>/dev/null || true

if [ ! -f "$GEOSKILL/split_manifest.json" ]; then
  python "$GEOSKILL/create_splits.py"
fi

: "${MODEL_PATH:=$ROOT/sft/ckpt_sft_full_qwen35_think1024_4gpu/final_hf}"
: "${TRAIN_JSONL:=$ROOT/json_data/zoom_seg_json/rl_level/rl-00000-of-00001.1.zoom_seg.think.jsonl}"
: "${OUT_DIR:=$GEOSKILL/ckpt_geoskill_round1}"
: "${SKILLBANK_PATH:=$GEOSKILL/skillbank_round0.json}"
: "${NPROC_PER_NODE:=4}"

LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export HF_HUB_VERBOSITY=${HF_HUB_VERBOSITY:-error}
export HF_HUB_DISABLE_PROGRESS_BARS=${HF_HUB_DISABLE_PROGRESS_BARS:-1}

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NPROC_PER_NODE" \
  "$RL_BBOX/train_grpo_bbox.py" \
  --model_path "$MODEL_PATH" \
  --train_jsonl "$TRAIN_JSONL" \
  --output_dir "$OUT_DIR" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
  --batch_size_per_device 1 \
  --num_generations "${NUM_GENERATIONS:-4}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --learning_rate "${LEARNING_RATE:-3e-6}" \
  --warmup_steps "${WARMUP_STEPS:-20}" \
  --beta "${BETA:-0.04}" \
  --rollout_temperature 0.7 \
  --rollout_top_p 0.8 \
  --rollout_top_k 20 \
  --rollout_min_p 0.0 \
  --rollout_repetition_penalty 1.0 \
  --max_new_tokens_turn1 512 \
  --w_format 0.05 \
  --w_iou 1.0 \
  --w_rg 1.0 \
  --enable_spatial_reward \
  --w_spatial "${W_SPATIAL:-1.5}" \
  --spatial_penalty "${SPATIAL_PENALTY:-0.5}" \
  --skillbank_path "$SKILLBANK_PATH" \
  --split_manifest "$GEOSKILL/split_manifest.json" \
  --train_split "${TRAIN_SPLIT:-rl_train}" \
  --save_steps "${SAVE_STEPS:-25}" \
  --log_steps 1 \
  "$@" \
  2>&1 | tee "$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
