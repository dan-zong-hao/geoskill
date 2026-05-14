#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/VQA
RL_BBOX="$ROOT/speedup/rl_bbox"
cd "$ROOT"

source .venv/bin/activate 2>/dev/null || true
source /etc/network_turbo 2>/dev/null || true

: "${MODEL_PATH:=$ROOT/sft/ckpt_sft_full_qwen35_think1024_4gpu/final_hf}"
: "${TRAIN_JSONL:=$ROOT/json_data/zoom_seg_json/rl_level/rl-00000-of-00001.1.zoom_seg.think.jsonl}"
: "${OUT_DIR:=$RL_BBOX/ckpt_bbox_grpo_4gpu_fixiou_ze_steps}"
: "${NPROC_PER_NODE:=4}"
: "${NUM_TRAIN_EPOCHS:=5}"
: "${GRAD_ACCUM:=1}"
: "${NUM_GENERATIONS:=4}"
: "${BATCH_SIZE_PER_DEVICE:=1}"

LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_HUB_VERBOSITY=${HF_HUB_VERBOSITY:-error}
export HF_HUB_DISABLE_PROGRESS_BARS=${HF_HUB_DISABLE_PROGRESS_BARS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

echo "=== bbox GRPO ZoomEarth-style steps start $(date '+%F %T %Z') ==="
echo "model=$MODEL_PATH"
echo "train_jsonl=$TRAIN_JSONL"
echo "out_dir=$OUT_DIR"
echo "nproc=$NPROC_PER_NODE epochs=$NUM_TRAIN_EPOCHS batch_per_device=$BATCH_SIZE_PER_DEVICE grad_accum=$GRAD_ACCUM num_generations=$NUM_GENERATIONS"
echo "expected unique prompts/update = nproc * batch_per_device * grad_accum = $((NPROC_PER_NODE * BATCH_SIZE_PER_DEVICE * GRAD_ACCUM))"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NPROC_PER_NODE" \
  "$RL_BBOX/train_grpo_bbox.py" \
  --model_path "$MODEL_PATH" \
  --train_jsonl "$TRAIN_JSONL" \
  --output_dir "$OUT_DIR" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --batch_size_per_device "$BATCH_SIZE_PER_DEVICE" \
  --num_generations "$NUM_GENERATIONS" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --learning_rate 3e-6 \
  --warmup_steps 50 \
  --beta 0.04 \
  --rollout_temperature 0.7 \
  --rollout_top_p 0.8 \
  --rollout_top_k 20 \
  --rollout_min_p 0.0 \
  --rollout_repetition_penalty 1.0 \
  --max_new_tokens_turn1 384 \
  --w_format 0.05 \
  --w_iou 1.0 \
  --w_rg 1.0 \
  --rg_alpha 200.0 \
  --save_steps 100 \
  --log_steps 1 \
  "$@" \
  2>&1 | tee "$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
