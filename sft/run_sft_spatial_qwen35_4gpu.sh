#!/bin/bash
# 4-GPU full-parameter SFT on spatial-repair zoom_seg data, starting from original Qwen3.5-4B.

set -euo pipefail
cd /root/autodl-tmp/VQA

source .venv/bin/activate 2>/dev/null || true

: "${PILOT_MODEL_PATH:=/root/autodl-tmp/VQA/models/Qwen3.5-4B}"
: "${TRAIN_JSONL:=/root/autodl-tmp/VQA/json_data/zoom_seg_json/sft_spatial/sft-00000-of-00001.zoom_seg.think_spatial.jsonl}"
: "${OUT_DIR:=/root/autodl-tmp/VQA/sft/ckpt_sft_spatial_qwen35_4gpu}"
: "${NPROC_PER_NODE:=4}"
: "${NUM_TRAIN_EPOCHS:=3}"
: "${BATCH_SIZE_PER_GPU:=1}"
: "${GRAD_ACCUM:=1}"
: "${LR:=3e-5}"
: "${WARMUP_STEPS:=500}"
: "${SAVE_STEPS:=200}"
: "${SAVE_TOTAL_LIMIT:=3}"

export PILOT_MODEL_PATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "=== sft_spatial start $(date '+%F %T %Z') ==="
echo "model=${PILOT_MODEL_PATH}"
echo "train_jsonl=${TRAIN_JSONL}"
echo "out_dir=${OUT_DIR}"
echo "nproc=${NPROC_PER_NODE} epochs=${NUM_TRAIN_EPOCHS} batch=${BATCH_SIZE_PER_GPU} grad_accum=${GRAD_ACCUM} lr=${LR}"

accelerate launch     --multi_gpu     --num_processes "${NPROC_PER_NODE}"     --num_machines 1     --mixed_precision bf16     /root/autodl-tmp/VQA/speedup/sft/train_sft_full_qwen35.py     --model_name "${PILOT_MODEL_PATH}"     --train_jsonl "${TRAIN_JSONL}"     --output_dir "${OUT_DIR}"     --num_train_epochs "${NUM_TRAIN_EPOCHS}"     --batch_size_per_gpu "${BATCH_SIZE_PER_GPU}"     --gradient_accumulation_steps "${GRAD_ACCUM}"     --lr "${LR}"     --warmup_steps "${WARMUP_STEPS}"     --weight_decay 0.01     --max_grad_norm 1.0     --seed 42     --dtype bfloat16     --save_steps "${SAVE_STEPS}"     --save_total_limit "${SAVE_TOTAL_LIMIT}"     --log_steps 10     --print_steps 20     --max_pixels $((64*64*28*28))     --max_length 4096     --no_resume     "$@"     2>&1 | tee "${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
