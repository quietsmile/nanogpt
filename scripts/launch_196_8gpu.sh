#!/bin/bash
# Launch nanogpt training to reproduce scaling_moe_00196 on 8× GPU.
# Run from /root/nanogpt (or wherever this repo lives).
set -eux

cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export NCCL_ALGO=${NCCL_ALGO:-Ring}
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

OUT_DIR=${OUT_DIR:-out-cybertron-moe-196}
mkdir -p "$OUT_DIR" logs

torchrun --standalone --nproc_per_node=8 \
  train.py config/cybertron_moe_196.py \
  --out_dir="$OUT_DIR" \
  2>&1 | tee logs/train_196_$(date +%Y%m%d_%H%M%S).log
