#!/bin/bash
set -eux
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export NCCL_ALGO=${NCCL_ALGO:-Ring}
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
OUT_DIR=out-cybertron-moe-196-bare
mkdir -p "$OUT_DIR" logs
torchrun --standalone --nproc_per_node=8 \
  train.py config/cybertron_moe_196_bare.py \
  --out_dir="$OUT_DIR" \
  2>&1 | tee logs/train_bare_$(date +%Y%m%d_%H%M%S).log
