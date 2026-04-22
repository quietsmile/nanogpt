#!/bin/bash
set -eux
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export NCCL_ALGO=${NCCL_ALGO:-Ring}
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p out-cybertron-moe-196-resume logs
torchrun --standalone --nproc_per_node=8 train.py config/cybertron_moe_196_resume.py 2>&1 | tee logs/train_resume_$(date +%Y%m%d_%H%M%S).log
