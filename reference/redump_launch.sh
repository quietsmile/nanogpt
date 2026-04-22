#!/bin/bash
# Re-dump ref training: resume from iter_0005988, run 1 iter to capture
# iter 5989 forward with MEGATRON_BITWISE dumps on all DP ranks at ep=3.
set -x
set -e
export GLOO_SOCKET_IFNAME=eth0

EXP_NAME=scaling_moe_00196_redump
EXP_DIR=/newcpfs/user/yuchen/karpathy/cybertron_dump
YAML_PATH=redump_00196.yaml
SAVE_ROOT=/newcpfs/user/yuchen/karpathy/cybertron_dump/redump_save

export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=22
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200
export NVTE_FUSED_ATTN=1
export NVTE_FLASH_ATTN=1
export NCCL_IB_GID_INDEX=3
export CYBERTRON_USE_HYDRA=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# --- DUMP SETTINGS ---
export MEGATRON_BITWISE=1
export CYBERTRON_DUMP_NAMES="output_layer,decoder.layers.1.mlp,decoder.layers.1.self_attention,embedding"
export MEGATRON_BITWISE_DUMP_PATH=/newcpfs/user/yuchen/karpathy/cybertron_dump/dumps_allranks
mkdir -p "$MEGATRON_BITWISE_DUMP_PATH"

GPUS_PER_NODE=$(nvidia-smi | grep NVIDIA | grep On | wc -l)

CYBERTRON_PATH="/newcpfs/user/yuchen/llm/cybertron_dots3.0_swa"
MEGATRON_CORE_PATH="/newcpfs/user/yuchen/llm/megatron_dots3.0_swa"
export PYTHONPATH="${CYBERTRON_PATH}:${MEGATRON_CORE_PATH}:${PYTHONPATH}"
cd "${CYBERTRON_PATH}"

SAVE_DIR="${SAVE_ROOT}"
export save_dir="${SAVE_DIR}/"
export exp_name="${EXP_NAME}"
export tensorboard_dir="${SAVE_DIR}/tensorboard/${EXP_NAME}"

mkdir -p "${SAVE_DIR}/${EXP_NAME}" \
         "${SAVE_DIR}/${EXP_NAME}/loguru" \
         "${SAVE_DIR}/${EXP_NAME}/logs" \
         "${SAVE_DIR}/${EXP_NAME}/logs/runs/.hydra"

# LOAD from original iter_0005988, SAVE elsewhere to avoid overwriting
# Symlink the iter_0005988 ckpt into our save dir for Megatron to find it.
ORIG_CKPT=/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196
mkdir -p "${SAVE_DIR}/${EXP_NAME}"
if [ ! -e "${SAVE_DIR}/${EXP_NAME}/iter_0005988" ]; then
    ln -s ${ORIG_CKPT}/iter_0005988 "${SAVE_DIR}/${EXP_NAME}/iter_0005988"
fi
echo "5988" > "${SAVE_DIR}/${EXP_NAME}/latest_checkpointed_iteration.txt"
LOAD_DIR="${SAVE_DIR}"

DISTRIBUTED_ARGS="--nproc_per_node ${GPUS_PER_NODE} --nnodes ${WORLD_SIZE} --node_rank ${RANK} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT}"
# Use MODIFIED pretrain script with allranks dumping
TRAIN_SCRIPT_PATH="/newcpfs/user/yuchen/karpathy/cybertron_dump/pretrain_deepseek_v2.py"

nvidia-smi

logfile="${SAVE_DIR}/${EXP_NAME}/logs/rank-${RANK}-${WORLD_SIZE}-${EXP_NAME}-run.log"

# Use modified yaml that sets exit_interval=5990 (run iter 5989 + 5990, exit)
HYDRA_FULL_ERROR=1 torchrun ${DISTRIBUTED_ARGS} ${TRAIN_SCRIPT_PATH} \
--config-path="${EXP_DIR}" \
--config-name="${YAML_PATH}" \
load_dir="${LOAD_DIR}" save_dir="${SAVE_DIR}" exp_name="${EXP_NAME}" \
cybertron.finetune=false \
cybertron.no_load_rng=false \
cybertron.no_load_optim=false \
cybertron.override_opt_param_scheduler=true \
cybertron.use_checkpoint_opt_param_scheduler=false \
cybertron.tokenizer_model=/prodcpfs/user/xiaoming/models/dots_tokenizer \
cybertron.data_cache_path="/prodcpfs/user/data/save/data/lossalign/data_cache" \
cybertron.distributed_timeout_minutes=900 \
2>&1 | tee "${logfile}"
