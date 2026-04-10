# Config for nanogpt cybertron baseline, aligned with scaling_moe_00198.
#
# Reference PAI job: dlc1v93gneqwnrdz (scaling_moe_00198)
# Reference config: /prodcpfs/user/data/GitLab/pretrain_scaling_ladder/scaling_moe_00198.yaml
#
# Model: dense equivalent of scaling_moe_00198 (no MoE, ffn_hidden=1920 for all layers)
#   hidden_size=656, n_layers=16, n_heads=8, n_kv_heads=4, kv_channels=64
#   RMSNorm, RoPE (base=50000), SwiGLU (ffn_hidden=1920), qk_layernorm=True
#   no bias, no weight tying, disable_scaled_init_method=True
#   vocab_size=152064 (Qwen tokenizer, padded to multiple of 64)
#
# Optimizer: AdamW (decoupled weight decay, matches FusedAdam in cybertron)
#   lr=0.000828, beta1=0.9, beta2=0.95, eps=1e-15, weight_decay=0.1, grad_clip=1.0
#
# LR schedule: WSD-exp (warmup-stable-decay with exponential decay)
#   warmup_samples=64000, constant_samples=1110656, decay_end_samples=1388416
#   min_lr=8.28e-05 (lr * 0.1)
#
# Data: cybertron blended dataset (pretrain_v3), seq_len=8192
#   Prepared by: python prepare_cybertron_data.py --config 198
#   BlendedDataset hash: b98975b5a0b37ed3a7f5437dc5951576 (data_cache_path: /prodcpfs/user/data/save/data/data_cache)

# I/O
out_dir = 'out-cybertron-baseline'
eval_interval = 500
log_interval = 1
eval_iters = 20
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# wandb (set wandb_log=True to enable)
wandb_log = False
wandb_project = 'nanogpt-cybertron'
wandb_run_name = 'cybertron_baseline_198'

# Data
# Run `python prepare_cybertron_data.py` first to create train.bin/val.bin
dataset = 'cybertron_baseline_cybertron'   # suffix '_cybertron' → sequential loading
gradient_accumulation_steps = 32  # global_batch_size=128 / (micro_batch=4 * 1 gpu)
                                   # train.py divides by world_size internally → 4 per GPU on 8 GPUs
batch_size = 4                    # micro batch size (matches cybertron micro_batch_size=4)
block_size = 8192                 # seq_length

# Model architecture (dense equivalent of scaling_moe_00198)
n_layer = 16
n_head = 8
n_embd = 656
n_kv_head = 4             # GQA: 8 query heads, 4 KV heads
kv_channels = 64          # per-head dim = 64 (cybertron kv_channels=64)
                          # Q: 656→8*64=512, K/V: 656→4*64=256, O: 512→656
use_rope = True
rotary_base = 50000       # matches cybertron rotary_base=50000
use_rmsnorm = True
norm_eps = 1e-5
use_swiglu = True
ffn_hidden_size = 1920    # matches cybertron ffn_hidden_size=1920
qk_layernorm = True       # matches cybertron qk_layernorm=True (RMSNorm on Q/K before RoPE)
tie_embeddings = False    # cybertron: untie_embeddings_and_output_weights=true
init_std = 0.006          # matches cybertron init_method_std=0.006
disable_scaled_init_method = True  # cybertron: disable_scaled_init_method=true
dropout = 0.0
bias = False
vocab_size_override = 152064  # Qwen tokenizer (set via meta.pkl or hardcoded)

# Optimizer (matches cybertron's FusedAdam/AdamW)
learning_rate = 0.000828  # cybertron lr=0.000828
weight_decay = 0.1        # cybertron weight_decay=0.1
beta1 = 0.9               # cybertron adam_beta1=0.9
beta2 = 0.95              # cybertron adam_beta2=0.95
adam_eps = 1e-15          # cybertron adam_eps=1e-15
grad_clip = 1.0           # cybertron clip_grad=1.0

# LR schedule: wsd-exp, sample-based (matching cybertron convention)
decay_lr = True
lr_decay_style = 'wsd-exp'
min_lr = 8.28e-05         # cybertron min_lr = lr * 0.1 = 0.000828 * 0.1

# Sample-based scheduling (cybertron uses samples, not iterations)
# With global_batch_size=128:
#   warmup_samples=64000 → 500 iters
#   constant_samples=1110656 → 8677 iters
#   decay_end_samples=1388416 → 10847 iters (= exit_interval)
warmup_samples = 64000
constant_samples = 1110656
decay_end_samples = 1388416
global_batch_size = 128   # used to convert sample counts to iter counts

# max_iters = decay_end_samples / global_batch_size = 10847
max_iters = 10847

# System
device = 'cuda'
dtype = 'bfloat16'        # cybertron bf16=true
compile = False           # disable for deterministic mode

# Bitwise deterministic training
deterministic = True
seed = 1337
