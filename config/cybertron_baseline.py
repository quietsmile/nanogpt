# Config for nanogpt cybertron baseline, aligned with scaling_moe_00196.
#
# Reference PAI job: dlc1q9arre48b0kx (scaling_moe_00196)
# Reference config: /prodcpfs/user/data/GitLab/pretrain_scaling_ladder/scaling_moe_00196.yaml
#
# Model: dense equivalent of scaling_moe_00196 (no MoE, ffn_hidden=1536 for all layers)
#   hidden_size=512, n_layers=9, n_heads=4, n_kv_heads=2
#   RMSNorm, RoPE (base=50000), SwiGLU (ffn_hidden=1536), no bias, no weight tying
#   vocab_size=152064 (Qwen tokenizer, padded to multiple of 64)
#
# Optimizer: AdamW (decoupled weight decay, matches FusedAdam in cybertron)
#   lr=0.0012, beta1=0.9, beta2=0.95, eps=1e-15, weight_decay=0.1, grad_clip=1.0
#
# LR schedule: WSD-exp (warmup-stable-decay with exponential decay)
#   warmup_samples=32000, constant_samples=383232, decay_end_samples=479040
#   min_lr=0.00012 (lr/10)
#
# Data: cybertron blended dataset (pretrain_v3 + ...), seq_len=8192
#   Prepared by: python prepare_cybertron_data.py

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
wandb_run_name = 'cybertron_baseline_adam'

# Data
# Run `python prepare_cybertron_data.py` first to create train.bin/val.bin
dataset = 'cybertron_baseline_cybertron'   # suffix '_cybertron' → sequential loading
gradient_accumulation_steps = 16  # global_batch_size=64 / (micro_batch=4 * 1 gpu)
batch_size = 4                    # micro batch size (matches cybertron micro_batch_size=4)
block_size = 8192                 # seq_length

# Model architecture (dense equivalent of scaling_moe_00196)
n_layer = 9
n_head = 4
n_embd = 512
n_kv_head = 2             # GQA: 4 query heads, 2 KV heads
use_rope = True
rotary_base = 50000       # matches cybertron rotary_base=50000
use_rmsnorm = True
norm_eps = 1e-5
use_swiglu = True
ffn_hidden_size = 1536    # matches cybertron ffn_hidden_size=1536
tie_embeddings = False    # cybertron: untie_embeddings_and_output_weights=true
init_std = 0.006          # matches cybertron init_method_std=0.006
dropout = 0.0
bias = False
vocab_size_override = 152064  # Qwen tokenizer (set via meta.pkl or hardcoded)

# Optimizer (matches cybertron's FusedAdam/AdamW)
learning_rate = 0.0012    # cybertron lr=0.0012
weight_decay = 0.1        # cybertron weight_decay=0.1
beta1 = 0.9               # cybertron adam_beta1=0.9
beta2 = 0.95              # cybertron adam_beta2=0.95
adam_eps = 1e-15          # cybertron adam_eps=1e-15
grad_clip = 1.0           # cybertron clip_grad=1.0

# LR schedule: wsd-exp, sample-based (matching cybertron convention)
decay_lr = True
lr_decay_style = 'wsd-exp'
min_lr = 0.00012          # cybertron min_lr = lr * 0.1 = 0.00012

# Sample-based scheduling (cybertron uses samples, not iterations)
# With global_batch_size=64:
#   warmup_samples=32000 → 500 iters
#   constant_samples=383232 → 5988 iters
#   decay_end_samples=479040 → 7485 iters (= exit_interval)
warmup_samples = 32000
constant_samples = 383232
decay_end_samples = 479040
global_batch_size = 64    # used to convert sample counts to iter counts

# max_iters = decay_end_samples / global_batch_size = 7485
max_iters = 7485

# System
device = 'cuda'
dtype = 'bfloat16'        # cybertron bf16=true
compile = False           # disable for deterministic mode

# Bitwise deterministic training
deterministic = True
seed = 1337
