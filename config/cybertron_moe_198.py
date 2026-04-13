# Config for nanogpt MoE model, aligned with scaling_moe_00198 actual architecture.
#
# Source: /prodcpfs/user/data/GitLab/pretrain_scaling_ladder/scaling_moe_00198.yaml
#
# Model architecture:
#   hidden_size=656, num_layers=16 (1 dense + 15 MoE), num_attention_heads=8,
#   num_query_groups=4, kv_channels=64
#   num_experts=144, topk=8, moe_ffn_hidden=224, shared_expert_hidden=224
#   dense_ffn_hidden=1920 (layer 0), moe_layer_freq=[0]+[1]*15
#   Grouped routing: n_group=8 (18 experts/group), topk_group=1
#   Group score = sum of top-(topk//topk_group)=8 scores per group → pick best group → top-8
#
# Parameter count (excl. wte+lmhead):
#   ~292M total routed params, ~24M active per token
#   wte+lmhead: 2 × 152064×656 = 199.5M
#
# Dense baseline (for comparison): config/cybertron_baseline.py
#   hidden=656, 16 layers, ffn_hidden=1920 all dense — same hidden/layers, no MoE

# I/O
out_dir = 'out-cybertron-moe-198'
eval_interval = 500
log_interval = 1
eval_iters = 20
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# wandb (set wandb_log=True to enable)
wandb_log = False
wandb_project = 'nanogpt-cybertron'
wandb_run_name = 'cybertron_moe_198'

# Data — same as dense baseline (pretrain_v3, 198 data cache)
dataset = 'cybertron_baseline_cybertron'
gradient_accumulation_steps = 128  # global_batch=128 / (micro_batch=1 × 1 gpu); ÷ world_size in train.py
batch_size = 1
block_size = 8192

# Model architecture — same hidden/layers as dense baseline, MoE on layers 1-15
n_layer = 16
n_head = 8
n_embd = 656
n_kv_head = 4
kv_channels = 64
use_rope = True
rotary_base = 50000
use_rmsnorm = True
norm_eps = 1e-5
use_swiglu = True
ffn_hidden_size = 1920      # dense layer 0 FFN hidden size (same as dense baseline)
qk_layernorm = True
tie_embeddings = False
init_std = 0.006
disable_scaled_init_method = True
dropout = 0.0
bias = False
vocab_size_override = 152064  # Qwen tokenizer

# MoE settings — grouped routing matching cybertron 198
use_moe = True
moe_layer_freq = [0] + [1] * 15   # layer 0 dense, layers 1-15 MoE
num_experts = 144                   # routed experts per MoE layer (8 groups × 18 per group)
moe_ffn_hidden_size = 224           # per-expert SwiGLU hidden (moe_ffn_hidden_size)
moe_router_topk = 8                 # top-8 per token
moe_n_group = 8                     # 144/8 = 18 experts per group
moe_topk_group = 1                  # select 1 group per token
moe_norm_topk_prob = True           # normalize top-8 scores to sum=1
moe_router_score_correction_coeff = 0.001  # bias update step for aux-free load balance
moe_shared_expert_hidden_size = 224  # always-on shared expert hidden size

# Optimizer — same as dense baseline (matches cybertron FusedAdam)
learning_rate = 0.000828
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
adam_eps = 1e-15
grad_clip = 1.0

# LR schedule: WSD-exp (same as dense baseline)
decay_lr = True
lr_decay_style = 'wsd-exp'
min_lr = 8.28e-05
warmup_samples = 64000
constant_samples = 1110656
decay_end_samples = 1388416
global_batch_size = 128
max_iters = 10847  # decay_end_samples / global_batch_size

# System
device = 'cuda'
dtype = 'bfloat16'
compile = False
deterministic = True
seed = 1337
