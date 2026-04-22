# Ablation config: identical to cybertron_moe_196.py but with ALL custom extensions DISABLED.
# Goal: isolate whether eod_mask_loss / mask_loss_id / seq_aux / eod_attn_mask contribute
# to the nano-vs-ref loss gap observed in the 2000-step main run.
#
# If this 500-step run's loss curve matches ref better than the main 196 run → extensions
# are hurting. If it's the same or worse → the 1.8 nat gap comes from arch/numerical/init.

# I/O
out_dir = 'out-cybertron-moe-196-bare'
eval_interval = 500
log_interval = 1
eval_iters = 20
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# wandb
wandb_log = False
wandb_project = 'nanogpt-cybertron'
wandb_run_name = 'cybertron_moe_196_bare'

# Data
dataset = 'cybertron_baseline'
gradient_accumulation_steps = 64
batch_size = 1
block_size = 8192

# Model architecture (same as 196)
n_layer = 9
n_head = 4
n_embd = 512
n_kv_head = 2
kv_channels = 64
use_rope = True
rotary_base = 50000
use_rmsnorm = True
norm_eps = 1e-5
use_swiglu = True
ffn_hidden_size = 1536
qk_layernorm = True
tie_embeddings = False
init_std = 0.006
disable_scaled_init_method = True
dropout = 0.0
bias = False
vocab_size_override = 152064

# MoE
use_moe = True
moe_layer_freq = [0] + [1] * 8
num_experts = 144
moe_ffn_hidden_size = 160
moe_router_topk = 8
moe_n_group = 8
moe_topk_group = 1
moe_norm_topk_prob = True
moe_router_score_correction_coeff = 0.001
moe_shared_expert_hidden_size = 160

# *** ABLATION: disable all custom extensions ***
eod_token_id = None            # no eod mask loss
mask_loss_id = None            # no extra mask
seq_aux_balance_alpha = 0.0    # disable aux loss
routed_scaling_factor = 1.0
# use_eod_attn_mask defaults to False in GPTConfig

# Optimizer
learning_rate = 0.0012
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
adam_eps = 1e-15
grad_clip = 1.0

# LR schedule
decay_lr = True
lr_decay_style = 'wsd-exp'
min_lr = 1.2e-4
warmup_samples = 32000
constant_samples = 383232
decay_end_samples = 479040
global_batch_size = 64
max_iters = 500

# System
device = 'cuda'
dtype = 'bfloat16'
compile = False
deterministic = True
seed = 1337
