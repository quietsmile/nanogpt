# Dense variant of cybertron_moe_196 — ALL 9 layers use dense SwiGLU (ffn=1536).
# Matches /newcpfs/user/yuchen/karpathy/dense_196/scaling_dense_00196.yaml so
# nano-dense and ref-dense trajectories can be directly compared. If dense
# loss tracks ref much better than MoE does, the gap is MoE-specific.

# I/O
out_dir = 'out-cybertron-dense-196'
eval_interval = 20000
log_interval = 1
eval_iters = 10
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'  # No seed ckpt; init fresh (ref also inits fresh for this ablation)

# wandb
wandb_log = False
wandb_project = 'nanogpt-cybertron-dense'
wandb_run_name = 'cybertron_dense_196'

# Data
dataset = 'cybertron_baseline'
gradient_accumulation_steps = 16   # DP=8, mb=4 → 8*4*16 = 64 = gbs
batch_size = 4
block_size = 8192

# Model architecture — same as scaling_dense_00196.yaml
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

# No MoE — explicitly off so nano uses dense SwiGLU on every layer
use_moe = False

# Cybertron loss flags (same as MoE config)
eod_token_id = 151643
mask_loss_id = 160000
seq_aux_balance_alpha = 0.0   # aux only meaningful for MoE; disabled here
routed_scaling_factor = 1.0

# Chunked CE to avoid OOM at mb=4 with V=152064
chunked_ce = True

# Optimizer
learning_rate = 0.0012
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
adam_eps = 1e-15
grad_clip = 1.0

# LR schedule (WSD-exp, samples-based, same as MoE 00196)
decay_lr = True
lr_decay_style = 'wsd-exp'
min_lr = 1.2e-4
warmup_samples = 32000
constant_samples = 383232
decay_end_samples = 479040
global_batch_size = 64
max_iters = 50   # quick ablation — just enough to see whether dense aligns nano↔ref

# System
device = 'cuda'
dtype = 'bfloat16'
compile = False
deterministic = True
seed = 1337
