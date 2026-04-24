# nano-dense_107 starting from REF iter_0 weights.
# Bitwise-initial params via scripts/seed_from_meg_ckpt.py conversion of
# /prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_dense_00107_init/iter_0000000.
# Isolates the "numerical stack" axis (fp32 vs bf16, DDP order) from the
# "initialization RNG" axis, since weights at iter 0 are byte-identical.

# I/O
out_dir = 'out-cybertron-dense-107-from-ref'
eval_interval = 20000
log_interval = 1
eval_iters = 10
eval_only = False
always_save_checkpoint = True
init_from = 'resume'      # load ckpt.pt from out_dir

# wandb
wandb_log = False
wandb_project = 'nanogpt-cybertron-dense'
wandb_run_name = 'cybertron_dense_107_from_ref'

# Data
dataset = 'cybertron_baseline'
gradient_accumulation_steps = 48
batch_size = 1
block_size = 8192

# Model architecture — IDENTICAL to cybertron_dense_107.py
n_layer = 8
n_head = 4
n_embd = 528
n_kv_head = 2
kv_channels = 128
use_rope = True
rotary_base = 50000
use_rmsnorm = True
norm_eps = 1e-5
use_swiglu = True
ffn_hidden_size = 1824
qk_layernorm = True
tie_embeddings = False
init_std = 0.006
disable_scaled_init_method = True
dropout = 0.0
bias = False
vocab_size_override = 152064
use_moe = False

# Cybertron loss flags
eod_token_id = 151643
mask_loss_id = 160000
seq_aux_balance_alpha = 0.0
routed_scaling_factor = 1.0
chunked_ce = True

# Optimizer
learning_rate = 0.001063
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
adam_eps = 1e-15
grad_clip = 1.0

# LR schedule
decay_lr = True
lr_decay_style = 'wsd-exp'
min_lr = 0.0001063
warmup_samples = 24000
constant_samples = 257664
decay_end_samples = 322128
global_batch_size = 48
max_iters = 6711

# System
device = 'cuda'
dtype = 'bfloat16'
compile = False
deterministic = True
seed = 1337
