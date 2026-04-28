"""nano dense 00108 — mirror scaling_dense_00108_nc.yaml from cybertron_exprs guofu_dev.

Architecture: n_layer=6, hidden=960, ffn=2960, n_head=8, n_kv_head=4, kv_channels=128.
Total params ~370M (vs 190M for 00107). Single 8-GPU node, 8002 iter.
"""
out_dir = "/prodcpfs/user/yuchen/scaling_exp/auto_test/v2_dense_108_full_s1337"
eval_interval = 20000
log_interval = 1
eval_iters = 10
always_save_checkpoint = True
init_from = "scratch"

dataset = "cybertron_baseline"
gradient_accumulation_steps = 64
batch_size = 1
block_size = 8192

# Architecture (mirrors scaling_dense_00108_nc.yaml)
n_layer = 6
n_head = 8
n_embd = 960
n_kv_head = 4
kv_channels = 128
use_rope = True
rotary_base = 50000
use_rmsnorm = True
norm_eps = 1e-5
use_swiglu = True
ffn_hidden_size = 2960
qk_layernorm = True
tie_embeddings = False
init_std = 0.006
disable_scaled_init_method = True
dropout = 0.0
bias = False
vocab_size_override = 152064

# Dense (no MoE)
use_moe = False

eod_token_id = 151643
mask_loss_id = 160000

# Optim
learning_rate = 0.000924
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
adam_eps = 1e-15
grad_clip = 1.0

decay_lr = True
lr_decay_style = "wsd-exp"
min_lr = 9.24e-05
warmup_samples = 32000
constant_samples = 480128
decay_end_samples = 512128
global_batch_size = 64
max_iters = 8002

device = "cuda"
dtype = "bfloat16"
compile = False
deterministic = False
chunked_ce = False
attention_impl = "te"
seed = 1337
