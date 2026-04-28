# Dense 00107 variant — matches
# /newcpfs/user/yuchen/karpathy/cybertron_exprs_guofu/llm/scaling/scaling_dense_00107_nc.yaml
# (guofu_dev branch). Full 6711-iter trajectory.
#
# Architectural differences vs cybertron_dense_196:
#   n_layer 9 → 8
#   hidden   512 → 528
#   ffn     1536 → 1824
#   kv_channels 64 → 128  (Q = 4*128 = 512 = hidden)
#   gbs 64 → 48  (mb=1, DP=8 → grad_accum = 6)
#   lr 0.0012 → 0.001063, min_lr 1.2e-4 → 1.063e-4
#   warmup_samples 32000 → 24000
#   constant_samples 383232 → 257664
#   decay_end_samples 479040 → 322128
#   max_iters 7485 → 6711
#
# NOTE: ref uses data_cache /prodcpfs/user/data/save/data/scaling/data_cache while
# nano uses data/cybertron_baseline (blend from /lossalign/data_cache). Data is NOT
# identical — but tokenizer is (same MD5), and architecture/optim config IS identical,
# so nano-vs-ref systematic drift (Adam / bf16 / DDP) should still manifest.

# I/O
out_dir = '/prodcpfs/user/yuchen/scaling_exp/auto_test/v2_dense_107_full_s300'
eval_interval = 20000
log_interval = 1
eval_iters = 10
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# wandb
wandb_log = False
wandb_project = 'nanogpt-cybertron-dense'
wandb_run_name = 'cybertron_dense_107'

# Data
dataset = 'cybertron_baseline'
gradient_accumulation_steps = 48   # nano convention: total across DP ranks. DP=8 → 6/rank, mb=1 → gbs=8*1*6=48
batch_size = 1
block_size = 8192

# Model architecture — matches scaling_dense_00107_nc.yaml
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

# Dense — no MoE (ref yaml has moe_layer_freq='[0]*8' meaning all layers dense)
use_moe = False

# Cybertron loss flags
eod_token_id = 151643
mask_loss_id = 160000
seq_aux_balance_alpha = 0.0
routed_scaling_factor = 1.0

# Speed: mb=1 dense 107 fits lm_head logits (mb×S×V×4B ≈ 5GB) without chunking.
chunked_ce = False
# Speed: TE FlashAttention is ~2× faster than SDPA-math fallback and loses nothing here.
attention_impl = 'te'

# Optimizer
learning_rate = 0.001063
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
adam_eps = 1e-15
grad_clip = 1.0

# LR schedule (WSD-exp, samples-based, per ref yaml)
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
# Speed: deterministic=True forces SDPA math fallback + disables TF32. Off → ~1.5-2× faster.
# Same-seed bitwise reproducibility is lost; seed-to-seed variance study is unaffected.
deterministic = False
seed = 300
