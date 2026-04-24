# Config for nanogpt MoE model, aligned with scaling_moe_00196 (PAI DLC dlc1q9arre48b0kx).
#
# Source yaml: /prodcpfs/user/data/GitLab/pretrain_scaling_ladder/scaling_moe_00196.yaml
# Megatron ckpt for shape validation:
#   /prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/iter_0001497
# Diffs vs cybertron_moe_198.py: arch (9L/512d/4h/2qg vs 16L/656d/8h/4qg),
#   full LR schedule, global_batch 64 vs 128, tokenizer dots vs Qwen.
#   All four 00196 code gaps (eod_mask_loss, mask_loss_id, seq_aux balance,
#   accurate_attn_mask_eod_token) are implemented — see tests/test_code_gaps.py.
#
# Architecture (from Megatron state_dict, iter_0001497):
#   n_layer=9 (0 dense + 1-8 MoE)
#   n_embd=512, n_head=4, n_kv_head=2, kv_channels=64
#   Dense layer 0 FFN: ffn_hidden=1536 (fc1 (3072,512) = SwiGLU double, fc2 (512,1536))
#   MoE per-expert: moe_ffn_hidden=160 (fc1 (320,512), fc2 (512,160))
#   Shared expert per MoE layer: same shape as one routed expert (hidden=160)
#   num_experts=144, 8 groups × 18 experts, topk=8, topk_group=1 (sigmoid, norm_topk)
#
# Total params (verified by summing Megatron ckpt shards):
#   embedding+output_layer: 155.71M
#   other non-expert: ~8.47M (attn + qk_norm + layer_norm + layer0 dense + 8× shared_expert)
#   routed experts: 283.12M (144 × 8 × (320*512 + 512*160) bytes)
#   GRAND TOTAL: 447.30M

# I/O
out_dir = 'out-cybertron-moe-196-noise2-mb4-sdpa'
eval_interval = 500
log_interval = 1
eval_iters = 20
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# wandb
wandb_log = False
wandb_project = 'nanogpt-cybertron'
wandb_run_name = 'cybertron_moe_196'

# Data — same data_pretrain_v3 blend the reference used
dataset = 'cybertron_baseline'  # prepare_cybertron_data.py writes here regardless of --exp
gradient_accumulation_steps = 16   # global_batch=64 / (micro_batch=1 × 1 gpu); train.py ÷ world_size
batch_size = 4
block_size = 8192

# Model architecture
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
ffn_hidden_size = 1536        # dense layer 0
qk_layernorm = True
tie_embeddings = False
init_std = 0.006
disable_scaled_init_method = True
dropout = 0.0
bias = False
vocab_size_override = 152064  # dots_tokenizer (HF), padded

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
moe_routing_type = 'greedy'  # ref uses flat top-K over all experts (yaml: moe_router_load_balancing_type=greedy)

# 00196 code-gap flags — all wired in model.py/train.py, covered by tests/test_code_gaps.py
eod_token_id = 151643         # mask loss where idx == EOD (ref loss_mask[data == eod] = 0.0)
mask_loss_id = 160000          # extra token id excluded from loss (same input-based mechanism)
seq_aux_balance_alpha = 0.0001 # sequence_wise_balance_loss_alpha (only applied while model.training)
routed_scaling_factor = 1.0

# Optimizer
learning_rate = 0.0012
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
adam_eps = 1e-15
grad_clip = 1.0

# LR schedule (WSD-exp, samples-based as in Megatron)
decay_lr = True
lr_decay_style = 'wsd-exp'
min_lr = 1.2e-4
warmup_samples = 32000
constant_samples = 383232
decay_end_samples = 479040
global_batch_size = 64
max_iters = 7485

# System
device = 'cuda'
dtype = 'bfloat16'
compile = False
# Speed: see dense_107 speed_optim memory
deterministic = False
seed = 2000

# Fast path
chunked_ce = True
attention_impl = 'sdpa'
