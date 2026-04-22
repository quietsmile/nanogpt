"""Tiny config for bitwise resume test — smallest possible MoE model matching
00196 architecture so the test runs in <1min per path on 8xL20Z/H100.

Used by scripts/bitwise_resume_test.sh. Same arch knobs as cybertron_moe_196
so the resume logic exercises the real MoE path (router + expert bias + fp32
weighted sum etc).
"""
out_dir = 'out-bw-test'
eval_interval = 10  # save ckpt at this boundary
log_interval = 1
eval_iters = 2
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'
wandb_log = False

dataset = 'cybertron_baseline'
gradient_accumulation_steps = 8  # after DP=8 → 1 per rank; single-GPU → 8
batch_size = 1
block_size = 1024  # smaller to keep step-time low

n_layer = 3  # 1 dense + 2 MoE
n_head = 4
n_embd = 256
n_kv_head = 2
kv_channels = 64
use_rope = True
rotary_base = 50000
use_rmsnorm = True
norm_eps = 1e-5
use_swiglu = True
ffn_hidden_size = 768
qk_layernorm = True
tie_embeddings = False
init_std = 0.006
disable_scaled_init_method = True
dropout = 0.0
bias = False
vocab_size_override = 152064

use_moe = True
moe_layer_freq = [0, 1, 1]
num_experts = 16
moe_ffn_hidden_size = 128
moe_router_topk = 4
moe_n_group = 4
moe_topk_group = 1
moe_norm_topk_prob = True
moe_router_score_correction_coeff = 0.001
moe_shared_expert_hidden_size = 128
moe_routing_type = 'greedy'

eod_token_id = 151643
mask_loss_id = 160000
seq_aux_balance_alpha = 0.0001
use_eod_attn_mask = False

learning_rate = 0.0012
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
adam_eps = 1e-15
grad_clip = 1.0

decay_lr = False
min_lr = 1.2e-4
global_batch_size = 8
max_iters = 10  # overridden per-path

device = 'cuda'
dtype = 'bfloat16'
compile = False
deterministic = True
seed = 1337
