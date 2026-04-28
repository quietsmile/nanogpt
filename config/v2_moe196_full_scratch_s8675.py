"""Stage 6B — full 7485-iter from scratch with refactor-v2 code (NANOGPT_V2_MODEL=1).

Target: tail-100 EMA within 1σ of v1.0 bucket fleet (2.8463 ± 0.0017, 3-seed).
Identical to cybertron_moe_196_pai_v10repro_bucket_full but init_from=scratch.
"""
out_dir = "/prodcpfs/user/yuchen/scaling_exp/auto_test/v2_moe196_full_scratch_s8675"
eval_interval = 20000
log_interval = 1
eval_iters = 10
always_save_checkpoint = True
init_from = "scratch"

dataset = "cybertron_baseline"
gradient_accumulation_steps = 64
batch_size = 1
block_size = 8192

n_layer = 9; n_head = 4; n_embd = 512; n_kv_head = 2; kv_channels = 64
use_rope = True; rotary_base = 50000; use_rmsnorm = True; norm_eps = 1e-5
use_swiglu = True; ffn_hidden_size = 1536
qk_layernorm = True; tie_embeddings = False; init_std = 0.006
disable_scaled_init_method = True; dropout = 0.0; bias = False
vocab_size_override = 152064

use_moe = True
moe_layer_freq = [0] + [1] * 8
num_experts = 144; moe_ffn_hidden_size = 160
moe_router_topk = 8; moe_n_group = 8; moe_topk_group = 1
moe_norm_topk_prob = True; moe_router_score_correction_coeff = 0.001
moe_shared_expert_hidden_size = 160; moe_routing_type = "greedy"

eod_token_id = 151643; mask_loss_id = 160000
seq_aux_balance_alpha = 0.0001; routed_scaling_factor = 1.0

learning_rate = 0.0012; weight_decay = 0.1
beta1 = 0.9; beta2 = 0.95; adam_eps = 1e-15; grad_clip = 1.0
decay_lr = True; lr_decay_style = "wsd-exp"
min_lr = 1.2e-4; warmup_samples = 32000
constant_samples = 383232; decay_end_samples = 479040
global_batch_size = 64; max_iters = 7485

device = "cuda"; dtype = "bfloat16"; compile = False
deterministic = False
chunked_ce = False
attention_impl = "te"
seed = 8675
