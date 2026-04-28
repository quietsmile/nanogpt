"""nano MoE 00197 Adam — mirror scaling_moe_00197_nc.yaml from cybertron_exprs guofu_dev.

Architecture: n_layer=10 (1 dense + 9 MoE), hidden=656, ffn=1920, moe_ffn=208,
n_head=8, n_kv_head=4, kv_channels=64. 144 experts, top-8.
Single 8-GPU, 8576 iter, gbs=80.
"""
out_dir = "/prodcpfs/user/yuchen/scaling_exp/auto_test/v2_moe_197_muon_full_s1337"
eval_interval = 20000
log_interval = 1
eval_iters = 10
always_save_checkpoint = True
init_from = "scratch"

dataset = "cybertron_baseline"
gradient_accumulation_steps = 80
batch_size = 1
block_size = 8192

n_layer = 10
n_head = 8
n_embd = 656
n_kv_head = 4
kv_channels = 64
use_rope = True
rotary_base = 50000
use_rmsnorm = True
norm_eps = 1e-5
use_swiglu = True
ffn_hidden_size = 1920
qk_layernorm = True
tie_embeddings = False
init_std = 0.006
disable_scaled_init_method = True
dropout = 0.0
bias = False
vocab_size_override = 152064

# MoE: layer 0 dense, layers 1-9 MoE (per yaml '[0]*1+[1]*9')
use_moe = True
moe_layer_freq = [0] + [1] * 9
num_experts = 144
moe_ffn_hidden_size = 208
moe_router_topk = 8
moe_n_group = 8
moe_topk_group = 1
moe_norm_topk_prob = True
moe_router_score_correction_coeff = 0.001
moe_shared_expert_hidden_size = 208
moe_routing_type = "greedy"

eod_token_id = 151643
mask_loss_id = 160000
seq_aux_balance_alpha = 0.0001
routed_scaling_factor = 1.0

learning_rate = 0.001
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
adam_eps = 1e-15
grad_clip = 1.0

decay_lr = True
lr_decay_style = "wsd-exp"
min_lr = 0.0001
warmup_samples = 40000
constant_samples = 600960
decay_end_samples = 686080
global_batch_size = 80
max_iters = 8576

device = "cuda"
dtype = "bfloat16"
compile = False
deterministic = False
chunked_ce = False
attention_impl = "te"
use_muon = True
muon_impl = "megatron_v2"
muon_lr = 0.001
muon_momentum = 0.95
muon_use_nesterov = True
muon_weight_decay = None
muon_use_decoupled_wd = True
muon_coefficient_type = "quintic"
muon_num_ns_steps = 5
muon_scale_mode = "spectral"
muon_matched_adamw_rms = 0.2
muon_fp32_matmul_prec = "medium"

seed = 1337
