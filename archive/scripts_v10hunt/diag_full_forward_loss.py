"""Full forward loss test: load nano with ref iter 5988 ckpt, feed ref's
iter 5988 training batch, compute loss, compare to ref logged lm_loss = 3.02503.
"""
import torch, torch.nn.functional as F
import sys
import numpy as np
sys.path.insert(0, "/root/nanogpt"); sys.path.insert(0, "/root/nanogpt/scripts")
from megatron_to_nano import load_all_megatron_shards, convert
from model import GPTConfig, GPT

DUMP = "/newcpfs/user/yuchen/karpathy/cybertron_dump/dumps"
def load_out(p):
    t = torch.load(p, weights_only=False, map_location="cuda")
    if isinstance(t, tuple): return t[0] if t else None
    return t

meg = load_all_megatron_shards("/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/iter_0005988")
sd = convert(meg)
cfg = GPTConfig(block_size=8192, vocab_size=152064, n_layer=9, n_head=4, n_embd=512,
    n_kv_head=2, kv_channels=64, dropout=0.0, bias=False, init_std=0.006, use_rope=True,
    rotary_base=50000, use_rmsnorm=True, norm_eps=1e-5, use_swiglu=True, ffn_hidden_size=1536,
    qk_layernorm=True, tie_embeddings=False, disable_scaled_init_method=True, use_moe=True,
    moe_layer_freq=[0]+[1]*8, num_experts=144, moe_ffn_hidden_size=160, moe_router_topk=8,
    moe_n_group=8, moe_topk_group=1, moe_norm_topk_prob=True,
    moe_router_score_correction_coeff=0.001, moe_shared_expert_hidden_size=160,
    moe_routing_type="greedy", eod_token_id=151643, mask_loss_id=160000,
    seq_aux_balance_alpha=0.0001, use_eod_attn_mask=False)
model = GPT(cfg).cuda()
model.load_state_dict(sd, strict=False)
model.eval()

# Ref at iter 5988 rank 7 (the single rank dumped) gives mbs0 b0-b3 + mbs1 b0-b3 = 8 samples
# But we need the CORRECT X, Y pair for that iter in nano's data order.
# Nano @ iter 5988: sample indices for rank r micro m batch b = r + 8*(N*g + m) + b*?
# Actually: _seq_data_pos for rank r at iter N = r + N * world_size * batch * grad_accum = r + 5988*64
# Each micro_step reads batch_size=1 samples at pos, then pos += 8
# So rank 0 iter 5988 micro_step 0: sample index = 5988*64 + 0 = 383232
# rank 0 iter 5988 micro_step m: sample 383232 + m*8 (m in 0..7)
# rank 7 iter 5988 micro_step m: sample 383232 + 7 + m*8

# Instead of replicating rank assignment, let's take the 64 samples consumed at iter 5988
# and run them as a single batch of 64. Sample indices [383232, 383232+64).
data = np.memmap('/root/nanogpt/data/cybertron_baseline/train.bin', dtype=np.int32, mode='r')
block = 8192
start = 5988 * 64  # iter 5988 start = sample index 383232
GBS = 64
# Run in micro-batches of 1 to fit memory
def forward_loss(model, mbs=1):
    total = 0.0
    total_aux = 0.0
    n = 0
    for i in range(0, GBS, mbs):
        X_mb = np.stack([data[(start+j)*block : (start+j)*block + block].astype(np.int64) for j in range(i, i+mbs)])
        Y_mb = np.stack([data[(start+j)*block + 1 : (start+j)*block + 1 + block].astype(np.int64) for j in range(i, i+mbs)])
        X_t = torch.from_numpy(X_mb).cuda()
        Y_t = torch.from_numpy(Y_mb).cuda()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(X_t, targets=Y_t)
        total += loss.item()
        n += 1
        del X_t, Y_t
        torch.cuda.empty_cache()
    return total / n
print(f"GBS = {GBS}, running 1 at a time...")

loss_with_aux = forward_loss(model, mbs=1)
print(f"nano avg loss (incl. aux α=0.0001) = {loss_with_aux:.6f}")

cfg2 = GPTConfig(**{**cfg.__dict__, 'seq_aux_balance_alpha': 0.0})
model2 = GPT(cfg2).cuda()
model2.load_state_dict(sd, strict=False)
model2.eval()
loss_lm_only = forward_loss(model2, mbs=1)
print(f"nano avg LM-only loss (aux α=0.0) = {loss_lm_only:.6f}")
print(f"ref   lm_loss logged at iter 5988 = 3.025030")
print(f"  Δ (nano_LM - ref) = {loss_lm_only.item() - 3.025030:+.6f}")
