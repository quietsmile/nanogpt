"""Test full forward in fp32 (no autocast). If loss matches ref's 3.025 better,
the 0.047 gap confirms as bf16 compounding. If not, bug elsewhere."""
import torch, torch.nn.functional as F
import sys, numpy as np
sys.path.insert(0, "/root/nanogpt"); sys.path.insert(0, "/root/nanogpt/scripts")
from megatron_to_nano import load_all_megatron_shards, convert
from model import GPTConfig, GPT

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
    seq_aux_balance_alpha=0.0, use_eod_attn_mask=False)
model = GPT(cfg).cuda()
model.load_state_dict(sd, strict=False)
model.eval()

# Convert model to fp32 entirely (weights already fp32 in nano, but ensure no bf16 leaks)
# Don't need to change weights; just skip autocast below.

data = np.memmap('/root/nanogpt/data/cybertron_baseline/train.bin', dtype=np.int32, mode='r')
block = 8192
start = 5988 * 64
GBS = 64

def forward_loss_bf16():
    total, n = 0.0, 0
    for i in range(GBS):
        X_mb = data[(start+i)*block : (start+i)*block + block].astype(np.int64)[None]
        Y_mb = data[(start+i)*block + 1 : (start+i)*block + 1 + block].astype(np.int64)[None]
        X = torch.from_numpy(X_mb).cuda(); Y = torch.from_numpy(Y_mb).cuda()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(X, targets=Y)
        total += loss.item(); n += 1
        torch.cuda.empty_cache()
    return total / n

def forward_loss_fp32():
    total, n = 0.0, 0
    for i in range(GBS):
        X_mb = data[(start+i)*block : (start+i)*block + block].astype(np.int64)[None]
        Y_mb = data[(start+i)*block + 1 : (start+i)*block + 1 + block].astype(np.int64)[None]
        X = torch.from_numpy(X_mb).cuda(); Y = torch.from_numpy(Y_mb).cuda()
        with torch.no_grad():
            # No autocast → compute runs in fp32 (model weights are fp32)
            _, loss = model(X, targets=Y)
        total += loss.item(); n += 1
        torch.cuda.empty_cache()
    return total / n

print("Running bf16 forward (matches production)...")
l_bf = forward_loss_bf16()
print(f"  nano bf16 LM loss = {l_bf:.6f}")

print("Running fp32 forward (no autocast)...")
l_fp = forward_loss_fp32()
print(f"  nano fp32 LM loss = {l_fp:.6f}")

print(f"\nref iter 5988 logged lm_loss = 3.025030")
print(f"  Δ(nano_bf16 - ref) = {l_bf - 3.025030:+.6f}")
print(f"  Δ(nano_fp32 - ref) = {l_fp - 3.025030:+.6f}")
print(f"  Δ(nano_bf16 - nano_fp32) = {l_bf - l_fp:+.6f}  (pure bf16 effect)")
