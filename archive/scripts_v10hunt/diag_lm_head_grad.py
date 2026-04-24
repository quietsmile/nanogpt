"""Compare nano's lm_head grad to ref's reconstructed lm_head grad.
Ref lm_head grad_W = sum_batch(dL/dy ⊗ x) where:
  - x = output_layer forward-input (final_hidden) [T, B, C=512]
  - dL/dy = output_layer backward-OUTPUT (grad at logits) [T, B, V=152064]
  - W shape [V, C]
Result: grad_W = einsum('tbv,tbc->vc', dL/dy, x)"""
import torch, torch.nn.functional as F
import sys, numpy as np
sys.path.insert(0, "/root/nanogpt"); sys.path.insert(0, "/root/nanogpt/scripts")
from megatron_to_nano import load_all_megatron_shards, convert
from model import GPTConfig, GPT

DUMP = "/newcpfs/user/yuchen/karpathy/cybertron_dump/dumps_allranks"
def load_out(p):
    t = torch.load(p, weights_only=False, map_location="cuda")
    if isinstance(t, tuple): return t[0] if t else None
    return t

# --- Reconstruct ref's lm_head grad from dumps ---
# We have 4 (dp, mbs) pairs. Sum across them gives this rank's contribution.
# For COMPARISON with nano on same samples, we want to process the exact
# same 64 samples nano uses. Sum across all 4 combinations gets us 16 samples.
# To match 64 samples we'd need all DP+EP combos; we only have ep=3 (=16 samples).

# Pick one dp+mbs pair first to verify bitwise match.
dp = 3
mbs = 0
x = load_out(f"{DUMP}/output_layer-iter5988-mbs{mbs}-forward-input-tp0.1-pp0.1-ep3.4-dp{dp}.8.pt")
dL_dy = load_out(f"{DUMP}/output_layer-iter5988-mbs{mbs}-backward-output-tp0.1-pp0.1-ep3.4-dp{dp}.8.pt")
print(f"x shape: {x.shape}, dtype={x.dtype}")
print(f"dL/dy shape: {dL_dy.shape}, dtype={dL_dy.dtype}")
print(f"x max_abs = {x.abs().max():.3f}, dL/dy max_abs = {dL_dy.abs().max():.3e}")
print(f"dL/dy L1 = {dL_dy.abs().float().mean():.3e}")

# Reconstruct W grad for this 4-sample mini-batch
# x: [T, B, C] → [T*B, C]; dL/dy: [T, B, V] → [T*B, V]
Tv, Bv, C = x.shape
_, _, V = dL_dy.shape
x_flat = x.reshape(-1, C).float()
dy_flat = dL_dy.reshape(-1, V).float()
# grad_W [V, C] = dy_flat^T @ x_flat
ref_grad_W_4samp = dy_flat.T @ x_flat
print(f"\nref lm_head.grad_W for this mb (4 samples): shape={ref_grad_W_4samp.shape}")
print(f"  norm = {ref_grad_W_4samp.norm().item():.4e}")
print(f"  mean abs = {ref_grad_W_4samp.abs().mean().item():.4e}")
print(f"  max abs = {ref_grad_W_4samp.abs().max().item():.4e}")

# Now nano: run forward+backward on EXACT SAME 4 samples, get lm_head.grad
# First need to identify which 4 nano samples correspond to this (dp3 mbs0).
# From verify_samples: dp3 mbs0 → nano indices 383244-383247
sample_indices = [383244, 383245, 383246, 383247]
data = np.memmap('/root/nanogpt/data/cybertron_baseline/train.bin', dtype=np.int32, mode='r')
block = 8192
X_list = [torch.from_numpy(data[s*block : s*block + block].astype(np.int64)) for s in sample_indices]
Y_list = [torch.from_numpy(data[s*block + 1 : s*block + 1 + block].astype(np.int64)) for s in sample_indices]

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
model.train()
model.lm_head.weight.grad = None

# Run forward+backward on 4 samples separately, accumulate grads
# To match ref's mb-style (4 samples at once), we'd do batch=4.
# But memory: lm_head logits [4, 8192, 152064] fp32 = 18GB, OOM.
# Process 1 sample at a time, sum grads.
total_n_unmasked = 0
grad_accum_strategy = 'ref_style'  # mb_loss = sum/n_tokens_in_mb, scaled by 1 (not /num_mb since only 1 mb)

# Ref-style: 1 mb of 4 samples
# mb_loss = sum(ce*mask) / sum(mask)  (token-weighted over 4 samples)
# loss for backward: mb_loss / num_microbatches
# For single-mb test: num_microbatches = 1 → backward uses mb_loss as-is
# Total grad = d(mb_loss)/dw = (1/n_tokens_mb) × sum_mb_tokens d(CE)/dw

# Since OOM prevents 4-sample batch, simulate by processing 1 sample at a time
# with scaling 1 / total_tokens_in_mb (not per-sample normalization).

# Count total tokens first
EOD = 151643
n_tokens_per_sample = []
for i, (X, Y) in enumerate(zip(X_list, Y_list)):
    mask = (X != EOD).sum().item()
    n_tokens_per_sample.append(mask)
total_tokens = sum(n_tokens_per_sample)
print(f"\ntotal unmasked tokens across 4 samples: {total_tokens}")

# Now run: per sample, compute CE sum (not mean), scale by 1/total_tokens, backward
for i, (X, Y) in enumerate(zip(X_list, Y_list)):
    X = X.unsqueeze(0).cuda()
    Y = Y.unsqueeze(0).cuda()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(X, targets=Y)
        logits_fp32 = logits.float()
        mask = (X.view(-1) != EOD)
        # CE per token
        ce = F.cross_entropy(logits_fp32.view(-1, logits_fp32.size(-1)), Y.view(-1),
                              ignore_index=-1, reduction='none')
        # Token-weighted mb loss (across 4 samples): sum_ce_all_samples / total_tokens
        # For this single sample contribution to mb loss:
        sample_contrib = (ce * mask.float()).sum() / total_tokens
        sample_contrib.backward()
    del X, Y, logits, logits_fp32
    torch.cuda.empty_cache()

nano_lm_head_grad = model.lm_head.weight.grad
print(f"\nnano lm_head.grad shape: {nano_lm_head_grad.shape}")
print(f"  norm     = {nano_lm_head_grad.norm().item():.4e}")
print(f"  mean abs = {nano_lm_head_grad.abs().mean().item():.4e}")
print(f"  max abs  = {nano_lm_head_grad.abs().max().item():.4e}")

# Compare
d = (nano_lm_head_grad.float() - ref_grad_W_4samp).abs()
print(f"\n=== nano lm_head grad vs reconstructed ref lm_head grad ===")
print(f"  |diff| L1  = {d.mean().item():.4e}")
print(f"  |diff| max = {d.max().item():.4e}")
print(f"  cosine = {torch.nn.functional.cosine_similarity(nano_lm_head_grad.float().flatten().unsqueeze(0), ref_grad_W_4samp.flatten().unsqueeze(0)).item():.6f}")
ratio = nano_lm_head_grad.norm() / ref_grad_W_4samp.norm()
print(f"  norm ratio nano/ref = {ratio.item():.4f}")
