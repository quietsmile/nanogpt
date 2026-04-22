"""Compare nano logits to ref logits at iter 5988, mbs0, batch 0."""
import torch, torch.nn.functional as F
import sys, numpy as np
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
    seq_aux_balance_alpha=0.0, use_eod_attn_mask=False)
model = GPT(cfg).cuda()
model.load_state_dict(sd, strict=False)
model.eval()

# Ref logits shape [8192, 4, 152064] — 8192 tokens × 4 batches × vocab
ref_logits = load_out(f"{DUMP}/output_layer-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")
print(f"ref_logits shape={ref_logits.shape} dtype={ref_logits.dtype}")
# Ref logits for batch 0 only: ref_logits[:, 0, :] shape [8192, 152064]
ref_b0 = ref_logits[:, 0, :]  # [T, V]

# Get batch 0 input tokens (ref mbs0 b0 maps to nano sample 383260 per our earlier finding)
data = np.memmap('/root/nanogpt/data/cybertron_baseline/train.bin', dtype=np.int32, mode='r')
block = 8192
# mbs0 b0 ref ⟷ nano sample 383260 (found previously by token match)
sample_idx = 383260
X = data[sample_idx*block : sample_idx*block + block].astype(np.int64)[None]  # [1, T]
X_t = torch.from_numpy(X).cuda()

# Forward nano, get logits only (no loss)
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    logits, _ = model(X_t, targets=None)
print(f"nano logits shape={logits.shape}")
# nano logits [B=1, T=1, V] — only last position because targets=None. Let me fix:
# Looking at model.py: if targets is None, logits = lm_head(x[:, [-1], :]) → only last
# Need targets to get full logits. Use dummy targets.
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    Y_dummy = torch.zeros_like(X_t)
    logits_full, _ = model(X_t, targets=Y_dummy)
print(f"full logits shape={logits_full.shape}")  # [1, T, V]

nano_b0 = logits_full[0]  # [T, V]
d = (nano_b0.float() - ref_b0.float()).abs()
print(f"\n=== logits diff (batch 0 only) ===")
print(f"  nano L1={nano_b0.abs().float().mean():.3e} max={nano_b0.abs().float().max():.3f}")
print(f"  ref  L1={ref_b0.abs().float().mean():.3e} max={ref_b0.abs().float().max():.3f}")
print(f"  |diff| L1={d.mean():.3e} max={d.max():.3e}")
rel = d.mean() / ref_b0.abs().float().mean()
print(f"  relative diff = {rel:.3e}")

# Compute per-token CE loss
target = torch.from_numpy(data[sample_idx*block + 1 : sample_idx*block + 1 + block].astype(np.int64)).cuda()  # [T]
with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
    nano_logprobs = F.log_softmax(nano_b0.float(), dim=-1)
    ref_logprobs = F.log_softmax(ref_b0.float(), dim=-1)
    nano_ce = -nano_logprobs.gather(1, target.unsqueeze(1)).squeeze(1)  # [T]
    ref_ce = -ref_logprobs.gather(1, target.unsqueeze(1)).squeeze(1)  # [T]

print(f"\n=== Per-token loss ===")
print(f"  nano mean CE = {nano_ce.mean():.6f}")
print(f"  ref  mean CE = {ref_ce.mean():.6f}")
print(f"  Δ = {(nano_ce - ref_ce).mean():+.6f}")
# Max per-token diff
ce_diff = (nano_ce - ref_ce).abs()
print(f"  per-token Δ L1={ce_diff.mean():.3e} max={ce_diff.max():.3e}")
# Where are the biggest diffs?
top5 = ce_diff.argsort(descending=True)[:5]
for i in top5:
    print(f"  t={i.item()}: nano_ce={nano_ce[i]:.3f} ref_ce={ref_ce[i]:.3f} target={target[i].item()}")
