"""Given identical permuted expert outputs, does nano scatter_add == te.moe_unpermute?"""
import torch
import transformer_engine.pytorch as te
torch.manual_seed(42)

S = 4000
E = 144
K = 8
C = 512

# Random topk
topk_idx = torch.stack([torch.randperm(E, device="cuda")[:K] for _ in range(S)])
# Weights summing to 1
w = torch.rand(S, K, dtype=torch.float32, device="cuda")
w = w / w.sum(dim=-1, keepdim=True)

probs = torch.zeros(S, E, dtype=torch.float32, device="cuda")
probs.scatter_(1, topk_idx, w)
routing_map = torch.zeros(S, E, dtype=torch.int32, device="cuda")
routing_map.scatter_(1, topk_idx, 1)

# Permute some random tokens to get permuted_tokens + row_id_map
x = torch.randn(S, C, dtype=torch.bfloat16, device="cuda") * 0.5
permuted, permuted_probs, row_id_map = te.moe_permute_with_probs(
    x, probs, routing_map, num_out_tokens=S*K,
)
# For unpermute test, fabricate expert outputs: simulate expert computation
# Take permuted_tokens (representing tokens fed to expert) and "transform" them
# → permuted_tokens * some_constant per expert
permuted_outputs = permuted * 2.0  # pretend experts doubled the tokens

# Method A: te.moe_unpermute
out_te = te.moe_unpermute(
    permuted_outputs, row_id_map, merging_probs=probs,
    restore_shape=x.shape, map_type='mask',
)

# Method B: manual scatter_add (nano's path)
# sorted_indices (= row_id_map content) needs to be derived.
# Actually for nano, we need to know which token in original array each permuted row belongs to.
# The row_id_map from TE is an internal mapping. Hard to extract directly.
# So let's use Megatron's non-fused logic: masked_select
routing_map_T = routing_map.T.contiguous().bool()
token_indices = torch.arange(S, device="cuda").unsqueeze(0).expand(E, -1)
sorted_indices = token_indices.masked_select(routing_map_T)  # [S*K] order: expert-major
# permuted_probs (permute output) corresponds to this order
# But TE's permute might produce different order. Let me try
# Verify by checking: te.moe_permute's permuted_tokens == x.index_select(0, sorted_indices)?
# If same → same order
permuted_check = x.index_select(0, sorted_indices)
d_check = (permuted.float() - permuted_check.float()).abs()
print(f"TE permute order matches masked_select: L_inf={d_check.max():.3e}, max_nnz={(d_check>0).sum().item()}")

# Do manual unpermute with same order
permuted_probs_check = probs.T.contiguous().masked_select(routing_map_T)
# Weighted sum: permuted_outputs * permuted_probs_check
weighted = permuted_outputs.float() * permuted_probs_check.unsqueeze(-1)
out_manual = torch.zeros(S, C, dtype=torch.float32, device="cuda")
out_manual.scatter_add_(0, sorted_indices.unsqueeze(1).expand(-1, C), weighted)
out_manual = out_manual.to(torch.bfloat16)

d = (out_te.float() - out_manual.float()).abs()
print(f"te.moe_unpermute vs manual scatter_add (same order, fp32 weighted sum):")
print(f"  L_inf = {d.max():.3e}")
print(f"  L1 = {d.mean():.3e}")
print(f"  nonzero = {(d>0).sum().item()}/{d.numel()}")
