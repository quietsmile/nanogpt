"""Verify: permute + (no-op) + unpermute + sum_weights = identity if merging_probs sums to 1."""
import torch
import transformer_engine.pytorch as te
torch.manual_seed(42)

S = 32768
E = 144
K = 8
C = 512

# Random input
x = torch.randn(S, C, dtype=torch.bfloat16, device="cuda") * 0.1

# Random topk indices (random routing)
topk_idx = torch.randint(0, E, (S, K), device="cuda")
# Ensure unique per row
for i in range(10):
    perm = torch.randperm(E, device="cuda")[:K]
topk_idx = torch.stack([torch.randperm(E, device="cuda")[:K] for _ in range(S)])
# Random weights summing to 1
weights = torch.rand(S, K, dtype=torch.float32, device="cuda")
weights = weights / weights.sum(dim=-1, keepdim=True)

# Build probs and routing_map
probs = torch.zeros(S, E, dtype=torch.float32, device="cuda")
probs.scatter_(1, topk_idx, weights)
routing_map = torch.zeros(S, E, dtype=torch.int32, device="cuda")
routing_map.scatter_(1, topk_idx, 1)

num_out = S * K

# Permute
permuted, permuted_probs, row_id_map = te.moe_permute_with_probs(
    x, probs, routing_map, num_out_tokens=num_out,
)
print(f"permuted.shape={permuted.shape}")

# Sanity: permuted_probs sum per unique token
# With sum(weights)=1 across K, each token's K contributions sum to 1 over all permuted rows belonging to it
# Unpermute directly (no expert computation — pass permuted back)
out = te.moe_unpermute(permuted, row_id_map, merging_probs=probs,
                       restore_shape=x.shape, map_type='mask')
# out[t] = sum over K of prob_{t,k} * permuted[permuted_idx_of_t_k] = sum_k prob_k * x[t] = x[t] * sum(prob)=x[t]
d = (out.float() - x.float()).abs()
print(f"permute+unpermute identity (sum_weights=1): L_inf={d.max():.3e} L1={d.mean():.3e} nonzero={(d>0).sum().item()}/{d.numel()}")
