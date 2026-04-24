import torch
import sys
sys.path.insert(0, "/root/nanogpt"); sys.path.insert(0, "/root/nanogpt/scripts")
from megatron_to_nano import load_all_megatron_shards, convert
from model import GPTConfig, GPT

DUMP = "/newcpfs/user/yuchen/karpathy/cybertron_dump/dumps"

meg = load_all_megatron_shards("/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/iter_0005988")
sd = convert(meg)
cfg = GPTConfig(block_size=8192, vocab_size=152064, n_layer=9, n_head=4, n_embd=512, n_kv_head=2, kv_channels=64, dropout=0.0, bias=False, init_std=0.006, use_rope=True, rotary_base=50000, use_rmsnorm=True, norm_eps=1e-5, use_swiglu=True, ffn_hidden_size=1536, qk_layernorm=True, tie_embeddings=False, disable_scaled_init_method=True, use_moe=True, moe_layer_freq=[0]+[1]*8, num_experts=144, moe_ffn_hidden_size=160, moe_router_topk=8, moe_n_group=8, moe_topk_group=1, moe_norm_topk_prob=True, moe_router_score_correction_coeff=0.001, moe_shared_expert_hidden_size=160, moe_routing_type="greedy", eod_token_id=151643, mask_loss_id=160000, seq_aux_balance_alpha=0.0001, use_eod_attn_mask=False)
model = GPT(cfg).cuda()
model.load_state_dict(sd, strict=False)
model.eval()

ref_mlp_in = torch.load(f"{DUMP}/decoder.layers.1.mlp-iter5988-mbs0-forward-input-tp0.1-pp0.1-ep3.4.pt", weights_only=False, map_location="cuda")[0]
ref_router = torch.load(f"{DUMP}/decoder.layers.1.mlp.router-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt", weights_only=False, map_location="cuda")
ref_probs, ref_rmap = ref_router[0], ref_router[1]

x_flat = ref_mlp_in.transpose(0,1).contiguous().view(-1, 512)
b1 = model.transformer.h[1]

with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    topk_idx, weights, _ = b1.mlp.router(x_flat)
    # weights should already be fp32 (after my earlier router fix)
    print(f"nano weights dtype: {weights.dtype}")

# Build nano's full-E probs tensor (fp32, like ref's)
S, K = topk_idx.shape
probs_nano = torch.zeros(S, 144, dtype=torch.float32, device="cuda")
probs_nano.scatter_(1, topk_idx, weights.float())

rmap_nano = torch.zeros(S, 144, dtype=torch.bool, device="cuda")
rmap_nano.scatter_(1, topk_idx, True)

# Diff
d_probs = (probs_nano - ref_probs.float()).abs()
print(f"probs diff: L_inf={d_probs.max():.3e} L1={d_probs.mean():.3e} nonzero={(d_probs>0).sum().item()}/{d_probs.numel()}")

rmap_match = (rmap_nano == ref_rmap).all(dim=-1)
print(f"routing_map row-match: {rmap_match.sum()}/{rmap_match.numel()}")
if not rmap_match.all():
    # How many experts differ per row on mismatching
    diff_per_row = (rmap_nano != ref_rmap).sum(dim=-1) / 2  # each differing expert = 2 flips
    print(f"mean experts differ per mismatching row: {diff_per_row[~rmap_match].float().mean().item():.3f}")
