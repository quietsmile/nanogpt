"""Check te.moe_unpermute vs manual scatter_add on REAL block 1 data."""
import torch, os, sys
sys.path.insert(0, "/root/nanogpt"); sys.path.insert(0, "/root/nanogpt/scripts")
import transformer_engine.pytorch as te
import torch.nn.functional as F
os.environ['NANO_TE_MOE'] = '0'  # use original nano MoE
from megatron_to_nano import load_all_megatron_shards, convert
from model import GPTConfig, GPT

DUMP = "/newcpfs/user/yuchen/karpathy/cybertron_dump/dumps"
def load_out(p):
    t = torch.load(p, weights_only=False, map_location="cuda")
    return t[0] if isinstance(t, tuple) else t

meg = load_all_megatron_shards("/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/iter_0005988")
sd = convert(meg)
cfg = GPTConfig(block_size=8192, vocab_size=152064, n_layer=9, n_head=4, n_embd=512, n_kv_head=2, kv_channels=64, dropout=0.0, bias=False, init_std=0.006, use_rope=True, rotary_base=50000, use_rmsnorm=True, norm_eps=1e-5, use_swiglu=True, ffn_hidden_size=1536, qk_layernorm=True, tie_embeddings=False, disable_scaled_init_method=True, use_moe=True, moe_layer_freq=[0]+[1]*8, num_experts=144, moe_ffn_hidden_size=160, moe_router_topk=8, moe_n_group=8, moe_topk_group=1, moe_norm_topk_prob=True, moe_router_score_correction_coeff=0.001, moe_shared_expert_hidden_size=160, moe_routing_type="greedy", eod_token_id=151643, mask_loss_id=160000, seq_aux_balance_alpha=0.0001, use_eod_attn_mask=False)
model = GPT(cfg).cuda()
model.load_state_dict(sd, strict=False)
model.eval()
mlp = model.transformer.h[1].mlp

ref_mlp_in = torch.load(f"{DUMP}/decoder.layers.1.mlp-iter5988-mbs0-forward-input-tp0.1-pp0.1-ep3.4.pt", weights_only=False, map_location="cuda")[0]
# T-major flat (matching ref)
x_flat = ref_mlp_in.view(-1, 512)
S, C = x_flat.shape
E, K = 144, 8
H = mlp.gate_weight.shape[-1]

with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    # Router
    topk_idx, weights, _ = mlp.router(x_flat)
    # Build probs/routing_map
    probs = torch.zeros(S, E, dtype=torch.float32, device="cuda")
    probs.scatter_(1, topk_idx, weights.float())
    routing_map = torch.zeros(S, E, dtype=torch.int32, device="cuda")
    routing_map.scatter_(1, topk_idx, 1)

    # TE permute + expert compute + unpermute
    permuted, perm_probs, row_id_map = te.moe_permute_with_probs(
        x_flat, probs, routing_map, num_out_tokens=S*K,
    )
    m_splits = routing_map.sum(dim=0).tolist()
    fc1_w_T = torch.cat([mlp.gate_weight.transpose(1,2), mlp.up_weight.transpose(1,2)], dim=1).contiguous()  # [E, 2H, C]
    fc2_w_T = mlp.down_weight.transpose(1,2).contiguous()
    h12 = torch.empty(permuted.shape[0], 2*H, dtype=torch.bfloat16, device="cuda")
    start = 0
    for e, m in enumerate(m_splits):
        if m == 0: continue
        end = start + m
        h12[start:end] = F.linear(permuted[start:end], fc1_w_T[e])
        start = end
    gate, up = h12.chunk(2, dim=-1)
    h_act = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
    out_perm = torch.empty(permuted.shape[0], C, dtype=torch.bfloat16, device="cuda")
    start = 0
    for e, m in enumerate(m_splits):
        if m == 0: continue
        end = start + m
        out_perm[start:end] = F.linear(h_act[start:end], fc2_w_T[e])
        start = end

    # Method TE: te.moe_unpermute
    routed_te = te.moe_unpermute(
        out_perm, row_id_map, merging_probs=probs,
        restore_shape=x_flat.shape, map_type='mask',
    )

    # Method Manual: Megatron non-fused unpermute
    routing_map_T = routing_map.T.contiguous().bool()
    token_indices = torch.arange(S, device="cuda").unsqueeze(0).expand(E, -1)
    sorted_indices = token_indices.masked_select(routing_map_T)  # expert-major
    permuted_probs_check = probs.T.contiguous().masked_select(routing_map_T)
    weighted_perm = out_perm.float() * permuted_probs_check.unsqueeze(-1)
    routed_manual = torch.zeros(S, C, dtype=torch.float32, device="cuda")
    routed_manual.scatter_add_(0, sorted_indices.unsqueeze(1).expand(-1, C), weighted_perm)
    routed_manual = routed_manual.to(torch.bfloat16)

def diff(a, b, name):
    d = (a.float() - b.float()).abs()
    print(f"  {name}: L_inf={d.max():.3e} L1={d.mean():.3e} nonzero={(d>0).sum().item()}/{d.numel()}")

diff(routed_te, routed_manual, "te.moe_unpermute vs manual scatter_add on REAL block 1")

# Compare to ref_mlp - ref_shared
ref_mlp = load_out(f"{DUMP}/decoder.layers.1.mlp-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")
ref_shared = load_out(f"{DUMP}/decoder.layers.1.mlp.shared_experts-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")
ref_routed = ref_mlp - ref_shared

diff(routed_te.view(8192, 4, 512), ref_routed, "te.moe_unpermute vs ref_routed (real)")
diff(routed_manual.view(8192, 4, 512), ref_routed, "manual scatter_add vs ref_routed (real)")
