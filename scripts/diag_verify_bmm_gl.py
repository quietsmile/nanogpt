"""Verify bmm-bucketed vs te.GroupedLinear on actual block 1 scenario."""
import torch, torch.nn.functional as F
import transformer_engine.pytorch as te
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
mlp = model.transformer.h[1].mlp

ref_mlp_in = torch.load(f"{DUMP}/decoder.layers.1.mlp-iter5988-mbs0-forward-input-tp0.1-pp0.1-ep3.4.pt", weights_only=False, map_location="cuda")[0]
x_flat = ref_mlp_in.view(-1, 512)  # [T*B, H] TB-major matching ref
S, C = x_flat.shape
E = mlp.num_experts
K = mlp.topk
H = mlp.gate_weight.shape[-1]

with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    topk_idx, weights, _ = mlp.router(x_flat)
    routing_map = torch.zeros(S, E, dtype=torch.int32, device="cuda")
    routing_map.scatter_(1, topk_idx, 1)
    probs = torch.zeros(S, E, dtype=torch.float32, device="cuda")
    probs.scatter_(1, topk_idx, weights.float())

    # Permute tokens using TE
    permuted, perm_probs, row_id_map = te.moe_permute_with_probs(
        x_flat, probs, routing_map, num_out_tokens=S*K,
    )
    m_splits = routing_map.sum(dim=0).tolist()
    print(f"m_splits stats: min={min(m_splits)}, max={max(m_splits)}, sum={sum(m_splits)}")

    # Fused fc1 weight [E, 2H, C]
    fc1_w_T = torch.cat([mlp.gate_weight.transpose(1,2), mlp.up_weight.transpose(1,2)], dim=1).contiguous()  # [E, 2H, C]

    # Method A: F.linear per-expert
    a_out = torch.empty(permuted.shape[0], 2*H, dtype=torch.bfloat16, device="cuda")
    start = 0
    for e, m in enumerate(m_splits):
        if m == 0: continue
        end = start + m
        a_out[start:end] = F.linear(permuted[start:end], fc1_w_T[e])
        start = end

    # Method B: te.GroupedLinear
    gl = te.GroupedLinear(num_gemms=E, in_features=C, out_features=2*H, bias=False, params_dtype=torch.float32, device="cuda")
    for e in range(E):
        getattr(gl, f"weight{e}").data = fc1_w_T[e].float()
    b_out = gl(permuted, m_splits=m_splits)

    # Method C: bmm with bucket padding (nano's original)
    counts = routing_map.sum(dim=0)
    M_per = int(counts.max().item())
    if M_per < 8: M_per = 8
    # Need sorted tokens by expert
    flat_experts = topk_idx.reshape(-1)
    # But permuted already is in expert-major order (TE permute matches masked_select)
    # So I need the per-expert tokens from permuted
    bucket = torch.zeros(E, M_per, C, dtype=torch.bfloat16, device="cuda")
    start = 0
    for e, m in enumerate(m_splits):
        if m == 0: continue
        end = start + m
        bucket[e, :m] = permuted[start:end]
        start = end
    # bmm: [E, M_per, C] @ [E, C, 2H] → [E, M_per, 2H]
    gate_up_w = fc1_w_T.transpose(1, 2).contiguous()  # [E, C, 2H]
    c_out_bucket = torch.bmm(bucket, gate_up_w)
    # Gather used rows back to [N, 2H]
    c_out = torch.empty(permuted.shape[0], 2*H, dtype=torch.bfloat16, device="cuda")
    start = 0
    for e, m in enumerate(m_splits):
        if m == 0: continue
        end = start + m
        c_out[start:end] = c_out_bucket[e, :m]
        start = end

def diff(a, b, name):
    d = (a.float() - b.float()).abs()
    print(f"  {name}: L_inf={d.max():.3e} L1={d.mean():.3e} nonzero={(d>0).sum().item()}/{d.numel()}")

diff(a_out, b_out, "F.linear per-expert vs te.GroupedLinear")
diff(a_out, c_out, "F.linear per-expert vs bmm-bucketed")
diff(b_out, c_out, "te.GroupedLinear vs bmm-bucketed")
