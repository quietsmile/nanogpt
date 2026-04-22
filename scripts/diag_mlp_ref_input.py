"""Feed ref's exact mlp input (post_attn) to nano's MLP, compare outputs."""
import torch, torch.nn.functional as F
import sys
sys.path.insert(0, "/root/nanogpt")
sys.path.insert(0, "/root/nanogpt/scripts")
from megatron_to_nano import load_all_megatron_shards, convert
from model import GPTConfig, GPT

DUMP = "/newcpfs/user/yuchen/karpathy/cybertron_dump/dumps"
meg = load_all_megatron_shards("/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/iter_0005988")
sd = convert(meg)
cfg = GPTConfig(block_size=8192, vocab_size=152064, n_layer=9, n_head=4, n_embd=512, n_kv_head=2, kv_channels=64, dropout=0.0, bias=False, init_std=0.006, use_rope=True, rotary_base=50000, use_rmsnorm=True, norm_eps=1e-5, use_swiglu=True, ffn_hidden_size=1536, qk_layernorm=True, tie_embeddings=False, disable_scaled_init_method=True, use_moe=True, moe_layer_freq=[0]+[1]*8, num_experts=144, moe_ffn_hidden_size=160, moe_router_topk=8, moe_n_group=8, moe_topk_group=1, moe_norm_topk_prob=True, moe_router_score_correction_coeff=0.001, moe_shared_expert_hidden_size=160, moe_routing_type='greedy', eod_token_id=151643, mask_loss_id=160000, seq_aux_balance_alpha=0.0001, use_eod_attn_mask=False)
model = GPT(cfg).cuda()
model.load_state_dict(sd, strict=False)
model.eval()
b0 = model.transformer.h[0]

# Ref's exact mlp input (TBH format)
ref_mlp_in_tuple = torch.load(f"{DUMP}/decoder.layers.0.mlp-iter5988-mbs0-forward-input-tp0.1-pp0.1-ep3.4.pt", weights_only=False, map_location="cuda")
ref_mlp_in = ref_mlp_in_tuple[0]  # [T, B, H]

# Convert to nano [B, T, H]
mlp_in_bth = ref_mlp_in.transpose(0, 1).contiguous()
B, T, H = mlp_in_bth.shape

# Ref mlp, fc1, fc2 outputs (load via helper)
def load_out(p):
    t = torch.load(p, weights_only=False, map_location="cuda")
    return t[0] if isinstance(t, tuple) else t

ref_fc1 = load_out(f"{DUMP}/decoder.layers.0.mlp.linear_fc1-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")
ref_fc2 = load_out(f"{DUMP}/decoder.layers.0.mlp.linear_fc2-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")
ref_mlp_out = load_out(f"{DUMP}/decoder.layers.0.mlp-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")

def diff(a, b, name):
    d = (a.float() - b.float()).abs()
    print(f"  {name}: L_inf={d.max():.3e} L1={d.mean():.3e} nonzero={(d>0).sum().item()}/{d.numel()}")

with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    # Nano: ln_2 + gate/up (fc1 equivalent)
    h_ln = b0.ln_2(mlp_in_bth)
    # Megatron fc1 layout: weight = [gate; up] concatenated, output = ln(x) @ [gate;up]
    # So nano equivalent: concat(gate_proj(h), up_proj(h), dim=-1)
    nano_fc1 = torch.cat([b0.mlp.gate_proj(h_ln), b0.mlp.up_proj(h_ln)], dim=-1)
    # Compare to ref fc1 (TBH format, 2*ffn_hidden=3072)
    diff(nano_fc1.transpose(0, 1), ref_fc1, "fc1 output (ref's mlp input)")

    # Also check: does nano.ln_2(ref_mlp_in) match what TE LNL would produce?
    # LN output is an intermediate inside LayerNormLinear — we don't have it dumped.
    # But we can compute nano's ln_2(x) and check vs te.RMSNorm(x):
    import transformer_engine.pytorch as te
    te_rms = te.RMSNorm(512, eps=1e-5, params_dtype=torch.float32, device="cuda")
    te_rms.weight.data = meg["decoder.layers.0.mlp.linear_fc1.layer_norm_weight"].float().cuda()
    te_h = te_rms(mlp_in_bth)
    diff(h_ln, te_h, "ln_2 output: nano RMSNorm vs te.RMSNorm")

    # Now: nano_fc1 vs te.LayerNormLinear(mlp_in, ln_w, fc1_w)
    te_lnl_fc1 = te.LayerNormLinear(512, 3072, eps=1e-5, bias=False, normalization="RMSNorm",
                                     params_dtype=torch.float32, device="cuda")
    te_lnl_fc1.layer_norm_weight.data = meg["decoder.layers.0.mlp.linear_fc1.layer_norm_weight"].float().cuda()
    te_lnl_fc1.weight.data = meg["decoder.layers.0.mlp.linear_fc1.weight"].float().cuda()
    te_fc1_out = te_lnl_fc1(mlp_in_bth)
    diff(nano_fc1, te_fc1_out, "fc1: nano (ln_2 + fc1) vs te.LayerNormLinear")
    diff(te_fc1_out.transpose(0, 1), ref_fc1, "te.LayerNormLinear output vs ref fc1")

    # --- Check silu*up ---
    gate, up = nano_fc1.chunk(2, dim=-1)
    # fp32 silu*up (current nano fix)
    h_act_new = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
    # Old nano (bf16 silu*up)
    h_act_old = F.silu(gate) * up
    # Compare each vs what te would do
    te_act_module = __import__("transformer_engine.pytorch.ops", fromlist=["SwiGLU"]).SwiGLU()
    h_act_te = te_act_module(nano_fc1)
    diff(h_act_new, h_act_te, "fp32 silu*up vs te.SwiGLU")
    diff(h_act_old, h_act_te, "bf16 silu*up vs te.SwiGLU")

    # Then fc2
    nano_fc2 = b0.mlp.down_proj(h_act_new)
    diff(nano_fc2.transpose(0, 1), ref_fc2, "fc2 output with ref mlp_in + current nano")
    diff(nano_fc2.transpose(0, 1), ref_mlp_out, "mlp output vs ref mlp output")
