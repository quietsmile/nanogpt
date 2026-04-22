"""Test nano SDPA vs ref core_attention given identical Q/K/V."""
import torch, torch.nn.functional as F
import transformer_engine.pytorch as te
import sys
sys.path.insert(0, "/root/nanogpt")
sys.path.insert(0, "/root/nanogpt/scripts")
from megatron_to_nano import load_all_megatron_shards
from model import GPTConfig, GPT, RMSNorm, RotaryEmbedding

DUMP = "/newcpfs/user/yuchen/karpathy/cybertron_dump/dumps"

meg = load_all_megatron_shards("/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/iter_0005988")

# Load ref linear_qkv output (bitwise matched to nano's QKV)
qkv_ref = torch.load(f"{DUMP}/decoder.layers.0.self_attention.linear_qkv-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt", weights_only=False, map_location="cuda")
if isinstance(qkv_ref, tuple): qkv_ref = qkv_ref[0]
# shape [T, B, 512]
print("qkv_ref shape:", qkv_ref.shape)

T, B, _ = qkv_ref.shape
# Megatron interleaved GQA layout: [g0_Q0,g0_Q1,g0_K,g0_V, g1_Q0,g1_Q1,g1_K,g1_V]
qkv5 = qkv_ref.view(T, B, 2, 4, 64)
q_ref = qkv5[..., :2, :].reshape(T, B, 4, 64).permute(1, 2, 0, 3).contiguous()  # [B, 4, T, 64]
k_ref = qkv5[..., 2:3, :].reshape(T, B, 2, 64).permute(1, 2, 0, 3).contiguous()  # [B, 2, T, 64]
v_ref = qkv5[..., 3:4, :].reshape(T, B, 2, 64).permute(1, 2, 0, 3).contiguous()

# Apply qk_layernorm
q_ln_w = meg["decoder.layers.0.self_attention.q_layernorm.weight"].float().cuda()
k_ln_w = meg["decoder.layers.0.self_attention.k_layernorm.weight"].float().cuda()

def diff(a, b, name):
    d = (a.float() - b.float()).abs()
    print(f"  {name}: L_inf={d.max():.3e} L1={d.mean():.3e} nonzero={(d>0).sum().item()}/{d.numel()}")

# Method A: nano path — RMSNorm, RoPE, SDPA
nano_qln = RMSNorm(64, eps=1e-5).cuda(); nano_qln.weight.data = q_ln_w.clone()
nano_kln = RMSNorm(64, eps=1e-5).cuda(); nano_kln.weight.data = k_ln_w.clone()
rope = RotaryEmbedding(64, base=50000, max_seq_len=8192).cuda()

with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    q_a = nano_qln(q_ref)
    k_a = nano_kln(k_ref)
    q_a, k_a = rope(q_a, k_a, seq_len=T)
    k_a_exp = k_a.unsqueeze(2).expand(B, 2, 2, T, 64).reshape(B, 4, T, 64)
    v_a_exp = v_ref.unsqueeze(2).expand(B, 2, 2, T, 64).reshape(B, 4, T, 64)
    attn_sdpa = F.scaled_dot_product_attention(q_a, k_a_exp, v_a_exp, is_causal=True)
    attn_sdpa_flat = attn_sdpa.transpose(1, 2).contiguous().view(B, T, 256)

# Method B: TE DotProductAttention
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    te_qln = te.RMSNorm(64, eps=1e-5, params_dtype=torch.float32, device="cuda")
    te_qln.weight.data = q_ln_w.clone()
    te_kln = te.RMSNorm(64, eps=1e-5, params_dtype=torch.float32, device="cuda")
    te_kln.weight.data = k_ln_w.clone()
    q_b = te_qln(q_ref)
    k_b = te_kln(k_ref)
    q_b, k_b = rope(q_b, k_b, seq_len=T)  # use nano RoPE (same impl)
    # TE DotProductAttention expects [B, T, H, D] bshd
    q_bshd = q_b.transpose(1, 2).contiguous()
    k_bshd = k_b.transpose(1, 2).contiguous()
    v_bshd = v_ref.transpose(1, 2).contiguous()
    te_attn_mod = te.DotProductAttention(num_attention_heads=4, kv_channels=64, num_gqa_groups=2,
                                          attention_dropout=0.0, qkv_format='bshd', attn_mask_type='causal')
    attn_te = te_attn_mod(q_bshd, k_bshd, v_bshd)  # [B, T, H*D=256]

# Compare each against ref core_attention
ref_core = torch.load(f"{DUMP}/decoder.layers.0.self_attention.core_attention-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt", weights_only=False, map_location="cuda")
if isinstance(ref_core, tuple): ref_core = ref_core[0]
print("ref_core shape:", ref_core.shape)

# Convert nano attn output to TBH format matching ref
attn_sdpa_tbh = attn_sdpa_flat.transpose(0, 1)  # [T, B, 256]
attn_te_tbh = attn_te.transpose(0, 1)  # [T, B, 256]

diff(attn_sdpa_tbh, ref_core, "SDPA (nano RMSNorm qkLN) vs ref core_attention")
diff(attn_te_tbh, ref_core, "TE DotProductAttention (te RMSNorm qkLN) vs ref core_attention")
diff(attn_sdpa_tbh, attn_te_tbh, "SDPA vs TE attn output (same Q/K/V)")
