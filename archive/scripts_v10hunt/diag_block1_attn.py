"""Compare block 1 attention components vs ref to find the 70x diff source."""
import torch, torch.nn.functional as F
import sys
sys.path.insert(0, "/root/nanogpt"); sys.path.insert(0, "/root/nanogpt/scripts")
from megatron_to_nano import load_all_megatron_shards, convert
from model import GPTConfig, GPT, RMSNorm, RotaryEmbedding

DUMP = "/newcpfs/user/yuchen/karpathy/cybertron_dump/dumps"
def load_out(p):
    t = torch.load(p, weights_only=False, map_location="cuda")
    return t[0] if isinstance(t, tuple) else t

meg = load_all_megatron_shards("/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/iter_0005988")

ref_b0 = load_out(f"{DUMP}/decoder.layers.0-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")  # [T,B,H]
ref_qkv_b1 = load_out(f"{DUMP}/decoder.layers.1.self_attention.linear_qkv-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")
ref_core_b1 = load_out(f"{DUMP}/decoder.layers.1.self_attention.core_attention-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")

T, B = 8192, 4

# Use ref's linear_qkv output directly → bitwise Q, K, V extraction
qkv5 = ref_qkv_b1.view(T, B, 2, 4, 64)
q_ref = qkv5[..., :2, :].reshape(T, B, 4, 64).permute(1, 2, 0, 3).contiguous()
k_ref = qkv5[..., 2:3, :].reshape(T, B, 2, 64).permute(1, 2, 0, 3).contiguous()
v_ref = qkv5[..., 3:4, :].reshape(T, B, 2, 64).permute(1, 2, 0, 3).contiguous()

# Apply qk_layernorm for block 1
q_ln_w = meg["decoder.layers.1.self_attention.q_layernorm.weight"].float().cuda()
k_ln_w = meg["decoder.layers.1.self_attention.k_layernorm.weight"].float().cuda()
nano_qln = RMSNorm(64, eps=1e-5).cuda(); nano_qln.weight.data = q_ln_w
nano_kln = RMSNorm(64, eps=1e-5).cuda(); nano_kln.weight.data = k_ln_w
rope = RotaryEmbedding(64, base=50000, max_seq_len=8192).cuda()

def diff(a, b, name):
    d = (a.float() - b.float()).abs()
    print(f"  {name}: L_inf={d.max():.3e} L1={d.mean():.3e} nonzero={(d>0).sum().item()}/{d.numel()}")

with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    q = nano_qln(q_ref); k = nano_kln(k_ref)
    # Check: post qk_layernorm max magnitudes (for block 1 vs block 0)
    print(f"Block 1 post-qkLN q.std={q.float().std():.4f} q.max_abs={q.float().abs().max():.4f}")
    print(f"Block 1 post-qkLN k.std={k.float().std():.4f} k.max_abs={k.float().abs().max():.4f}")

    q, k = rope(q, k, seq_len=T)
    k_exp = k.unsqueeze(2).expand(B, 2, 2, T, 64).reshape(B, 4, T, 64)
    v_exp = v_ref.unsqueeze(2).expand(B, 2, 2, T, 64).reshape(B, 4, T, 64)
    attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)
    attn_flat = attn_out.transpose(1, 2).contiguous().view(B, T, 256)
    diff(attn_flat.transpose(0, 1), ref_core_b1, "block 1 core_attention (nano SDPA)")

    # Compute max attention logits (Q@K^T/sqrt(d)) to check if they're larger
    qf = q.float(); kf = k_exp.float()
    att_logits = (qf @ kf.transpose(-2, -1)) / 8.0  # sqrt(64)=8
    print(f"  Max attn logit abs: {att_logits.abs().max():.4f}")
    print(f"  Attn logit std: {att_logits.std():.4f}")
