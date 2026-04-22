"""Compare nano's Q/K/V (post qk_layernorm, post RoPE) vs ref's core_attention input."""
import torch, torch.nn.functional as F
import transformer_engine.pytorch as te
import sys
sys.path.insert(0, "/root/nanogpt"); sys.path.insert(0, "/root/nanogpt/scripts")
from megatron_to_nano import load_all_megatron_shards
from model import GPTConfig, GPT, RMSNorm, RotaryEmbedding

DUMP = "/newcpfs/user/yuchen/karpathy/cybertron_dump/dumps"
def load_out(p):
    t = torch.load(p, weights_only=False, map_location="cuda")
    return t[0] if isinstance(t, tuple) else t

meg = load_all_megatron_shards("/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/iter_0005988")

# Ref's core_attention inputs
core_in = torch.load(f"{DUMP}/decoder.layers.0.self_attention.core_attention-iter5988-mbs0-forward-input-tp0.1-pp0.1-ep3.4.pt", weights_only=False, map_location="cuda")
q_ref, k_ref, v_ref = core_in[0], core_in[1], core_in[2]
print(f"ref Q.shape={q_ref.shape} dtype={q_ref.dtype}")  # [T, B, H=4, D=64]
print(f"ref K.shape={k_ref.shape}")  # [T, B, 2, 64]
print(f"ref V.shape={v_ref.shape}")  # [T, B, 2, 64]

# Ref linear_qkv output (already verified bitwise with nano)
qkv_ref = load_out(f"{DUMP}/decoder.layers.0.self_attention.linear_qkv-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")
T, B, _ = qkv_ref.shape
# GQA-interleaved split: qkv layout = 2 groups × [Q0, Q1, K, V]
qkv5 = qkv_ref.view(T, B, 2, 4, 64)
q_nano_preln = qkv5[..., :2, :].reshape(T, B, 4, 64)  # [T, B, 4, 64]
k_nano_preln = qkv5[..., 2:3, :].reshape(T, B, 2, 64)
v_nano_preln = qkv5[..., 3:4, :].reshape(T, B, 2, 64)

# Apply qk_layernorm (TE's)
te_q_ln = te.RMSNorm(64, eps=1e-5, params_dtype=torch.float32, device="cuda")
te_q_ln.weight.data = meg["decoder.layers.0.self_attention.q_layernorm.weight"].float().cuda()
te_k_ln = te.RMSNorm(64, eps=1e-5, params_dtype=torch.float32, device="cuda")
te_k_ln.weight.data = meg["decoder.layers.0.self_attention.k_layernorm.weight"].float().cuda()

def diff(a, b, name):
    d = (a.float() - b.float()).abs()
    print(f"  {name}: L_inf={d.max():.3e} L1={d.mean():.3e} nonzero={(d>0).sum().item()}/{d.numel()}")

with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    q1 = te_q_ln(q_nano_preln)
    k1 = te_k_ln(k_nano_preln)

    # TE RoPE (sbhd format) — ref uses sbhd convention internally
    from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb
    inv_freq = 1.0 / (50000 ** (torch.arange(0, 64, 2, dtype=torch.float32, device="cuda") / 64))
    t_arange = torch.arange(T, device="cuda", dtype=torch.float32)
    freqs = torch.outer(t_arange, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1).view(T, 1, 1, 64)

    # TE RoPE with tensor_format='sbhd' - q/k shape [T, B, H, D]
    q_rope_te = apply_rotary_pos_emb(q1, emb, tensor_format='sbhd')
    k_rope_te = apply_rotary_pos_emb(k1, emb, tensor_format='sbhd')

    diff(q_rope_te, q_ref, "Q post qkLN+RoPE (TE) vs ref")
    diff(k_rope_te, k_ref, "K post qkLN+RoPE (TE) vs ref")
    diff(v_nano_preln, v_ref, "V (no ln, no rope) vs ref")

    # Also try nano's RoPE via manual impl matching nano model.py
    nano_qln = RMSNorm(64, eps=1e-5).cuda()
    nano_qln.weight.data = meg["decoder.layers.0.self_attention.q_layernorm.weight"].float().cuda()
    nano_kln = RMSNorm(64, eps=1e-5).cuda()
    nano_kln.weight.data = meg["decoder.layers.0.self_attention.k_layernorm.weight"].float().cuda()
    rope = RotaryEmbedding(64, base=50000, max_seq_len=8192).cuda()
    # nano takes [B, H, T, D]
    q_bhtd = q_nano_preln.permute(1, 2, 0, 3).contiguous()  # [B, 4, T, 64]
    k_bhtd = k_nano_preln.permute(1, 2, 0, 3).contiguous()
    q_n = nano_qln(q_bhtd); k_n = nano_kln(k_bhtd)
    q_n, k_n = rope(q_n, k_n, seq_len=T)
    # convert to sbhd for diff
    q_n_sbhd = q_n.permute(2, 0, 1, 3).contiguous()  # [T, B, 4, 64]
    k_n_sbhd = k_n.permute(2, 0, 1, 3).contiguous()
    diff(q_n_sbhd, q_ref, "Q post qkLN+RoPE (nano) vs ref")
    diff(k_n_sbhd, k_ref, "K post qkLN+RoPE (nano) vs ref")
