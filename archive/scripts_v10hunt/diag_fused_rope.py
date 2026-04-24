"""Test fused_apply_rotary_pos_emb vs ref's output."""
import torch, sys
sys.path.insert(0, "/root/nanogpt"); sys.path.insert(0, "/root/nanogpt/scripts")
import transformer_engine.pytorch as te
from megatron.core.extensions.transformer_engine import fused_apply_rotary_pos_emb
from megatron_to_nano import load_all_megatron_shards

DUMP = "/newcpfs/user/yuchen/karpathy/cybertron_dump/dumps"
meg = load_all_megatron_shards("/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/iter_0005988")

core_in = torch.load(f"{DUMP}/decoder.layers.0.self_attention.core_attention-iter5988-mbs0-forward-input-tp0.1-pp0.1-ep3.4.pt", weights_only=False, map_location="cuda")
q_ref_pr, k_ref_pr, v_ref_pr = core_in[0], core_in[1], core_in[2]

qkv_ref = torch.load(f"{DUMP}/decoder.layers.0.self_attention.linear_qkv-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt", weights_only=False, map_location="cuda")
if isinstance(qkv_ref, tuple): qkv_ref = qkv_ref[0]
T, B, _ = qkv_ref.shape
qkv5 = qkv_ref.view(T, B, 2, 4, 64)
q_pre = qkv5[..., :2, :].reshape(T, B, 4, 64)
k_pre = qkv5[..., 2:3, :].reshape(T, B, 2, 64)

te_q_ln = te.RMSNorm(64, eps=1e-5, params_dtype=torch.float32, device="cuda")
te_q_ln.weight.data = meg["decoder.layers.0.self_attention.q_layernorm.weight"].float().cuda()
te_k_ln = te.RMSNorm(64, eps=1e-5, params_dtype=torch.float32, device="cuda")
te_k_ln.weight.data = meg["decoder.layers.0.self_attention.k_layernorm.weight"].float().cuda()

def diff(a, b, name):
    d = (a.float() - b.float()).abs()
    print(f"  {name}: L_inf={d.max():.3e} L1={d.mean():.3e} nonzero={(d>0).sum().item()}/{d.numel()}")

# Freqs
inv_freq = 1.0 / (50000 ** (torch.arange(0, 64, 2, dtype=torch.float32, device="cuda") / 64))
t_arange = torch.arange(T, device="cuda", dtype=torch.float32)
freqs = torch.outer(t_arange, inv_freq)
emb = torch.cat((freqs, freqs), dim=-1).view(T, 1, 1, 64)

with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    q_ln = te_q_ln(q_pre)
    k_ln = te_k_ln(k_pre)
    # Apply fused_apply_rotary_pos_emb (what ref uses)
    q_fused = fused_apply_rotary_pos_emb(q_ln, emb, interleaved=False)
    k_fused = fused_apply_rotary_pos_emb(k_ln, emb, interleaved=False)
    diff(q_fused, q_ref_pr, "Q via fused_apply_rotary_pos_emb vs ref")
    diff(k_fused, k_ref_pr, "K via fused_apply_rotary_pos_emb vs ref")

    # Compare to unfused TE (what we had)
    from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb
    q_unfused = apply_rotary_pos_emb(q_ln, emb, tensor_format='sbhd')
    k_unfused = apply_rotary_pos_emb(k_ln, emb, tensor_format='sbhd')
    diff(q_unfused, q_ref_pr, "Q via apply_rotary_pos_emb (non-fused) vs ref")
    diff(k_unfused, k_ref_pr, "K via apply_rotary_pos_emb (non-fused) vs ref")
