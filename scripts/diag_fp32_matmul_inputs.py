"""Hypothesis: ref's LayerNormLinear keeps LN output in fp32 precision internally;
nano casts to bf16 before matmul. This loses 8 bits before the matmul reduction.

Test: run nano's block 0 but force all matmul INPUTS to be fp32 (via autocast
disabled), keep weights fp32, let matmul output bf16 for residual stream."""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPT_DIR)


def build_nano():
    from model import GPTConfig, GPT
    cfg = GPTConfig(
        block_size=8192, vocab_size=152064, n_layer=9, n_head=4, n_embd=512,
        n_kv_head=2, kv_channels=64, dropout=0.0, bias=False, init_std=0.006,
        use_rope=True, rotary_base=50000, use_rmsnorm=True, norm_eps=1e-5,
        use_swiglu=True, ffn_hidden_size=1536,
        qk_layernorm=True, tie_embeddings=False, disable_scaled_init_method=True,
        use_moe=True, moe_layer_freq=[0] + [1] * 8, num_experts=144,
        moe_ffn_hidden_size=160, moe_router_topk=8, moe_n_group=8, moe_topk_group=1,
        moe_norm_topk_prob=True, moe_router_score_correction_coeff=0.001,
        moe_shared_expert_hidden_size=160, moe_routing_type='greedy',
        eod_token_id=151643, mask_loss_id=160000,
        seq_aux_balance_alpha=0.0001, use_eod_attn_mask=False,
    )
    return GPT(cfg), cfg


def manual_block0_fp32_matmul(block, x):
    """Block 0 forward but all matmul inputs cast to fp32 explicitly (no autocast)."""
    import math
    B, T, C = x.shape
    attn = block.attn
    # Pre-attn RMSNorm (already fp32-internal)
    ln1 = block.ln_1
    norm = x.float().pow(2).mean(-1, keepdim=True).add(ln1.eps).rsqrt()
    h_fp32 = x.float() * norm * ln1.weight.float()  # [B, T, C] fp32 — not cast to bf16!

    # QKV matmul in fp32 matmul, then cast output to bf16 for residual-consistent dtype
    with torch.amp.autocast('cuda', enabled=False):
        q = F.linear(h_fp32, attn.q_proj.weight.float()).to(torch.bfloat16)
        k = F.linear(h_fp32, attn.k_proj.weight.float()).to(torch.bfloat16)
        v = F.linear(h_fp32, attn.v_proj.weight.float()).to(torch.bfloat16)
    q = q.view(B, T, attn.n_head, attn.head_dim).transpose(1, 2)
    k = k.view(B, T, attn.n_kv_head, attn.head_dim).transpose(1, 2)
    v = v.view(B, T, attn.n_kv_head, attn.head_dim).transpose(1, 2)
    if attn.q_layernorm is not None:
        q = attn.q_layernorm(q)
        k = attn.k_layernorm(k)
    q, k = attn.rotary_emb(q, k, seq_len=T)
    if attn.n_rep > 1:
        k = k.unsqueeze(2).expand(B, attn.n_kv_head, attn.n_rep, T, attn.head_dim).reshape(B, attn.n_head, T, attn.head_dim)
        v = v.unsqueeze(2).expand(B, attn.n_kv_head, attn.n_rep, T, attn.head_dim).reshape(B, attn.n_head, T, attn.head_dim)
    a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    a = a.transpose(1, 2).contiguous().view(B, T, -1)
    # c_proj with fp32 matmul input
    with torch.amp.autocast('cuda', enabled=False):
        attn_out = F.linear(a.float(), attn.c_proj.weight.float()).to(torch.bfloat16)
    x = x + attn_out

    # Pre-MLP RMSNorm → fp32 h
    ln2 = block.ln_2
    norm2 = x.float().pow(2).mean(-1, keepdim=True).add(ln2.eps).rsqrt()
    h2_fp32 = x.float() * norm2 * ln2.weight.float()

    # SwiGLU MLP with fp32 inputs
    mlp = block.mlp
    with torch.amp.autocast('cuda', enabled=False):
        gate = F.linear(h2_fp32, mlp.gate_proj.weight.float())
        up = F.linear(h2_fp32, mlp.up_proj.weight.float())
        h_mlp = F.silu(gate) * up  # fp32
        mlp_out = F.linear(h_mlp, mlp.down_proj.weight.float()).to(torch.bfloat16)
    x = x + mlp_out
    return x


def main():
    DUMP_DIR = '/newcpfs/user/yuchen/karpathy/cybertron_dump/dumps'
    CKPT_DIR = '/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/iter_0005988'

    device = torch.device('cuda:0')
    from megatron_to_nano import load_all_megatron_shards, convert
    meg = load_all_megatron_shards(CKPT_DIR)
    sd = convert(meg)
    model, cfg = build_nano()
    model = model.to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    # Load ref embedding output (bitwise input to block 0)
    ref_embed = torch.load(f'{DUMP_DIR}/embedding-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt',
                           weights_only=False, map_location='cpu')
    ref_b0 = torch.load(f'{DUMP_DIR}/decoder.layers.0-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt',
                        weights_only=False, map_location='cpu')
    ref_b0 = ref_b0[0] if isinstance(ref_b0, tuple) else ref_b0

    x_input = ref_embed.transpose(0, 1).contiguous().cuda()  # [B, T, H] bf16

    def diff(out, tag):
        a = out.transpose(0, 1).contiguous().cpu().float()
        b = ref_b0.float()
        d = (a - b).abs()
        rel = d.mean() / b.abs().mean().clamp_min(1e-8)
        print(f'  {tag}: L∞={d.max().item():.4e}  L1={d.mean().item():.4e}  rel_mean={rel.item():.4e}')

    print('Block 0 output vs ref_b0:')
    # (a) Baseline nano (bf16 autocast)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out, _ = model.transformer.h[0](x_input)
    diff(out, '(a) nano default autocast-bf16')

    # (b) Force fp32 matmul inputs
    with torch.no_grad():
        out_fp32in = manual_block0_fp32_matmul(model.transformer.h[0], x_input)
    diff(out_fp32in, '(b) fp32 matmul inputs, bf16 outputs')

    # (c) Full fp32 (no autocast anywhere)
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
        out_fp32, _ = model.transformer.h[0](x_input.float())
    diff(out_fp32, '(c) full fp32 forward')


if __name__ == '__main__':
    main()
