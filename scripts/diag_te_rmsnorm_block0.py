"""Confirmed: te.RMSNorm + F.linear bitwise == te.LayerNormLinear. So if we swap
nano's RMSNorm with te.RMSNorm, and keep F.linear, block 0 should match ref far
better than current 0.0009 L1. Test: replace nano's ln_1 and ln_2 with te.RMSNorm
in block 0, feed ref embedding, measure block 0 output diff."""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import transformer_engine.pytorch as te

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

    # Swap block 0's ln_1, ln_2, and q_layernorm, k_layernorm with te.RMSNorm
    b0 = model.transformer.h[0]
    # Save ref weights to load into te.RMSNorm
    orig_ln1_w = b0.ln_1.weight.data.clone()
    orig_ln2_w = b0.ln_2.weight.data.clone()
    orig_q_ln_w = b0.attn.q_layernorm.weight.data.clone() if b0.attn.q_layernorm else None
    orig_k_ln_w = b0.attn.k_layernorm.weight.data.clone() if b0.attn.k_layernorm else None

    # Create TE RMSNorm replacements
    te_ln1 = te.RMSNorm(cfg.n_embd, eps=cfg.norm_eps, params_dtype=torch.float32, device=device)
    te_ln1.weight.data = orig_ln1_w.float()
    te_ln2 = te.RMSNorm(cfg.n_embd, eps=cfg.norm_eps, params_dtype=torch.float32, device=device)
    te_ln2.weight.data = orig_ln2_w.float()
    te_q_ln = te.RMSNorm(cfg.kv_channels, eps=cfg.norm_eps, params_dtype=torch.float32, device=device)
    te_q_ln.weight.data = orig_q_ln_w.float()
    te_k_ln = te.RMSNorm(cfg.kv_channels, eps=cfg.norm_eps, params_dtype=torch.float32, device=device)
    te_k_ln.weight.data = orig_k_ln_w.float()

    b0.ln_1 = te_ln1
    b0.ln_2 = te_ln2
    b0.attn.q_layernorm = te_q_ln
    b0.attn.k_layernorm = te_k_ln

    # Load ref embedding as block 0 input
    ref_embed = torch.load(f'{DUMP_DIR}/embedding-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt',
                           weights_only=False, map_location='cpu')
    ref_b0 = torch.load(f'{DUMP_DIR}/decoder.layers.0-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt',
                        weights_only=False, map_location='cpu')
    ref_b0 = ref_b0[0] if isinstance(ref_b0, tuple) else ref_b0

    x_input = ref_embed.transpose(0, 1).contiguous().to(device)

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out, _ = b0(x_input)

    a = out.transpose(0, 1).contiguous().cpu().float()
    b = ref_b0.float()
    d = (a - b).abs()
    rel_mean = d.mean() / b.abs().mean().clamp_min(1e-8)
    print(f'Block 0 diff with te.RMSNorm swaps (ln_1, ln_2, q/k_layernorm):')
    print(f'  shape={list(a.shape)}')
    print(f'  L∞={d.max().item():.4e}')
    print(f'  L1={d.mean().item():.4e}')
    print(f'  rel_mean={rel_mean.item():.4e}')
    print()
    print(f'Compare to nano default (L1=9.3e-4, rel_mean=2.0e-3)')


if __name__ == '__main__':
    main()
