"""Isolated per-layer diff: feed ref's layer(i-1) output as input to nano layer i.
Remove accumulation, see each layer's standalone contribution.

Target: ≤0.0001 level per-layer diff. Anything larger = investigate."""
import os, sys
import numpy as np
import torch

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


def load_ref_tensor(dump_dir, name, mbs=0):
    p = f'{dump_dir}/{name}-iter5988-mbs{mbs}-forward-output-tp0.1-pp0.1-ep3.4.pt'
    t = torch.load(p, weights_only=False, map_location='cpu')
    if isinstance(t, tuple):
        t = t[0]
    return t  # [T, B, H]


def stat(a, b, name):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    rel = d.mean() / b.abs().mean().clamp_min(1e-8)
    rel_max = d.max() / b.abs().mean().clamp_min(1e-8)
    print(f'  {name}: L∞={d.max().item():.3e} L1={d.mean().item():.3e} rel_mean={rel.item():.3e} rel_max={rel_max.item():.3e}')
    return d.mean().item()


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

    # Load ref's layer outputs
    ref = {}
    ref['embed'] = load_ref_tensor(DUMP_DIR, 'embedding')  # [T, B, H]
    for i in range(9):
        ref[f'b{i}'] = load_ref_tensor(DUMP_DIR, f'decoder.layers.{i}')

    # Convert ref tensors from [T, B, H] to nano's [B, T, H] and push to GPU bf16
    def to_nano(t):
        # ref is bf16 [T, B, H]. Move to cuda as bf16, transpose to [B, T, H]
        return t.transpose(0, 1).contiguous().cuda()  # bf16

    ref_gpu = {k: to_nano(v) for k, v in ref.items()}

    print(f'\n=== ISOLATED PER-LAYER DIFF (nano block i on ref layer-(i-1) output) ===')
    print(f'{"layer":<8}{"L∞":>12}{"L1":>12}{"rel_mean":>12}{"rel_max":>12}')

    for i in range(9):
        inp_ref = ref_gpu['embed'] if i == 0 else ref_gpu[f'b{i-1}']
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            block = model.transformer.h[i]
            out, _ = block(inp_ref)
        # out: [B, T, H]; transpose to [T, B, H] to match ref
        out_tb = out.transpose(0, 1).contiguous().cpu()
        ref_i = ref[f'b{i}']
        a, b = out_tb.float(), ref_i.float()
        d = (a - b).abs()
        rel_mean = (d.mean() / b.abs().mean().clamp_min(1e-8)).item()
        rel_max = (d.max() / b.abs().mean().clamp_min(1e-8)).item()
        print(f'block{i:<4}{d.max().item():>12.4e}{d.mean().item():>12.4e}{rel_mean:>12.4e}{rel_max:>12.4e}')

    # Also check ln_f
    ref_ln_f = load_ref_tensor(DUMP_DIR, 'decoder.final_layernorm')
    inp = to_nano(ref['b8'])
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out = model.transformer.ln_f(inp)
    a, b = out.transpose(0, 1).contiguous().cpu().float(), ref_ln_f.float()
    d = (a - b).abs()
    rel_mean = (d.mean() / b.abs().mean().clamp_min(1e-8)).item()
    rel_max = (d.max() / b.abs().mean().clamp_min(1e-8)).item()
    print(f'{"ln_f":<8}{d.max().item():>12.4e}{d.mean().item():>12.4e}{rel_mean:>12.4e}{rel_max:>12.4e}')


if __name__ == '__main__':
    main()
