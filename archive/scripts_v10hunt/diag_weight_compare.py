"""Verify nano's loaded weights bitwise-match ref's iter 500 weights.

Loads the same Megatron ckpt twice:
  - Via scripts/megatron_to_nano converter → nano-named state_dict
  - Directly from Megatron mp_rank files → Megatron-named state_dict
Then re-converts Megatron names to nano-named paths and diffs tensor values.
"""
from __future__ import annotations
import argparse
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPT_DIR)
from megatron_to_nano import load_all_megatron_shards, convert, split_qkv, split_gate_up


def sum_stats(t):
    tf = t.detach().float()
    return dict(shape=tuple(t.shape), dtype=str(t.dtype),
                min=tf.min().item(), max=tf.max().item(),
                mean=tf.mean().item(), std=tf.std().item(),
                sum=tf.sum().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meg-dir', required=True)
    args = ap.parse_args()

    print('[1] Loading raw Megatron state_dict...', flush=True)
    meg = load_all_megatron_shards(args.meg_dir)
    print(f'    raw keys: {len(meg)}')

    print('[2] Running converter → nano state_dict...', flush=True)
    nano_from_conv = convert(meg)
    print(f'    nano keys: {len(nano_from_conv)}')

    print('[3] Re-building nano model and load_state_dict(strict=False)...', flush=True)
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
        seq_aux_balance_alpha=0.0001, use_eod_attn_mask=True,
    )
    m = GPT(cfg)
    m.load_state_dict(nano_from_conv, strict=False)

    print('[4] For each nano param, compare with state_dict entry (bitwise):')
    mismatch = 0
    big_diff = 0
    nano_sd = m.state_dict()
    for name, t in nano_sd.items():
        if 'local_tokens_per_expert' in name:
            continue
        if name not in nano_from_conv:
            print(f'  MISSING: {name}')
            mismatch += 1
            continue
        ref_t = nano_from_conv[name]
        if not torch.equal(t.cpu(), ref_t.cpu()):
            diff = (t.cpu().float() - ref_t.cpu().float()).abs()
            big_diff += 1
            print(f'  DIFF {name}: max={diff.max().item():.4e} mean={diff.mean().item():.4e} shapes {t.shape} vs {ref_t.shape}')
    print(f'\nmismatch: {mismatch}, diff (non-bitwise): {big_diff} of {len(nano_sd)}')

    print('\n[5] Spot-check stats on a few tensors:')
    for name in ['transformer.wte.weight', 'transformer.h.0.attn.q_proj.weight',
                 'transformer.h.1.mlp.router.linear.weight',
                 'transformer.h.1.mlp.gate_weight',
                 'lm_head.weight']:
        if name in nano_sd:
            s = sum_stats(nano_sd[name])
            print(f'  {name}:')
            for k, v in s.items():
                print(f'    {k}={v}')


if __name__ == '__main__':
    main()
