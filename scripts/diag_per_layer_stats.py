"""Per-layer forward+backward diagnostic at iter 0 → iter 1.

Hooks every attention / MLP module, records act_std / act_mean / act_max of their
inputs and outputs during a single forward pass on iter-1 batch (using ref iter_0
weights). Also reports grad norm after a single backward.

Compare output with ref's master.log at iter 1 to find the first layer that diverges.
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPT_DIR)


def stat(t):
    tf = t.detach().float()
    return dict(
        std=tf.std().item(),
        mean=tf.mean().item(),
        max=tf.abs().max().item(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meg-dir', required=True, help='path to iter_0000000 or iter_0000010 dir')
    ap.add_argument('--data', default=os.path.join(ROOT, 'data/cybertron_baseline/train.bin'))
    ap.add_argument('--batch-step', type=int, default=1,
                    help='Which training step batch to use (1-indexed; step N uses samples (N-1)*64..N*64-1)')
    args = ap.parse_args()

    # Load ref weights into nano
    from megatron_to_nano import load_all_megatron_shards, convert
    from model import GPTConfig, GPT
    meg = load_all_megatron_shards(args.meg_dir)
    sd = convert(meg)

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
    m = GPT(cfg).cuda()
    m.load_state_dict(sd, strict=False)

    # Register hooks on per-layer points of interest
    records = []

    def hook_capture(name):
        def h(mod, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            x = inp[0] if isinstance(inp, tuple) else inp
            records.append((name, 'input', stat(x)))
            if isinstance(out, torch.Tensor):
                records.append((name, 'output', stat(out)))
        return h

    for i, blk in enumerate(m.transformer.h):
        blk.ln_1.register_forward_hook(hook_capture(f'L{i}.ln_1'))
        blk.attn.register_forward_hook(hook_capture(f'L{i}.attn'))
        blk.ln_2.register_forward_hook(hook_capture(f'L{i}.ln_2'))
        blk.mlp.register_forward_hook(hook_capture(f'L{i}.mlp'))
    m.transformer.ln_f.register_forward_hook(hook_capture('final_ln'))

    # Load iter-N batch
    arr = np.memmap(args.data, dtype=np.int32, mode='r')
    s = args.batch_step
    samples = [np.array(arr[(s-1)*64*8192 + i*8192 : (s-1)*64*8192 + i*8192 + 8193].astype(np.int64))
               for i in range(64)]

    print(f'[forward] iter-{s} batch, 64 samples, ref iter_{s} loss target from master.log\n')
    m.train()  # training mode for router load stats
    # bf16 autocast (match training) + single-sample micro to fit 80GB GPU
    idx = torch.from_numpy(samples[0][:8192]).unsqueeze(0).cuda()
    tgt = torch.from_numpy(samples[0][1:8193]).unsqueeze(0).cuda()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits, loss = m(idx, targets=tgt)
    print(f'loss on first 1 sample (bf16 autocast): {loss.item():.4f}\n')

    # Emit per-layer stats
    print(f'{"layer":<14}{"io":>7}{"std":>12}{"mean":>14}{"|max|":>12}')
    for name, io, s_ in records:
        print(f'{name:<14}{io:>7}{s_["std"]:>12.4e}{s_["mean"]:>14.4e}{s_["max"]:>12.4e}')

    # Backward single sample
    print('\n[backward] single-sample backward, compute grad norm')
    m.zero_grad()
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1e9).item()
    print(f'grad norm (unclipped): {gn:.4f}  (ref @ iter 1 = 2.033)')


if __name__ == '__main__':
    main()
