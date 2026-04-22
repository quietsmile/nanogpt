"""Diagnose forward-only Δ at iter 500: is 0.06 nat bf16 drift or a real arch bug?

- Load iter-500 weights (from ref)
- Run forward on iter-501 batch in 3 precisions and compare vs ref iter 501 = 4.7368:
  (a) bf16 autocast (current default)
  (b) fp32 (full model + autocast disabled)
  (c) per-layer activation dump to find which layer accumulates the diff
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


def load_nano_from_meg(meg_dir: str):
    from megatron_to_nano import load_all_megatron_shards, convert
    from model import GPTConfig, GPT
    meg = load_all_megatron_shards(meg_dir)
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
        seq_aux_balance_alpha=0.0001, use_eod_attn_mask=True,
    )
    m = GPT(cfg)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if 'local_tokens_per_expert' not in k]
    assert not real_missing, f'missing {real_missing[:5]}'
    return m.cuda()


def load_batch(step, data_path, block_size=8192, gbs=64):
    arr = np.memmap(data_path, dtype=np.int32, mode='r')
    start = (step - 1) * gbs
    return [np.array(arr[(start+s)*block_size : (start+s)*block_size + block_size + 1].astype(np.int64))
            for s in range(gbs)]


def eval_loss(model, samples, dtype=None, micro=2):
    model.eval()
    losses = []
    with torch.no_grad():
        ctx = torch.amp.autocast('cuda', dtype=dtype) if dtype is not None else torch.cuda.amp.autocast(enabled=False)
        with ctx:
            for c0 in range(0, len(samples), micro):
                idx = torch.stack([torch.from_numpy(samples[c0+i][:8192]) for i in range(micro)], 0).cuda()
                tgt = torch.stack([torch.from_numpy(samples[c0+i][1:8193]) for i in range(micro)], 0).cuda()
                _, loss = model(idx, targets=tgt)
                losses.append(loss.item())
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meg-dir', required=True)
    ap.add_argument('--data',    default=os.path.join(ROOT, 'data/cybertron_baseline/train.bin'))
    ap.add_argument('--step',    type=int, default=501)
    ap.add_argument('--ref-loss', type=float, default=4.736839,
                    help='reference loss from master.log at this iter')
    args = ap.parse_args()

    print(f'[1/3] Loading nano model with ref iter-500 weights ...', flush=True)
    m = load_nano_from_meg(args.meg_dir)
    print(f'      model on cuda, dtype={next(m.parameters()).dtype}')

    samples = load_batch(args.step, args.data)
    print(f'[2/3] Loaded batch {args.step}: {len(samples)} samples, first seq first 8 tokens: {samples[0][:8].tolist()}')

    print(f'\n[3/3] Forward-only loss comparisons (ref iter {args.step} loss = {args.ref_loss:.6f})')
    # (a) bf16 autocast (match training config)
    l_bf16 = eval_loss(m, samples, dtype=torch.bfloat16)
    print(f'  bf16 autocast: {l_bf16:.6f}  Δ={l_bf16-args.ref_loss:+.6f}')

    # (b) fp32 (autocast disabled, model fp32)
    l_fp32 = eval_loss(m, samples, dtype=None)
    print(f'  fp32:           {l_fp32:.6f}  Δ={l_fp32-args.ref_loss:+.6f}')

    # (c) cast model to bf16 and forward without autocast (pure bf16 path)
    m_bf16 = m.to(torch.bfloat16)
    l_bf16_pure = eval_loss(m_bf16, samples, dtype=None)
    print(f'  bf16 native:    {l_bf16_pure:.6f}  Δ={l_bf16_pure-args.ref_loss:+.6f}')

    # (d) toggle EOD attn mask off to test if the mask itself matters
    m.to(torch.float32)
    print('\n--- toggling use_eod_attn_mask off (simple causal) ---')
    m.config.use_eod_attn_mask = False
    l_fp32_no_eod = eval_loss(m, samples, dtype=None)
    print(f'  fp32 (no EOD):  {l_fp32_no_eod:.6f}  Δ={l_fp32_no_eod-args.ref_loss:+.6f}')

    # (e) restore EOD, test without seq_aux and without loss masks
    m.config.use_eod_attn_mask = True
    m.config.seq_aux_balance_alpha = 0.0
    m.config.mask_loss_id = None
    m.config.eod_token_id = None
    l_fp32_no_masks = eval_loss(m, samples, dtype=None)
    print(f'  fp32 (no masks, with EOD attn): {l_fp32_no_masks:.6f}  Δ={l_fp32_no_masks-args.ref_loss:+.6f}')

    # (f) Force the MANUAL (non-flash) attention path, keeping EOD on
    m.config.use_eod_attn_mask = True
    m.config.eod_token_id = 151643
    m.config.mask_loss_id = 160000
    m.config.seq_aux_balance_alpha = 0.0001
    for blk in m.transformer.h:
        blk.attn.flash = False
    l_fp32_manual = eval_loss(m, samples, dtype=None)
    print(f'  fp32 (manual attn + EOD): {l_fp32_manual:.6f}  Δ={l_fp32_manual-args.ref_loss:+.6f}')

    # (g) Isolate: EOD attn mask ON but loss-side EOD masking OFF
    for blk in m.transformer.h:
        blk.attn.flash = True
    m.config.use_eod_attn_mask = True
    m.config.eod_token_id = 151643  # needed for attn mask
    m.config.mask_loss_id = None    # disable loss-side masking
    # To keep EOD mask active in attn but NOT mask loss, we hack: skip the loss mask
    # path in model.py. Easier: set eod to an impossible token for LOSS masking only.
    # Actually the loss masks in model.py use self.config.eod_token_id directly — we
    # can't separate them without code change. So we CAN'T easily isolate loss-mask.
    # Instead, measure: EOD attn ON but seq_aux=0 (to rule out seq_aux contribution)
    m.config.seq_aux_balance_alpha = 0.0
    l_fp32_no_seqaux = eval_loss(m, samples, dtype=None)
    print(f'  fp32 (EOD ON, seq_aux=0): {l_fp32_no_seqaux:.6f}  Δ={l_fp32_no_seqaux-args.ref_loss:+.6f}')


if __name__ == '__main__':
    main()
