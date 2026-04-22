"""Pragmatic test: halve nano's gradient (via loss = loss / (2 * ga)) and check
if total grad_norm matches ref's ~1.0 per-param derivation.

If so, we've confirmed there's a 2× scaling mismatch somewhere (probably in
how we divide by grad_accum vs ref). Bisect the source from there.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.distributed as dist

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPT_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meg-dir', required=True)
    ap.add_argument('--data', default=os.path.join(ROOT, 'data/cybertron_baseline/train.bin'))
    ap.add_argument('--loss-divisor', type=float, default=1.0,
                    help='Extra divisor applied to per-micro loss (1 = no change, 2 = halve grad)')
    args = ap.parse_args()

    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
    device = torch.device('cuda', local_rank)

    from megatron_to_nano import load_all_megatron_shards, convert
    from model import GPTConfig, GPT, MoERouter
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
    m = GPT(cfg).to(device)
    m.load_state_dict(sd, strict=False)
    if world > 1:
        m = torch.nn.parallel.DistributedDataParallel(m, device_ids=[local_rank])
    raw = m.module if hasattr(m, 'module') else m
    raw.train()

    arr = np.memmap(args.data, dtype=np.int32, mode='r')
    samples = [rank + i * world for i in range(8)]
    ga = 8
    total_loss = 0.0

    if rank == 0:
        print(f'[rank0] loss_divisor={args.loss_divisor}, samples={samples}')

    for i, sid in enumerate(samples):
        idx = torch.from_numpy(np.array(arr[sid*8192 : sid*8192+8192].astype(np.int64))).unsqueeze(0).to(device)
        tgt = torch.from_numpy(np.array(arr[sid*8192+1 : sid*8192+8193].astype(np.int64))).unsqueeze(0).to(device)
        if world > 1:
            m.require_backward_grad_sync = (i == ga - 1)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _, loss = m(idx, targets=tgt)
            total_loss += loss.item()
            (loss / (ga * args.loss_divisor)).backward()

    for mod in raw.modules():
        if isinstance(mod, MoERouter):
            mod.update_expert_bias()

    if rank == 0:
        total_sq = 0.0
        for name, p in raw.named_parameters():
            if p.grad is None: continue
            total_sq += p.grad.float().norm().item() ** 2
        total_gn = total_sq ** 0.5
        print(f'[rank0] total grad norm: {total_gn:.6f}')
        print(f'[rank0] avg loss: {total_loss/ga:.6f}')
        print(f'[rank0] implied grad (if this halving makes it match ref \'s 1.0): {total_gn:.6f}')

    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
