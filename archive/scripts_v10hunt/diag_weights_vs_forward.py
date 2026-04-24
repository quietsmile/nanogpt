"""Decompose mid-training Δ into "forward kernel diff" vs "trained weights diff".

Test A: forward kernel diff
  - Load ref iter 5988 weights into nano
  - Forward on iter-5989 batch → loss_A
  - Compare with ref's logged single-iter loss at iter 5989 = loss_ref
  - |loss_A − loss_ref| = pure kernel difference (same weights, diff kernel)

Test B: trained weights diff
  - Load nano iter 7000 weights → forward on fixed batch K → loss_nano
  - Load ref iter 7485 weights → forward on same batch K → loss_ref_trained
  - |loss_nano − loss_ref_trained| = effect of different training trajectories
    (note: 485 step training gap, not a clean comparison)
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.distributed as dist

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPT_DIR)


def build_nano(attention_impl='sdpa'):
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
        attention_impl=attention_impl,
    )
    return GPT(cfg), cfg


def forward_on_batch(model, sample_indices, data_mmap, dtype=torch.bfloat16):
    """DDP-style forward: rank r handles samples[r::world]. Returns local loss average."""
    device = next(model.parameters()).device
    raw = model.module if hasattr(model, 'module') else model
    raw.eval()
    losses = []
    ctx = torch.amp.autocast('cuda', dtype=dtype) if dtype == torch.bfloat16 else torch.amp.autocast('cuda', enabled=False)
    with torch.no_grad(), ctx:
        for sid in sample_indices:
            idx = torch.from_numpy(np.array(data_mmap[sid*8192 : sid*8192+8192].astype(np.int64))).unsqueeze(0).to(device)
            tgt = torch.from_numpy(np.array(data_mmap[sid*8192+1 : sid*8192+8193].astype(np.int64))).unsqueeze(0).to(device)
            _, loss = model(idx, targets=tgt)
            losses.append(loss.item())
    return sum(losses) / len(losses) if losses else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['A', 'B', 'C'])
    ap.add_argument('--attention-impl', default='sdpa', choices=['sdpa', 'fp32_manual', 'te'])
    ap.add_argument('--fwd-dtype', default='bf16', choices=['bf16', 'fp32'])
    ap.add_argument('--data', default=os.path.join(ROOT, 'data/cybertron_baseline/train.bin'))
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if world > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
    device = torch.device('cuda', local_rank)

    from megatron_to_nano import load_all_megatron_shards, convert

    arr = np.memmap(args.data, dtype=np.int32, mode='r')

    if args.mode == 'A':
        # Test A: for each available ref ckpt at iter K, nano forward on iter (K+1) batch.
        # Compare to ref's logged loss at iter (K+1) which used the SAME weights.
        # This is the cleanest pure-forward-kernel-diff measurement.
        import re
        ref_losses = {}
        for line in open('/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/logs/rank-0-1-scaling_moe_00196-run.log'):
            m = re.search(r'iteration\s+(\d+)/.*lm loss:\s*([\d.eE+-]+)', line)
            if m:
                ref_losses[int(m.group(1))] = float(m.group(2))

        # (ckpt_iter, next_iter) pairs — each pair is weights and batch that ref forwarded
        pairs = [(1497, 1498), (2994, 2995), (4491, 4492), (5988, 5989)]

        if rank == 0:
            print(f'[Test A] 4 matched pairs: (ref ckpt iter N) + (batch iter N+1) → nano fwd vs ref logged')
            print(f'{"ckpt":>6}{"batch":>7}{"nano_fwd":>12}{"ref_logged":>12}{"Δ":>10}')

        results = []
        for ckpt_iter, next_iter in pairs:
            ref_dir = f'/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/iter_{ckpt_iter:07d}'
            meg = load_all_megatron_shards(ref_dir)
            sd = convert(meg)
            model, _ = build_nano(attention_impl=args.attention_impl)
            model = model.to(device)
            model.load_state_dict(sd, strict=False)
            all_samples = list(range((next_iter - 1) * 64, next_iter * 64))
            my_samples = [all_samples[i] for i in range(rank, 64, world)]
            local_loss = forward_on_batch(model, my_samples, arr,
                                          dtype=(torch.bfloat16 if args.fwd_dtype == 'bf16' else torch.float32))
            t = torch.tensor(local_loss, device=device)
            if world > 1:
                dist.all_reduce(t, op=dist.ReduceOp.AVG)
            ref_loss = ref_losses.get(next_iter)
            if rank == 0:
                delta = t.item() - ref_loss if ref_loss else None
                print(f'{ckpt_iter:>6}{next_iter:>7}{t.item():>12.6f}{ref_loss if ref_loss else 0:>12.6f}{delta if delta else 0:>+10.6f}')
                results.append((ckpt_iter, next_iter, t.item(), ref_loss, delta))
            del model
            torch.cuda.empty_cache()

        if rank == 0 and results:
            import statistics
            diffs = [r[4] for r in results if r[4] is not None]
            print(f'\n[Test A] avg Δ (nano_fwd - ref_fwd, SAME weights) = {statistics.mean(diffs):+.6f}')
            print(f'[Test A] stddev across 4 ckpts = {statistics.stdev(diffs):.6f}')
            print('\n→ This is pure forward-kernel diff (PyTorch SDPA vs TE flash-attn-2).')

    elif args.mode == 'C':
        # Test C: with ref 5988 weights, sweep batch iter {5987..5992} to check offset alignment
        import re
        ref_losses = {}
        for line in open('/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/logs/rank-0-1-scaling_moe_00196-run.log'):
            m = re.search(r'iteration\s+(\d+)/.*lm loss:\s*([\d.eE+-]+)', line)
            if m: ref_losses[int(m.group(1))] = float(m.group(2))
        ref_dir = '/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/iter_0005988'
        meg = load_all_megatron_shards(ref_dir)
        sd = convert(meg)
        model, _ = build_nano(attention_impl=args.attention_impl)
        model = model.to(device)
        model.load_state_dict(sd, strict=False)
        if rank == 0:
            print(f'[Test C] ref ckpt iter 5988 → sweep batch iters 5985..5992')
            print(f'{"batch":>7}{"nano_fwd":>12}{"ref_logged":>12}{"Δ":>10}')
        for bi in range(5985, 5993):
            all_samples = list(range((bi - 1) * 64, bi * 64))
            my_samples = [all_samples[i] for i in range(rank, 64, world)]
            local_loss = forward_on_batch(model, my_samples, arr,
                                          dtype=(torch.bfloat16 if args.fwd_dtype == 'bf16' else torch.float32))
            t = torch.tensor(local_loss, device=device)
            if world > 1:
                dist.all_reduce(t, op=dist.ReduceOp.AVG)
            if rank == 0:
                rl = ref_losses.get(bi, 0)
                print(f'{bi:>7}{t.item():>12.6f}{rl:>12.6f}{t.item()-rl:>+10.6f}')
    elif args.mode == 'B':
        # Test B: nano 7000 weights vs ref 7485 weights, same fixed batch
        batch_iter = 100  # fixed eval batch: samples 6400..6463
        all_samples = list(range((batch_iter - 1) * 64, batch_iter * 64))
        my_samples = [all_samples[i] for i in range(rank, 64, world)]

        # Forward with nano trained weights
        if rank == 0:
            print(f'[Test B] eval on iter-{batch_iter} batch (samples {all_samples[0]}..{all_samples[-1]})')
            print(f'  Loading nano iter 7000 ckpt...')
        nano_ckpt = torch.load('/root/nanogpt/out-cybertron-moe-196-from0/ckpt.pt',
                                map_location='cpu', weights_only=False)
        model, _ = build_nano(attention_impl=args.attention_impl)
        model = model.to(device)
        model.load_state_dict(nano_ckpt['model'], strict=False)
        loss_nano = forward_on_batch(model, my_samples, arr)
        t_nano = torch.tensor(loss_nano, device=device)
        if world > 1:
            dist.all_reduce(t_nano, op=dist.ReduceOp.AVG)

        # Forward with ref trained weights (iter 7485)
        if rank == 0:
            print(f'  Loading ref iter 7485 ckpt...')
        ref_dir = '/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/iter_0007485'
        meg = load_all_megatron_shards(ref_dir)
        sd_ref = convert(meg)
        model_ref, _ = build_nano(attention_impl=args.attention_impl)
        model_ref = model_ref.to(device)
        model_ref.load_state_dict(sd_ref, strict=False)
        loss_ref = forward_on_batch(model_ref, my_samples, arr)
        t_ref = torch.tensor(loss_ref, device=device)
        if world > 1:
            dist.all_reduce(t_ref, op=dist.ReduceOp.AVG)

        if rank == 0:
            print(f'\n[Test B RESULT]')
            print(f'  nano iter-7000 weights, nano forward: {t_nano.item():.6f}')
            print(f'  ref  iter-7485 weights, nano forward: {t_ref.item():.6f}')
            print(f'  Δ (nano_weights - ref_weights) = {t_nano.item() - t_ref.item():+.6f}')
            print(f'  (both forwarded by nano kernel; weight diff = 485 extra ref training steps)')

    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
