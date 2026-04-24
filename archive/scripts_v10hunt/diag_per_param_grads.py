"""Per-param gradient diagnostic at iter 1 from iter_0 weights.

Dumps: {param_name: {'shape': ..., 'grad_norm': ..., 'grad_mean': ..., 'grad_std': ...}}

Use for direct layer-by-layer comparison vs ref's iter_1 exp_avg (= 0.1 * grad_1).
Requires 8-GPU DDP to match ref's micro_bs=1 × grad_accum=8 per rank × 8 ranks grad
averaging. Runs on rank 0 only and saves the dump.
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
    ap.add_argument('--batch-step', type=int, default=1)
    ap.add_argument('--out', default=os.path.join(ROOT, 'reports/short_window/nano_grads_iter1.json'))
    args = ap.parse_args()

    # Init DDP
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
    device = torch.device('cuda', local_rank)

    # Load ref weights
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

    # Batch: match train.py's DDP-interleaved per-rank sample order.
    # At global step B, samples = [B*64 .. (B+1)*64). Per-rank r: samples r, r+W, r+2W, ... r+7W
    arr = np.memmap(args.data, dtype=np.int32, mode='r')
    start = (args.batch_step - 1) * 64
    samples = [start + rank + i * world for i in range(8)]  # 8 samples per rank (interleaved)
    if rank == 0:
        print(f'[rank0] samples indices (first 8): {samples[:8]}')

    raw = m.module if hasattr(m, 'module') else m
    raw.train()
    ga = 8
    total_loss = 0.0
    for i, sid in enumerate(samples):
        idx = torch.from_numpy(np.array(arr[sid*8192 : sid*8192 + 8192].astype(np.int64))).unsqueeze(0).to(device)
        tgt = torch.from_numpy(np.array(arr[sid*8192 + 1 : sid*8192 + 8193].astype(np.int64))).unsqueeze(0).to(device)
        if world > 1:
            m.require_backward_grad_sync = (i == ga - 1)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _, loss = m(idx, targets=tgt)
            total_loss += loss.item()
            (loss / ga).backward()

    # Trigger aux-free bias update (matches training pipeline)
    for mod in raw.modules():
        if isinstance(mod, MoERouter):
            mod.update_expert_bias()

    # Per-param grad stats (rank 0 only)
    if rank == 0:
        stats = {}
        total_sq = 0.0
        for name, p in raw.named_parameters():
            g = p.grad
            if g is None:
                continue
            gf = g.float()
            s = gf.std().item()
            me = gf.mean().item()
            mx = gf.abs().max().item()
            nrm = gf.norm().item()
            stats[name] = {
                'shape': list(g.shape),
                'norm': nrm,
                'std': s,
                'mean': me,
                'abs_max': mx,
                'numel': g.numel(),
            }
            total_sq += nrm ** 2
        total_gn = total_sq ** 0.5
        print(f'[rank0] total grad norm (pre-clip, across ranks via DDP sync): {total_gn:.6f}')
        print(f'[rank0] loss avg over {ga} samples: {total_loss/ga:.6f}')
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'total_grad_norm': total_gn,
                       'loss': total_loss / ga,
                       'params': stats}, f, indent=2)
        print(f'[rank0] wrote {args.out}')

    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
