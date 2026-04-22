"""Short-window alignment test: load iter 500 Megatron ckpt, train 10 steps, compare.

Run this on a GPU box after PAI job dlc1i2muiyvsehyh produces iter_0000500 + iter_0000510.
Generates:
  reports/short_window/nano_losses.json  -- nano's per-step losses 501..510
  reports/short_window/compare.json      -- diffs vs ref (pulled from master.log)
  reports/short_window/weight_diff.json  -- iter-510 weight diffs (nano vs ref)

Usage (on GPU box):
  python3 scripts/short_window_test.py \\
      --ref-dir /prodcpfs/user/yuchen/scaling_exp/auto_test_short/scaling_moe_00196_short510 \\
      --out reports/short_window
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import subprocess

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)


def convert_megatron_to_nano(megatron_iter_dir: str) -> dict:
    """Use scripts/megatron_to_nano.py's converter to load + convert."""
    sys.path.insert(0, SCRIPT_DIR)
    from megatron_to_nano import load_all_megatron_shards, convert
    meg = load_all_megatron_shards(megatron_iter_dir)
    return convert(meg)


def build_nano_model():
    from model import GPTConfig, GPT
    cfg = GPTConfig(
        block_size=8192, vocab_size=152064, n_layer=9, n_head=4, n_embd=512,
        n_kv_head=2, kv_channels=64,
        dropout=0.0, bias=False, init_std=0.006,
        use_rope=True, rotary_base=50000,
        use_rmsnorm=True, norm_eps=1e-5,
        use_swiglu=True, ffn_hidden_size=1536,
        qk_layernorm=True, tie_embeddings=False, disable_scaled_init_method=True,
        use_moe=True, moe_layer_freq=[0] + [1] * 8, num_experts=144,
        moe_ffn_hidden_size=160, moe_router_topk=8, moe_n_group=8, moe_topk_group=1,
        moe_norm_topk_prob=True, moe_router_score_correction_coeff=0.001,
        moe_shared_expert_hidden_size=160,
        moe_routing_type='greedy',
        eod_token_id=151643, mask_loss_id=160000,
        seq_aux_balance_alpha=0.0001,
        use_eod_attn_mask=False,  # ref uses regular per-sample causal (accurate_attn_mask_with_cp=False)
    )
    return GPT(cfg), cfg


def load_batch(step: int, data_path: str, block_size: int = 8192, gbs: int = 64):
    """Load GBS samples for the given training step (1-indexed)."""
    arr = np.memmap(data_path, dtype=np.int32, mode='r')
    start = (step - 1) * gbs
    batches = []
    for s in range(gbs):
        idx = (start + s) * block_size
        sample = np.array(arr[idx : idx + block_size + 1].astype(np.int64))
        batches.append(sample)
    return batches


def parse_ref_losses(log_path: str, iters: range) -> dict:
    """Pull lm loss values from ref master log."""
    out = {}
    with open(log_path) as f:
        for line in f:
            m = re.search(r'iteration\s+(\d+)/.*lm loss:\s*([\d.eE+-]+)', line)
            if m:
                it = int(m.group(1))
                if it in iters:
                    out[it] = float(m.group(2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref-dir', required=True,
                    help='Path to scaling_moe_00196_short510/ (contains iter_0000500, iter_0000510, logs/)')
    ap.add_argument('--out', default=os.path.join(ROOT, 'reports/short_window'))
    ap.add_argument('--data', default=os.path.join(ROOT, 'data/cybertron_baseline/train.bin'))
    ap.add_argument('--n-steps', type=int, default=10, help='Number of steps to train (501..500+N)')
    ap.add_argument('--start-iter', type=int, default=500)
    ap.add_argument('--load-optim', default=None,
                    help='Path to nano_optim_iter{N}.pt produced by optim_megatron_to_nano.py')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    iter500_dir = os.path.join(args.ref_dir, f'iter_{args.start_iter:07d}')
    iter510_dir = os.path.join(args.ref_dir, f'iter_{args.start_iter + args.n_steps:07d}')
    ref_log = os.path.join(args.ref_dir, 'logs',
                           f'rank-0-8-{os.path.basename(args.ref_dir)}-run.log')

    assert os.path.isdir(iter500_dir), f'missing {iter500_dir}'
    assert os.path.isdir(iter510_dir), f'missing {iter510_dir}'
    assert os.path.isfile(ref_log), f'missing {ref_log}'

    # 1. Convert iter-500 ref weights → nano state_dict
    print(f'[1/5] Converting {iter500_dir} → nano...', flush=True)
    sd_500 = convert_megatron_to_nano(iter500_dir)
    print(f'      {len(sd_500)} keys')

    # 2. Build nano model, load weights, verify shape
    print('[2/5] Building nano model + loading weights', flush=True)
    model, cfg = build_nano_model()
    model = model.cuda()
    missing, unexpected = model.load_state_dict(sd_500, strict=False)
    # local_tokens_per_expert is a nano-only buffer, init'd to zero in MoERouter constructor.
    # It's fine to be missing from a Megatron ckpt.
    missing_real = [k for k in missing if 'local_tokens_per_expert' not in k]
    assert not missing_real, f'missing keys: {missing_real[:5]}'
    print(f'      loaded; unexpected: {len(unexpected)}; missing (router buffer init=0): {len(missing) - len(missing_real)}')

    # 3. Parse ref losses for iter 501..510
    iters = range(args.start_iter + 1, args.start_iter + args.n_steps + 1)
    ref_losses = parse_ref_losses(ref_log, iters)
    print(f'[3/5] Ref losses iter {args.start_iter + 1}..{args.start_iter + args.n_steps}:')
    for it in iters:
        print(f'      iter {it}: {ref_losses.get(it, "NA"):.4f}' if it in ref_losses else f'      iter {it}: NA')

    # 4. Run nano forward at iter 500 (using iter-501 batch to check match)
    # Note: fresh optimizer state. This is a BOUND on alignment — if forward diverges,
    # we have issues. If forward matches, weight update trajectory differences come
    # from optim state mismatch (which we'll have to handle separately).
    print('[4/5] Nano forward-only on iter-501 batch (should match ref iter 500 eval if that exists)', flush=True)
    samples_501 = load_batch(args.start_iter + 1, args.data)
    model.eval()
    losses_fwd = []
    with torch.no_grad():
        for s in range(0, len(samples_501), 2):
            idx = torch.stack([torch.from_numpy(samples_501[s + i][:8192]) for i in range(2)], 0).cuda()
            tgt = torch.stack([torch.from_numpy(samples_501[s + i][1:8193]) for i in range(2)], 0).cuda()
            _, loss = model(idx, targets=tgt)
            losses_fwd.append(loss.item())
    nano_fwd_501 = sum(losses_fwd) / len(losses_fwd)
    print(f'      nano forward loss on iter-501 batch: {nano_fwd_501:.6f}')
    if (args.start_iter + 1) in ref_losses:
        d = nano_fwd_501 - ref_losses[args.start_iter + 1]
        print(f'      ref iter {args.start_iter + 1} loss: {ref_losses[args.start_iter + 1]:.6f}  Δ={d:+.6f}')

    # 5. Train 10 steps, log each step loss
    print(f'[5/5] Training {args.n_steps} steps from iter {args.start_iter}'
          f'{" (with loaded optim state)" if args.load_optim else " (fresh AdamW)"}', flush=True)
    from model import MoERouter
    model.train()
    # Match ref AdamW param-group split: dim>=2 params get wd=0.1, dim<2 get wd=0.0.
    opt = model.configure_optimizers(weight_decay=0.1, learning_rate=1.2e-3,
                                     betas=(0.9, 0.95), eps=1e-15, device_type='cuda')
    # Inject pre-trained optim state if provided
    if args.load_optim:
        payload = torch.load(args.load_optim, map_location='cpu', weights_only=False)
        nano_optim = payload['state']
        print(f'      loading {len(nano_optim)} param-optim states (step={payload["step"]})')
        # Build name→param map
        name_to_param = dict(model.named_parameters())
        injected = 0
        for name, state in nano_optim.items():
            if name not in name_to_param:
                print(f'      WARN: param {name} not in model, skipping')
                continue
            p = name_to_param[name]
            opt.state[p] = {
                'step': torch.tensor(float(state['step']), device=p.device),
                'exp_avg': state['exp_avg'].to(p.device, p.dtype),
                'exp_avg_sq': state['exp_avg_sq'].to(p.device, p.dtype),
            }
            injected += 1
        print(f'      injected state for {injected}/{len(name_to_param)} params')

    def step_lr(step_iter):
        # WSD-exp matching ref: lr = min(step_iter / warmup_iters, 1.0) * peak
        # (ref uses iter/warmup_iters, NOT (iter+1)/warmup)
        warmup_iters = 500
        peak = 1.2e-3
        return peak * min(step_iter / warmup_iters, 1.0)

    nano_losses = {}
    grad_norms = {}
    for step_idx in range(args.n_steps):
        it = args.start_iter + 1 + step_idx
        for pg in opt.param_groups:
            pg['lr'] = step_lr(it)
        opt.zero_grad(set_to_none=True)
        samples = load_batch(it, args.data)
        total_loss = 0.0
        ga = len(samples)
        for s in range(ga):
            idx = torch.from_numpy(samples[s][:8192]).unsqueeze(0).cuda()
            tgt = torch.from_numpy(samples[s][1:8193]).unsqueeze(0).cuda()
            _, loss = model(idx, targets=tgt)
            total_loss += loss.item()
            (loss / ga).backward()
            del idx, tgt, loss
            torch.cuda.empty_cache()
        avg_loss = total_loss / ga

        # Aux-free bias update
        for m in model.modules():
            if isinstance(m, MoERouter):
                m.update_expert_bias()

        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
        opt.step()
        nano_losses[it] = avg_loss
        grad_norms[it] = gn
        ref = ref_losses.get(it, None)
        if ref is not None:
            print(f'      iter {it}: nano={avg_loss:.4f}  ref={ref:.4f}  Δ={avg_loss - ref:+.4f}  gn={gn:.3f}', flush=True)
        else:
            print(f'      iter {it}: nano={avg_loss:.4f}  gn={gn:.3f}', flush=True)

    # Write report
    rep = {
        'ref_dir': args.ref_dir, 'start_iter': args.start_iter, 'n_steps': args.n_steps,
        'nano_forward_only_iter501': nano_fwd_501,
        'ref_iter501_loss': ref_losses.get(args.start_iter + 1),
        'nano_losses': nano_losses, 'ref_losses': ref_losses, 'grad_norms': grad_norms,
        'diffs': {it: (nano_losses[it] - ref_losses[it]) for it in nano_losses if it in ref_losses},
    }
    with open(os.path.join(args.out, 'compare.json'), 'w') as f:
        json.dump(rep, f, indent=2)
    print(f'\nWrote {args.out}/compare.json')


if __name__ == '__main__':
    main()
