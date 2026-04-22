"""Recover per-param {step, exp_avg, exp_avg_sq} from Megatron distributed_optimizer ckpt.

Megatron stores optim state in distrib_optim.pt as FLAT tensors per bucket — concatenation
of all params in that bucket. The order is determined by param_and_grad_buffer.py: it
ITERATES named_parameters in REVERSE order and appends each to a bucket, keyed by
(dtype, grad_dtype). So to unflatten, walk model.named_parameters() in reverse and slice
out each param's numel.

Buckets split by is_expert_parallel:
  bucket 0: non-expert params (DP-replicated, only rank 0 has it populated)
  bucket 1: expert params (sharded across 4 EP ranks)

This script reads all 4 shards and reconstructs per-param optim state in Megatron key space.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import torch

CYBERTRON = '/newcpfs/user/yuchen/llm/cybertron_dots3.0_swa'
MEGATRON  = '/newcpfs/user/yuchen/llm/megatron_dots3.0_swa'
sys.path.insert(0, CYBERTRON)
sys.path.insert(0, MEGATRON)


def slice_bucket(flat: torch.Tensor, names_sizes: list) -> dict:
    """Given a flat tensor and a list of (name, numel), return {name: slice_tensor}.

    Slices from END to START because Megatron reserves the bucket end for the FIRST
    added param (reverse order).
    """
    out = {}
    end = flat.numel()
    for name, n in names_sizes:
        out[name] = flat[end - n : end].clone()
        end -= n
    assert end == 0, f'bucket flat size mismatch: {end} left over'
    return out


def split_expert_param_name(name: str):
    """experts.linear_fc{12}.weight{I} → (prefix, global_i) for EP sharding."""
    import re
    m = re.match(r'(.+experts\.linear_fc[12])\.weight(\d+)$', name)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', required=True, help='Path to iter_XXXXXXX directory')
    ap.add_argument('--out', required=True, help='Output path for the recovered dict (.pt)')
    args = ap.parse_args()

    # 1. Load all 4 shards' model state_dict (non-expert duplicated, experts sharded)
    all_models = []
    all_distrib = []
    for r in range(4):
        rdir = f'{args.ckpt_dir}/mp_rank_00_{r:03d}'
        mck = torch.load(f'{rdir}/model_optim_rng.pt', map_location='cpu', weights_only=False)
        dck = torch.load(f'{rdir}/distrib_optim.pt', map_location='cpu', weights_only=False)
        sd = mck['model']
        if hasattr(sd, 'state_dict'):
            sd = sd.state_dict()
        all_models.append(sd)
        all_distrib.append(dck)

    # 2. Build global ordered list of (name, tensor) in REVERSE (Megatron's add order)
    #    Iterate the "canonical" (rank-0) model first to get key ordering.
    rank0_sd = all_models[0]
    # Only fp-trainable tensors go through optimizer. Extra state is skipped.
    # e_score_correction_bias is updated manually, not via Adam → exclude from optim state.
    names_ordered = [k for k, v in rank0_sd.items()
                     if isinstance(v, torch.Tensor) and not k.endswith('_extra_state')
                     and v.dtype in (torch.bfloat16, torch.float16, torch.float32)
                     and 'e_score_correction_bias' not in k]

    # Split non-expert vs expert by name
    def is_expert_param(name: str):
        return '.experts.linear_fc' in name and name.endswith('_extra_state') is False and 'weight' in name.split('.')[-1]

    non_expert_names = [n for n in names_ordered if not is_expert_param(n)]
    expert_names_local = [n for n in names_ordered if is_expert_param(n)]  # rank-0 local indices

    # Reverse each — Megatron's ParamAndGradBuffer iterates params in forward order
    # and appends to bucket. But since buckets are contiguous, the first-added
    # param occupies the START of bucket (not end). Recent Megatron uses "reverse"
    # add order to match backward overlap — easiest to just TRY both orders and
    # pick the one whose fp32 slices match bf16 model weights.

    def try_unflatten(names_in_order, flat_param, flat_exp_avg, flat_exp_avg_sq,
                      sd_for_check, rank_for_exp=None):
        """Try to unflatten using given order; verify param slice matches sd."""
        sizes = [(n, sd_for_check[n].numel()) for n in names_in_order]
        total = sum(s for _, s in sizes)
        if total != flat_param.numel():
            return None, f'total mismatch: expect {total}, got {flat_param.numel()}'
        out = {}
        offset = 0
        # Try START-TO-END order first
        for n, sz in sizes:
            p_slice = flat_param[offset:offset + sz]
            ref_val = sd_for_check[n].flatten().float()
            # Allow small fp32↔bf16 diff
            if not torch.allclose(p_slice, ref_val, atol=1e-2, rtol=1e-2):
                return None, f'param {n} mismatch at offset {offset} (abs max diff {(p_slice-ref_val).abs().max().item():.4e})'
            out[n] = {
                'exp_avg': flat_exp_avg[offset:offset + sz].view_as(sd_for_check[n]).clone(),
                'exp_avg_sq': flat_exp_avg_sq[offset:offset + sz].view_as(sd_for_check[n]).clone(),
            }
            offset += sz
        return out, None

    # Non-expert bucket (bucket 0, rank 0 only)
    bucket0 = all_distrib[0][0][0][(torch.bfloat16, torch.float32)]
    print(f'bucket0: {bucket0["param"].numel()} fp32 elements')
    print(f'  trying FORWARD order ({len(non_expert_names)} non-expert params, '
          f'total {sum(rank0_sd[n].numel() for n in non_expert_names)} elements)')
    result_ne, err = try_unflatten(non_expert_names, bucket0['param'],
                                    bucket0['exp_avg'], bucket0['exp_avg_sq'], rank0_sd)
    if result_ne is None:
        print(f'  FWD failed: {err}')
        print(f'  trying REVERSE order')
        result_ne, err = try_unflatten(list(reversed(non_expert_names)),
                                        bucket0['param'], bucket0['exp_avg'],
                                        bucket0['exp_avg_sq'], rank0_sd)
        if result_ne is None:
            print(f'  REVERSE also failed: {err}')
            sys.exit(2)
        else:
            print(f'  reverse order works: {len(result_ne)} params recovered')
    else:
        print(f'  forward order works: {len(result_ne)} params recovered')

    # Expert bucket (bucket 1, each rank has its own shard of 36 experts)
    # rank r has experts [r*36 .. r*36+35]; param names use LOCAL indices weight0..weight35
    # We want to recover per-global-expert optim state.
    result_exp = {}
    for r in range(4):
        bucket1 = all_distrib[r][1][0][(torch.bfloat16, torch.float32)]
        sd_r = all_models[r]
        exp_names_r = [n for n in sd_r if is_expert_param(n) and isinstance(sd_r[n], torch.Tensor)]
        print(f'rank{r} bucket1: {bucket1["param"].numel()} fp32 elements, '
              f'{len(exp_names_r)} local-expert params, total {sum(sd_r[n].numel() for n in exp_names_r)} elements')
        rec, err = try_unflatten(exp_names_r, bucket1['param'], bucket1['exp_avg'],
                                  bucket1['exp_avg_sq'], sd_r)
        if rec is None:
            print(f'  FWD failed: {err}; trying REVERSE')
            rec, err = try_unflatten(list(reversed(exp_names_r)), bucket1['param'],
                                      bucket1['exp_avg'], bucket1['exp_avg_sq'], sd_r)
            if rec is None:
                print(f'  REVERSE failed: {err}')
                sys.exit(3)
        # Remap local expert indices to global
        for local_name, st in rec.items():
            import re
            m = re.match(r'(.+experts\.linear_fc[12])\.weight(\d+)$', local_name)
            if m:
                prefix, local_i = m.group(1), int(m.group(2))
                gi = r * 36 + local_i
                result_exp[f'{prefix}.weight{gi}'] = st
            else:
                # Non-expert param in bucket1? Shouldn't happen
                result_exp[local_name] = st

    # Merge
    combined = {}
    combined.update(result_ne)
    combined.update(result_exp)
    print(f'\nRecovered optim state for {len(combined)} Megatron params')

    # Also carry the step counter
    step = all_models[0]['args'].iteration if hasattr(all_models[0].get('args', {}), 'iteration') else None
    if step is None:
        # Pull from param_groups (mega optim state)
        ck0 = torch.load(f'{args.ckpt_dir}/mp_rank_00_000/model_optim_rng.pt',
                         map_location='cpu', weights_only=False)
        pgs = ck0['optimizer'][0]['optimizer']['param_groups']
        step = pgs[0]['step']
    print(f'step: {step}')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({'state': combined, 'step': step}, args.out)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
