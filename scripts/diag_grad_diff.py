"""Compare nano's per-param grad (from diag_per_param_grads.py JSON dump) against
ref's per-param grad derived from iter_1 exp_avg.

Since at iter 0 exp_avg = 0 and Adam β1 = 0.9, exp_avg_1 = (1 - β1) * grad_1 = 0.1 * grad_1.
So ref grad_1 = exp_avg_1 / 0.1 = 10 * exp_avg_1.

For each Megatron param, also apply nano's layout mapping (via optim_megatron_to_nano)
so names match between the two dumps.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPT_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nano-dump', required=True, help='output of diag_per_param_grads.py')
    ap.add_argument('--ref-optim', required=True, help='meg_optim_iter1.pt')
    ap.add_argument('--out', default=os.path.join(ROOT, 'reports/short_window/grad_diff.json'))
    args = ap.parse_args()

    nano = json.load(open(args.nano_dump))
    ref = torch.load(args.ref_optim, map_location='cpu', weights_only=False)

    from optim_megatron_to_nano import remap_optim
    ref_nano = remap_optim(ref['state'], ref['step'])  # {nano_name: {exp_avg, exp_avg_sq, step}}

    print(f'Total nano params with grad: {len(nano["params"])}')
    print(f'Total ref params after remap: {len(ref_nano)}')

    # Per-param comparison
    rows = []
    missing = []
    beta1 = 0.9
    total_nano_sq = 0.0
    total_ref_sq = 0.0
    total_diff_sq = 0.0
    for name, st in nano['params'].items():
        if name not in ref_nano:
            missing.append(name)
            continue
        # ref grad_1 = exp_avg_1 / (1 - β1) = exp_avg / 0.1 = 10 * exp_avg
        ref_ea = ref_nano[name]['exp_avg'].float()
        ref_grad_norm = (ref_ea.norm() / (1 - beta1)).item()
        ref_std = (ref_ea.std() / (1 - beta1)).item()
        nano_gn = st['norm']
        nano_std = st['std']
        rows.append({
            'name': name, 'shape': st['shape'],
            'nano_gn': nano_gn, 'ref_gn': ref_grad_norm,
            'gn_ratio': nano_gn / ref_grad_norm if ref_grad_norm > 0 else None,
            'nano_std': nano_std, 'ref_std': ref_std,
            'numel': st['numel'],
        })
        total_nano_sq += nano_gn ** 2
        total_ref_sq += ref_grad_norm ** 2

    print(f'\ntotal nano grad norm: {total_nano_sq**0.5:.4f}')
    print(f'total ref  grad norm: {total_ref_sq**0.5:.4f}')
    print(f'ratio: {(total_nano_sq**0.5)/(total_ref_sq**0.5):.4f}')
    if missing:
        print(f'MISSING in ref: {missing[:5]}...({len(missing)})')

    # Top 20 by nano grad norm, showing ratio
    rows.sort(key=lambda r: -r['nano_gn'])
    print(f'\n{"name":<55}{"shape":>20}{"nano_gn":>10}{"ref_gn":>10}{"n/r":>7}')
    for r in rows[:25]:
        print(f'{r["name"]:<55}{str(r["shape"]):>20}{r["nano_gn"]:>10.4f}{r["ref_gn"]:>10.4f}{r["gn_ratio"]:>7.3f}')

    # Top 10 by RATIO deviation (biggest outliers)
    for r in rows:
        r['ratio_dev'] = abs(r['gn_ratio'] - 1.0) if r['gn_ratio'] else 0
    rows.sort(key=lambda r: -r['ratio_dev'])
    print(f'\n--- Top 25 outliers by |ratio - 1| ---')
    print(f'{"name":<55}{"shape":>20}{"nano_gn":>10}{"ref_gn":>10}{"n/r":>7}')
    for r in rows[:25]:
        print(f'{r["name"]:<55}{str(r["shape"]):>20}{r["nano_gn"]:>10.4e}{r["ref_gn"]:>10.4e}{r["gn_ratio"]:>7.3f}')

    with open(args.out, 'w') as f:
        json.dump({'rows': rows, 'total_nano': total_nano_sq**0.5, 'total_ref': total_ref_sq**0.5}, f, indent=2)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
