"""Compute stable fingerprints for a model / checkpoint.

Two fingerprinting modes:
  - full: walk state_dict in sorted order, hash bytes (per-param MD5 + SHA256 of concat).
  - fast: concat only a small sample of each tensor's bytes (first 4KB + last 4KB).

Usage:
  python -m tools.ckpt_fingerprint path/to/ckpt.pt [--fast] [--json path/to/out.json]

Checkpoint formats supported:
  - nanogpt ckpt.pt  (keys: 'model', 'optimizer', ...)
  - raw state_dict   (dict of {name: tensor})
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from typing import Dict

import torch


def _bytes_of(t: torch.Tensor) -> bytes:
    """Return canonical bytes of a tensor: contiguous, CPU, native layout."""
    return t.detach().contiguous().cpu().numpy().tobytes()


def fingerprint_state_dict(sd: Dict[str, torch.Tensor], fast: bool = False) -> dict:
    per_param = {}
    h_total = hashlib.sha256()
    h_md5_total = hashlib.md5()
    total_bytes = 0
    for name in sorted(sd.keys()):
        v = sd[name]
        if not isinstance(v, torch.Tensor):
            continue
        if fast:
            b = _bytes_of(v)
            sample = b[:4096] + b[-4096:]
            h = hashlib.sha256(sample).hexdigest()[:16]
        else:
            b = _bytes_of(v)
            h = hashlib.sha256(b).hexdigest()[:16]
            h_total.update(b)
            h_md5_total.update(b)
            total_bytes += len(b)
        per_param[name] = {
            'shape': list(v.shape),
            'dtype': str(v.dtype),
            'numel': v.numel(),
            'hash': h,
        }
    out = {
        'mode': 'fast' if fast else 'full',
        'n_params': len(per_param),
        'per_param': per_param,
    }
    if not fast:
        out['total_bytes'] = total_bytes
        out['total_sha256'] = h_total.hexdigest()
        out['total_md5'] = h_md5_total.hexdigest()
    return out


def load_checkpoint_state_dict(path: str) -> Dict[str, torch.Tensor]:
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ck, dict) and 'model' in ck:
        sd = ck['model']
    else:
        sd = ck
    if hasattr(sd, 'state_dict'):
        sd = sd.state_dict()
    return sd


def compare(fp1: dict, fp2: dict) -> dict:
    n1 = set(fp1['per_param'])
    n2 = set(fp2['per_param'])
    only1 = sorted(n1 - n2)
    only2 = sorted(n2 - n1)
    both = sorted(n1 & n2)
    differ = []
    same = 0
    for name in both:
        a = fp1['per_param'][name]; b = fp2['per_param'][name]
        if a.get('hash') != b.get('hash') or a.get('shape') != b.get('shape'):
            differ.append({'name': name, 'a': a, 'b': b})
        else:
            same += 1
    total_match = (fp1.get('total_sha256') == fp2.get('total_sha256')
                   and fp1.get('total_sha256') is not None)
    return {
        'both': len(both),
        'same': same,
        'differ': differ[:20],
        'n_differ': len(differ),
        'only_in_a': only1[:20],
        'only_in_b': only2[:20],
        'total_sha256_match': total_match,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt', help='path to ckpt.pt')
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--json', default=None)
    ap.add_argument('--compare', default=None, help='second ckpt to compare against')
    args = ap.parse_args()

    sd = load_checkpoint_state_dict(args.ckpt)
    fp = fingerprint_state_dict(sd, fast=args.fast)
    if args.compare:
        sd2 = load_checkpoint_state_dict(args.compare)
        fp2 = fingerprint_state_dict(sd2, fast=args.fast)
        result = {'a': fp, 'b': fp2, 'compare': compare(fp, fp2)}
    else:
        result = fp
    if args.json:
        os.makedirs(os.path.dirname(args.json) or '.', exist_ok=True)
        with open(args.json, 'w') as f:
            json.dump(result, f, indent=1)
        print(f"wrote {args.json}")
    else:
        summary = {k: v for k, v in fp.items() if k != 'per_param'}
        print(json.dumps(summary, indent=1))
        print(f"sample per-param (first 5):")
        for n in sorted(fp['per_param'])[:5]:
            print(f"  {n}: {fp['per_param'][n]}")


if __name__ == '__main__':
    main()
