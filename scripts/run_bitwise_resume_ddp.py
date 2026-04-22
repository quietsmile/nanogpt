"""Phase 4 bitwise resume — DDP runner (4 ranks).

Launches train.py 3× via torchrun:
  A1: torchrun --nproc_per_node=4 train.py ... max_iters=N, out=/tmp/bw_A1
  A2: torchrun --nproc_per_node=4 train.py ... max_iters=N+M, init_from=resume, out=/tmp/bw_A2
  B:  torchrun --nproc_per_node=4 train.py ... max_iters=N+M, out=/tmp/bw_B

Rank-0 ckpt is compared bitwise. NCCL_ALGO=Ring ensures deterministic all-reduce.

Usage:
    cd /root/nanogpt
    python3 scripts/run_bitwise_resume_ddp.py --n 10 --m 10
"""
import argparse, hashlib, io, os, shutil, subprocess, sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256_of(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def run_ddp(out_dir, max_iters, eval_interval, init_from, nranks=4, seed=1337, timeout=1200):
    os.makedirs(out_dir, exist_ok=True)
    env = os.environ.copy()
    env['NCCL_ALGO'] = 'Ring'
    env['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(nranks))
    cmd = [
        'torchrun', '--standalone', f'--nproc_per_node={nranks}',
        'train.py', 'config/bitwise_resume_test.py',
        f'--out_dir={out_dir}',
        f'--max_iters={max_iters}',
        f'--eval_interval={eval_interval}',
        f'--init_from={init_from}',
        f'--seed={seed}',
    ]
    print(f"[RUN] NCCL_ALGO=Ring {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env, timeout=timeout)
    print(f"[RUN] rc={r.returncode}")
    if r.returncode != 0:
        print("STDOUT tail:", r.stdout[-3000:])
        print("STDERR tail:", r.stderr[-3000:])
    return r.returncode == 0


def summarize_ckpt(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    return {
        'iter_num': ck.get('iter_num'),
        'model_sha': sha256_of({k: v for k, v in sorted(ck['model'].items())}),
        'optim_sha': sha256_of(ck['optimizer']),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--m', type=int, default=10)
    ap.add_argument('--nranks', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1337)
    args = ap.parse_args()

    N, M = args.n, args.m
    a1_dir = '/tmp/bw_ddp_A1'
    a2_dir = '/tmp/bw_ddp_A2'
    b_dir = '/tmp/bw_ddp_B'
    for d in (a1_dir, a2_dir, b_dir):
        shutil.rmtree(d, ignore_errors=True)

    print(f"\n=== DDP-A1: train {N} steps ===")
    assert run_ddp(a1_dir, N, N, 'scratch', args.nranks, args.seed)
    os.makedirs(a2_dir, exist_ok=True)
    shutil.copy(os.path.join(a1_dir, 'ckpt.pt'), os.path.join(a2_dir, 'ckpt.pt'))

    print(f"\n=== DDP-A2: resume, train {M} more ===")
    assert run_ddp(a2_dir, N + M, M, 'resume', args.nranks, args.seed)

    print(f"\n=== DDP-B: {N+M} steps straight ===")
    assert run_ddp(b_dir, N + M, N + M, 'scratch', args.nranks, args.seed)

    a = summarize_ckpt(os.path.join(a2_dir, 'ckpt.pt'))
    b = summarize_ckpt(os.path.join(b_dir, 'ckpt.pt'))
    print(f"A iter={a['iter_num']} model={a['model_sha'][:12]} optim={a['optim_sha'][:12]}")
    print(f"B iter={b['iter_num']} model={b['model_sha'][:12]} optim={b['optim_sha'][:12]}")
    model_ok = a['model_sha'] == b['model_sha']
    optim_ok = a['optim_sha'] == b['optim_sha']
    print(f"model: {'PASS' if model_ok else 'FAIL'}")
    print(f"optim: {'PASS' if optim_ok else 'FAIL'}")
    if model_ok and optim_ok:
        print("\n✓ DDP BITWISE RESUME PASSED")
        return 0
    print("\n✗ DDP BITWISE RESUME FAILED")
    return 1


if __name__ == '__main__':
    sys.exit(main())
