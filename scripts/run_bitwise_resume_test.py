"""Phase 4 bitwise resume — single-GPU runner.

A-path:  train N steps → save ckpt → new process loads ckpt → train M more → save
B-path:  train N+M steps straight → save

Compare model + optimizer sha256 at step N+M.

Run on GPU box:
    cd /root/nanogpt
    CUDA_VISIBLE_DEVICES=0 python3 scripts/run_bitwise_resume_test.py --n 10 --m 10
"""
import argparse, hashlib, io, os, shutil, subprocess, sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256_of(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def run_train(out_dir, max_iters, eval_interval, init_from, seed=1337, timeout=900):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable, 'train.py', 'config/bitwise_resume_test.py',
        f'--out_dir={out_dir}',
        f'--max_iters={max_iters}',
        f'--eval_interval={eval_interval}',
        f'--init_from={init_from}',
        f'--seed={seed}',
    ]
    print(f"[RUN] {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    print(f"[RUN] rc={r.returncode}")
    if r.returncode != 0:
        print("STDOUT tail:", r.stdout[-2000:])
        print("STDERR tail:", r.stderr[-2000:])
    return r.returncode == 0


def summarize_ckpt(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model_sd = ck['model']
    opt_sd = ck['optimizer']
    return {
        'iter_num': ck.get('iter_num'),
        'model_sha': sha256_of({k: v for k, v in sorted(model_sd.items())}),
        'optim_sha': sha256_of(opt_sd),
        'n_params': sum(v.numel() for v in model_sd.values() if torch.is_tensor(v)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--m', type=int, default=10)
    ap.add_argument('--seed', type=int, default=1337)
    args = ap.parse_args()

    N, M = args.n, args.m
    a1_dir = '/tmp/bw_A1'
    a2_dir = '/tmp/bw_A2'
    b_dir = '/tmp/bw_B'
    for d in (a1_dir, a2_dir, b_dir):
        shutil.rmtree(d, ignore_errors=True)

    print(f"\n=== A1: train {N} steps from scratch ===")
    assert run_train(a1_dir, N, N, 'scratch', args.seed), "A1 failed"
    os.makedirs(a2_dir, exist_ok=True)
    shutil.copy(os.path.join(a1_dir, 'ckpt.pt'), os.path.join(a2_dir, 'ckpt.pt'))

    print(f"\n=== A2: resume from A1 ckpt, train {M} more (total N+M={N+M}) ===")
    assert run_train(a2_dir, N + M, M, 'resume', args.seed), "A2 failed"

    print(f"\n=== B: train N+M={N+M} straight from scratch ===")
    assert run_train(b_dir, N + M, N + M, 'scratch', args.seed), "B failed"

    print("\n=== Comparing final ckpts ===")
    a_final = summarize_ckpt(os.path.join(a2_dir, 'ckpt.pt'))
    b_final = summarize_ckpt(os.path.join(b_dir, 'ckpt.pt'))

    print(f"A iter={a_final['iter_num']} model_sha={a_final['model_sha'][:12]} optim_sha={a_final['optim_sha'][:12]}")
    print(f"B iter={b_final['iter_num']} model_sha={b_final['model_sha'][:12]} optim_sha={b_final['optim_sha'][:12]}")

    model_match = a_final['model_sha'] == b_final['model_sha']
    optim_match = a_final['optim_sha'] == b_final['optim_sha']
    iter_match = a_final['iter_num'] == b_final['iter_num']
    print(f"\nmodel state_dict: {'PASS' if model_match else 'FAIL'}")
    print(f"optimizer state: {'PASS' if optim_match else 'FAIL'}")
    print(f"iter_num match:  {'PASS' if iter_match else 'FAIL'}")

    if model_match and optim_match and iter_match:
        print("\n✓ BITWISE RESUME PASSED")
        return 0
    # Dive deeper on failure
    print("\n✗ BITWISE RESUME FAILED — per-param diff:")
    a_ck = torch.load(os.path.join(a2_dir, 'ckpt.pt'), map_location='cpu', weights_only=False)
    b_ck = torch.load(os.path.join(b_dir, 'ckpt.pt'), map_location='cpu', weights_only=False)
    a_sd = a_ck['model']; b_sd = b_ck['model']
    diffs = []
    for k in sorted(a_sd.keys()):
        if k not in b_sd:
            print(f"  {k}: missing in B")
            continue
        av, bv = a_sd[k], b_sd[k]
        if not torch.is_tensor(av):
            continue
        if not torch.equal(av, bv):
            d = (av.float() - bv.float()).abs()
            diffs.append((k, d.max().item(), d.mean().item()))
    diffs.sort(key=lambda x: -x[1])
    for k, mx, mn in diffs[:20]:
        print(f"  {k}: max={mx:.3e} mean={mn:.3e}")
    return 1


if __name__ == '__main__':
    sys.exit(main())
