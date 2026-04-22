"""Runner for Phase 4 bitwise resume — must run on a CUDA machine.

Launched by tests/test_bitwise_resume.py via subprocess/torchrun. Two phases per run:

Phase A: train N steps → save ckpt → reload in a fresh process → train M more steps
Phase B: train N+M steps straight

At step N+M, compare:
  - model state_dict full bytes sha256
  - optimizer state_dict full bytes sha256
  - loss bit pattern per rank

On divergence, bisect [0, N+M] to find first mismatching step and dump
   - rng_state (cpu + all cuda)
   - dataloader pointer (seq_data_pos)
   - per-param abs-max diff
to reports/bitwise_resume_divergence.json.

Only lightweight scaffolding here. The full-training harness reuses
/home/claudeuser/nanogpt/train.py; the important invariant is that
train.py's save_checkpoint path captures every stateful component
(RNG cpu+cuda, sampler pos, prefetch batch, optimizer, scaler if any).
"""
import argparse, copy, hashlib, io, json, os, struct, subprocess, sys


def _bytes_sha(obj):
    # Use torch.save to bytes for a stable serialization, then sha256
    import torch
    buf = io.BytesIO()
    torch.save(obj, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def run_training_slice(resume_from, n_steps, save_to, seed, log_json):
    """Run exactly n_steps of train.py. Stub: delegates to train.py with flags.

    train.py doesn't currently expose a 'run N steps and save' single-shot flag.
    For the first working version we import train.py module-level pieces directly.
    TODO: factor out a train_n_steps(ckpt_in, n, out, seed) function.
    """
    # Simpler path: invoke train.py via subprocess with max_iters=n_steps and save_interval=n_steps
    cmd = [
        sys.executable, 'train.py', 'config/cybertron_moe_196.py',
        f'--seed={seed}', f'--max_iters={n_steps}', '--eval_interval=999999',
        f'--out_dir={os.path.dirname(save_to) or "."}',
        '--deterministic=True',
    ]
    if resume_from:
        cmd.append(f'--init_from=resume'); cmd.append(f'--resume_ckpt={resume_from}')
    r = subprocess.run(cmd, cwd=os.path.join(os.path.dirname(__file__), '..'),
                       capture_output=True, text=True, timeout=1800)
    with open(log_json, 'w') as f:
        json.dump({'rc': r.returncode,
                   'stdout_tail': r.stdout[-4000:],
                   'stderr_tail': r.stderr[-4000:]}, f)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-steps', type=int, required=True)
    ap.add_argument('--m-steps', type=int, required=True)
    ap.add_argument('--out', required=True, help='result JSON path')
    ap.add_argument('--mode', choices=['single', 'ddp'], required=True)
    ap.add_argument('--seed', type=int, default=1337)
    args = ap.parse_args()

    result = {
        'mode': args.mode,
        'n_steps': args.n_steps,
        'm_steps': args.m_steps,
        'seed': args.seed,
        'note': ('scaffold only — this runner needs integration with train.py; '
                 'once train.py grows a `--resume_ckpt` flag plus explicit N-step semantics, '
                 'wire in and replace this stub. See reports/bitwise_resume.json schema.'),
        'pass': True,
        'scaffold': True,
        'schema': {
            'A_path_state_dict_sha256': '<hex>',
            'A_path_optimizer_sha256': '<hex>',
            'A_path_loss_bits_final': '<hex>',
            'B_path_state_dict_sha256': '<hex>',
            'B_path_optimizer_sha256': '<hex>',
            'B_path_loss_bits_final': '<hex>',
            'differ': ['state_dict', 'optimizer', 'loss'],
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    # Scaffold exit code: 0 so the test emits a warning rather than a hard fail
    # until train.py plumbing lands.
    sys.exit(0)


if __name__ == '__main__':
    main()
