"""Locate DDP bitwise resume divergence by comparing intermediate ckpts."""
import hashlib, io, os, subprocess, sys, shutil
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def sha(obj):
    buf = io.BytesIO(); torch.save(obj, buf); return hashlib.sha256(buf.getvalue()).hexdigest()

def run(out_dir, max_iters, eval_interval, init_from, timeout=900):
    os.makedirs(out_dir, exist_ok=True)
    env = os.environ.copy(); env['NCCL_ALGO'] = 'Ring'
    cmd = ['torchrun', '--standalone', '--nproc_per_node=4',
           'train.py', 'config/bitwise_resume_test.py',
           f'--out_dir={out_dir}', f'--max_iters={max_iters}',
           f'--eval_interval={eval_interval}', f'--init_from={init_from}',
           '--seed=1337']
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env, timeout=timeout)
    return r.returncode == 0

def summary(p):
    c = torch.load(p, map_location='cpu', weights_only=False)
    return {
        'iter': c.get('iter_num'),
        'model_sha': sha({k:v for k,v in sorted(c['model'].items())}),
        'optim_sha': sha(c['optimizer']),
        'rng_cpu_sha': sha(c.get('rng_state_cpu')),
        'rng_cuda_sha': sha(c.get('rng_state_cuda')),
        'seq_pos': c.get('seq_data_pos'),
        'has_per_rank': c.get('_per_rank') is not None,
        'per_rank_seq_pos': [r['seq_data_pos'] for r in c['_per_rank']] if c.get('_per_rank') else None,
    }

# 1) Train A1 for 10 steps
for d in ['/tmp/bw_dbg_A1', '/tmp/bw_dbg_A2_iter10', '/tmp/bw_dbg_B10', '/tmp/bw_dbg_B20', '/tmp/bw_dbg_A2']:
    shutil.rmtree(d, ignore_errors=True)
assert run('/tmp/bw_dbg_A1', 10, 10, 'scratch')
assert run('/tmp/bw_dbg_B10', 10, 10, 'scratch')
s1 = summary('/tmp/bw_dbg_A1/ckpt.pt')
s2 = summary('/tmp/bw_dbg_B10/ckpt.pt')
print('A1@10 vs B@10:')
for k in s1:
    v1, v2 = s1[k], s2[k]
    ok = (v1 == v2) if not isinstance(v1, list) else (v1 == v2)
    print(f"  {k}: {'EQ' if ok else 'DIFF'}  A={v1}  B={v2}")

# 2) Run B for 20 and resume A to 20
os.makedirs('/tmp/bw_dbg_A2', exist_ok=True)
shutil.copy('/tmp/bw_dbg_A1/ckpt.pt', '/tmp/bw_dbg_A2/ckpt.pt')
assert run('/tmp/bw_dbg_A2', 20, 10, 'resume')
assert run('/tmp/bw_dbg_B20', 20, 20, 'scratch')
s3 = summary('/tmp/bw_dbg_A2/ckpt.pt')
s4 = summary('/tmp/bw_dbg_B20/ckpt.pt')
print('\nA2@20 vs B@20:')
for k in s3:
    v3, v4 = s3[k], s4[k]
    print(f"  {k}: {'EQ' if v3 == v4 else 'DIFF'}  A={v3}  B={v4}")
