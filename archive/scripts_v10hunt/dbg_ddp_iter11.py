"""Confirm DDP resume state is bitwise-identical at iter 11 (1 step post-resume).

A: ckpt at iter 10 (A1) → resume → save at iter 11 (max_iters=11 eval_interval=1)
B: scratch → save at iter 11 (max_iters=11 eval_interval=1)
Compare model + optimizer sha at iter 11.

Then: compare at iter 12 too (max_iters=12 eval_interval=2).
"""
import hashlib, io, os, shutil, subprocess, sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def sha(o):
    buf=io.BytesIO(); torch.save(o, buf); return hashlib.sha256(buf.getvalue()).hexdigest()

def run(out_dir, max_iters, eval_interval, init_from, timeout=900):
    os.makedirs(out_dir, exist_ok=True)
    env = os.environ.copy(); env['NCCL_ALGO'] = 'Ring'
    cmd = ['torchrun', '--standalone', '--nproc_per_node=4',
           'train.py', 'config/bitwise_resume_test.py',
           f'--out_dir={out_dir}', f'--max_iters={max_iters}',
           f'--eval_interval={eval_interval}', f'--init_from={init_from}',
           '--seed=1337']
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env, timeout=timeout)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2000:])
    return r.returncode == 0

def summary(p):
    c = torch.load(p, map_location='cpu', weights_only=False)
    return {'iter': c.get('iter_num'),
            'model': sha({k:v for k,v in sorted(c['model'].items())}),
            'optim': sha(c['optimizer'])}

for d in ['/tmp/dbg11_A1','/tmp/dbg11_A2_11','/tmp/dbg11_B_11','/tmp/dbg11_A2_12','/tmp/dbg11_B_12']:
    shutil.rmtree(d, ignore_errors=True)

# Train A1 for 10 steps
print("A1 10 steps...")
assert run('/tmp/dbg11_A1', 10, 10, 'scratch')

# A2_11: resume → train 1 more, save at iter 11
os.makedirs('/tmp/dbg11_A2_11', exist_ok=True)
shutil.copy('/tmp/dbg11_A1/ckpt.pt', '/tmp/dbg11_A2_11/ckpt.pt')
print("A2 resume to iter 11...")
assert run('/tmp/dbg11_A2_11', 11, 1, 'resume')

# B_11: scratch to iter 11
print("B scratch to iter 11...")
assert run('/tmp/dbg11_B_11', 11, 1, 'scratch')

# A2_12: resume → train 2 more, save at iter 12
os.makedirs('/tmp/dbg11_A2_12', exist_ok=True)
shutil.copy('/tmp/dbg11_A1/ckpt.pt', '/tmp/dbg11_A2_12/ckpt.pt')
print("A2 resume to iter 12...")
assert run('/tmp/dbg11_A2_12', 12, 2, 'resume')

# B_12: scratch to iter 12
print("B scratch to iter 12...")
assert run('/tmp/dbg11_B_12', 12, 2, 'scratch')

a11, b11 = summary('/tmp/dbg11_A2_11/ckpt.pt'), summary('/tmp/dbg11_B_11/ckpt.pt')
a12, b12 = summary('/tmp/dbg11_A2_12/ckpt.pt'), summary('/tmp/dbg11_B_12/ckpt.pt')
print(f"\niter 11: A model={a11['model'][:12]} optim={a11['optim'][:12]}")
print(f"         B model={b11['model'][:12]} optim={b11['optim'][:12]}")
print(f"         model_eq={a11['model']==b11['model']}  optim_eq={a11['optim']==b11['optim']}")
print(f"\niter 12: A model={a12['model'][:12]} optim={a12['optim'][:12]}")
print(f"         B model={b12['model'][:12]} optim={b12['optim'][:12]}")
print(f"         model_eq={a12['model']==b12['model']}  optim_eq={a12['optim']==b12['optim']}")
