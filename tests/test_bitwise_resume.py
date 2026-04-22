"""Phase 4: bitwise resume validation.

REQUIRES GPU. Skipped on DSW-only environments.

Run on one of the 8-GPU H100 boxes (see MEMORY gpu_8card_ips.md):
  torchrun --standalone --nproc_per_node=1 -m pytest tests/test_bitwise_resume.py::TestBitwiseResume::test_single_gpu -v
  torchrun --standalone --nproc_per_node=4 -m pytest tests/test_bitwise_resume.py::TestBitwiseResume::test_ddp -v

Strategy:
  A-path: seed S, deterministic mode, N=20 steps, save ckpt → new process loads and runs M=20 more.
  B-path: seed S, same config, run N+M=40 steps straight.
  Assert: at step N+M, A and B produce identical:
    - sorted state_dict sha256 (from tools.ckpt_fingerprint)
    - optimizer.state_dict sha256
    - loss float32 bit pattern

On failure, bisect to find first diverging step and dump RNG/dataloader/optimizer diff
to reports/bitwise_resume_divergence.json.
"""
import hashlib
import json
import os
import struct
import subprocess
import sys
import unittest

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Canonical report — never overwritten by the scaffold runner. Real measurements
# live here (sha256 of state_dict/optim, A-vs-B drift, etc.). Updated manually
# or by a future real runner, not by the scaffold.
REPORT = os.path.join(ROOT, 'reports', 'bitwise_resume.json')
# Scaffold runner output — the current stub runner writes here so it stops
# clobbering the canonical report on every pytest run. Once the runner is
# wired into train.py properly, this can merge back into REPORT.
SCAFFOLD_OUT = os.path.join(ROOT, 'reports', 'bitwise_resume_scaffold_out.json')
DIVERGENCE = os.path.join(ROOT, 'reports', 'bitwise_resume_divergence.json')


def _has_cuda():
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def _loss_bits(x):
    return struct.pack('<f', float(x)).hex()


@unittest.skipUnless(_has_cuda(), "no CUDA device — skipping bitwise resume tests")
class TestBitwiseResume(unittest.TestCase):
    """The actual logic runs via a companion runner script because it needs
    subprocess-level process restart for 'path A resume from ckpt'."""

    def test_single_gpu(self):
        runner = os.path.join(ROOT, 'scripts', 'bitwise_resume_runner.py')
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'
        cmd = [sys.executable, runner, '--n-steps', '20', '--m-steps', '20',
               '--out', SCAFFOLD_OUT, '--mode', 'single']
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
        print("STDOUT:", r.stdout[-2000:])
        print("STDERR:", r.stderr[-2000:])
        self.assertEqual(r.returncode, 0, "runner failed")
        with open(SCAFFOLD_OUT) as f:
            result = json.load(f)
        if result.get('scaffold'):
            self.skipTest('runner is a scaffold — train.py N-step/resume flags not wired yet')
        self.assertTrue(result.get('pass'), f"A/B diverged: {result.get('differ', '<no detail>')}")

    def test_ddp(self):
        runner = os.path.join(ROOT, 'scripts', 'bitwise_resume_runner.py')
        env = os.environ.copy()
        env['NCCL_ALGO'] = 'Ring'  # deterministic all-reduce order
        cmd = ['torchrun', '--standalone', '--nproc_per_node=4',
               runner, '--n-steps', '10', '--m-steps', '10',
               '--out', SCAFFOLD_OUT, '--mode', 'ddp']
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
        print("STDOUT:", r.stdout[-2000:])
        print("STDERR:", r.stderr[-2000:])
        self.assertEqual(r.returncode, 0, "DDP runner failed")
        with open(SCAFFOLD_OUT) as f:
            result = json.load(f)
        if result.get('scaffold'):
            self.skipTest('runner is a scaffold — train.py N-step/resume flags not wired yet')
        self.assertTrue(result.get('pass'), f"DDP A/B diverged: {result.get('differ')}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
