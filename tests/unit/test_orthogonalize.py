"""NewtonSchulz / PolarExpress vs v1.0.0 muon_megatron.newton_schulz — bitwise fp32."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # for muon_megatron

from muon_megatron import newton_schulz as legacy_newton_schulz  # noqa: E402

from nanogpt.optim import StepContext  # noqa: E402
from nanogpt.optim.steps import NewtonSchulz, PolarExpress  # noqa: E402


def test_newton_schulz_2d_quintic_bitwise():
    torch.manual_seed(42)
    x = torch.randn(16, 32, dtype=torch.float32)

    legacy = legacy_newton_schulz(x.clone(), steps=5, coefficient_type="quintic")
    step = NewtonSchulz(coefficient_type="quintic", steps=5)
    ctx = StepContext(param=torch.empty(16, 32), grad=x.clone(), state={}, group={})
    step(ctx)

    # bitwise at fp32
    assert torch.equal(legacy, ctx.grad), f"max|Δ|={(legacy - ctx.grad).abs().max()}"


def test_newton_schulz_3d_quintic_bitwise():
    torch.manual_seed(7)
    x = torch.randn(4, 8, 16, dtype=torch.float32)

    legacy = legacy_newton_schulz(x.clone(), steps=5, coefficient_type="quintic")
    step = NewtonSchulz(coefficient_type="quintic", steps=5)
    ctx = StepContext(param=torch.empty(4, 8, 16), grad=x.clone(), state={}, group={})
    step(ctx)

    assert torch.equal(legacy, ctx.grad)


def test_newton_schulz_tall_matrix_transpose_bitwise():
    """Input with rows > cols triggers internal transpose — verify it still matches."""
    torch.manual_seed(11)
    x = torch.randn(64, 16, dtype=torch.float32)  # m > n

    legacy = legacy_newton_schulz(x.clone(), steps=5, coefficient_type="quintic")
    step = NewtonSchulz(coefficient_type="quintic", steps=5)
    ctx = StepContext(param=torch.empty(64, 16), grad=x.clone(), state={}, group={})
    step(ctx)

    assert torch.equal(legacy, ctx.grad)


def test_polar_express_bitwise():
    torch.manual_seed(3)
    x = torch.randn(8, 16, dtype=torch.float32)

    legacy = legacy_newton_schulz(x.clone(), steps=8, coefficient_type="polar_express")
    step = PolarExpress(steps=8)
    ctx = StepContext(param=torch.empty(8, 16), grad=x.clone(), state={}, group={})
    step(ctx)

    assert torch.equal(legacy, ctx.grad)


def test_newton_schulz_unknown_coeff_rejected():
    with pytest.raises(ValueError):
        NewtonSchulz(coefficient_type="not_a_thing", steps=5)


def test_newton_schulz_wrong_steps_rejected():
    step = NewtonSchulz(coefficient_type="quintic_new", steps=7)  # 7 not multiple of 5
    ctx = StepContext(param=torch.empty(4, 4), grad=torch.randn(4, 4), state={}, group={})
    with pytest.raises(ValueError):
        step(ctx)


def test_polar_express_requires_multiple_of_8():
    with pytest.raises(ValueError):
        PolarExpress(steps=5)


def test_newton_schulz_records_meta():
    step = NewtonSchulz(coefficient_type="quintic", steps=5)
    ctx = StepContext(param=torch.empty(4, 4), grad=torch.randn(4, 4), state={}, group={})
    step(ctx)
    assert ctx.meta["orth_steps"] == 5
    assert ctx.meta["orth_coeffs"] == "quintic"


def test_newton_schulz_rejects_non_fp32():
    step = NewtonSchulz()
    ctx = StepContext(param=torch.empty(4, 4),
                      grad=torch.randn(4, 4, dtype=torch.float16),
                      state={}, group={})
    with pytest.raises(ValueError):
        step(ctx)
