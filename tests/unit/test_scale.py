"""Scaling steps — bitwise vs muon_megatron.get_muon_scale_factor."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from muon_megatron import get_muon_scale_factor as legacy_scale  # noqa: E402

from nanogpt.optim import StepContext  # noqa: E402
from nanogpt.optim.steps import ShapeScale, SpectralScale, UnitRMSScale  # noqa: E402


@pytest.mark.parametrize("out_dim,in_dim", [(512, 1536), (2048, 512), (144, 512), (160, 512)])
def test_spectral_scale_matches_legacy(out_dim, in_dim):
    p = torch.empty(out_dim, in_dim)
    ref = legacy_scale(out_dim, in_dim, mode="spectral", muon_matched_adamw_rms=0.2)

    step = SpectralScale(muon_matched_adamw_rms=0.2)
    ctx = StepContext(param=p, grad=torch.ones(out_dim, in_dim), state={}, group={})
    step(ctx)
    assert abs(ctx.lr_scale - ref) < 1e-12, f"{ctx.lr_scale} vs {ref}"


@pytest.mark.parametrize("out_dim,in_dim", [(512, 1536), (1536, 512)])
def test_shape_scale_matches_legacy(out_dim, in_dim):
    ref = legacy_scale(out_dim, in_dim, mode="shape_scaling")
    p = torch.empty(out_dim, in_dim)
    step = ShapeScale()
    ctx = StepContext(param=p, grad=torch.ones(out_dim, in_dim), state={}, group={})
    step(ctx)
    assert abs(ctx.lr_scale - ref) < 1e-12


@pytest.mark.parametrize("out_dim,in_dim", [(512, 1536), (160, 512), (2048, 512)])
def test_unit_rms_scale_matches_legacy(out_dim, in_dim):
    ref = legacy_scale(out_dim, in_dim, mode="unit_rms_norm")
    p = torch.empty(out_dim, in_dim)
    step = UnitRMSScale()
    ctx = StepContext(param=p, grad=torch.ones(out_dim, in_dim), state={}, group={})
    step(ctx)
    assert abs(ctx.lr_scale - ref) < 1e-12


def test_spectral_scale_3d_uses_last_two_dims():
    p = torch.empty(144, 512, 1536)  # MoE expert stacked shape
    step = SpectralScale(muon_matched_adamw_rms=0.2)
    ctx = StepContext(param=p, grad=torch.ones_like(p), state={}, group={})
    step(ctx)
    # scale from (512, 1536), not (144, 1536)
    expected = legacy_scale(512, 1536, mode="spectral", muon_matched_adamw_rms=0.2)
    assert abs(ctx.lr_scale - expected) < 1e-12


def test_spectral_scale_sets_update_to_grad_casted_to_param_dtype():
    p = torch.empty(4, 8, dtype=torch.bfloat16)
    g = torch.ones(4, 8, dtype=torch.float32)
    step = SpectralScale()
    ctx = StepContext(param=p, grad=g, state={}, group={})
    step(ctx)
    assert ctx.update is not None
    assert ctx.update.dtype == torch.bfloat16
