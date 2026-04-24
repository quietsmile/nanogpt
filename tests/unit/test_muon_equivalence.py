"""Bitwise equivalence: nanogpt.optim.Muon(pipeline=recipes.muon_megatron)
    == muon_megatron.Muon(...)

Tiny CPU tensor, multi-step sequence. Each step's result (p after step()) must
be exactly equal to legacy Muon output.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from muon_megatron import Muon as LegacyMegMuon  # noqa: E402

from nanogpt.optim import Muon as NewMuon  # noqa: E402
from nanogpt.optim.recipes import muon_megatron as recipe_megatron  # noqa: E402


def _build_params(seed, shape, count=3):
    torch.manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(*shape)) for _ in range(count)]


def test_muon_megatron_bitwise_5_steps_2d():
    torch.manual_seed(0)
    # Two param sets with same init
    params_legacy = _build_params(seed=0, shape=(32, 64), count=2)
    params_new = _build_params(seed=0, shape=(32, 64), count=2)

    lr, wd, mom = 1.2e-3, 0.1, 0.95

    legacy_opt = LegacyMegMuon(
        params_legacy, lr=lr, momentum_beta=mom, use_nesterov=True,
        weight_decay=wd, use_decoupled_weight_decay=True,
        coefficient_type="quintic", num_ns_steps=5,
        scale_mode="spectral", muon_matched_adamw_rms=0.2,
        fp32_matmul_prec="medium",
    )
    new_opt = NewMuon(
        params_new, pipeline=recipe_megatron(), lr=lr, weight_decay=wd,
    )

    for step_i in range(5):
        # Same random gradients for both
        torch.manual_seed(100 + step_i)
        for p_l, p_n in zip(params_legacy, params_new):
            g = torch.randn_like(p_l.data)
            p_l.grad = g.clone()
            p_n.grad = g.clone()
        legacy_opt.step()
        new_opt.step()
        for i, (p_l, p_n) in enumerate(zip(params_legacy, params_new)):
            diff = (p_l.data - p_n.data).abs().max().item()
            assert torch.equal(p_l.data, p_n.data), \
                f"step {step_i} param {i} not bitwise-equal, max|Δ|={diff}"


def test_muon_megatron_bitwise_3d_moe():
    torch.manual_seed(1)
    # 3D batched MoE expert weight shape [E, C, H]
    params_legacy = _build_params(seed=1, shape=(8, 32, 64), count=1)
    params_new = _build_params(seed=1, shape=(8, 32, 64), count=1)

    legacy_opt = LegacyMegMuon(
        params_legacy, lr=1.2e-3, momentum_beta=0.95, use_nesterov=True,
        weight_decay=0.1, use_decoupled_weight_decay=True,
        coefficient_type="quintic", num_ns_steps=5,
        scale_mode="spectral", muon_matched_adamw_rms=0.2,
        fp32_matmul_prec="medium",
    )
    new_opt = NewMuon(
        params_new, pipeline=recipe_megatron(), lr=1.2e-3, weight_decay=0.1,
    )

    for step_i in range(3):
        torch.manual_seed(200 + step_i)
        for p_l, p_n in zip(params_legacy, params_new):
            g = torch.randn_like(p_l.data)
            p_l.grad = g.clone()
            p_n.grad = g.clone()
        legacy_opt.step()
        new_opt.step()
        for p_l, p_n in zip(params_legacy, params_new):
            assert torch.equal(p_l.data, p_n.data)


def test_muon_rejects_invalid_pipeline():
    import pytest
    p = torch.nn.Parameter(torch.ones(2, 2))
    with pytest.raises(ValueError):
        NewMuon([p], pipeline=[], lr=1e-3)
    with pytest.raises(TypeError):
        NewMuon([p], pipeline=[object()], lr=1e-3)


def test_muon_rejects_non_2d_3d_params():
    import pytest
    p = torch.nn.Parameter(torch.ones(4))  # 1D
    p.grad = torch.ones(4)
    new_opt = NewMuon([p], pipeline=recipe_megatron(), lr=1e-3)
    with pytest.raises(ValueError):
        new_opt.step()


def test_muon_hook_fires_per_step_name():
    p = torch.nn.Parameter(torch.randn(4, 8))
    p.grad = torch.randn(4, 8)
    names_seen = []

    def hook(step_name, ctx):
        names_seen.append(step_name)

    opt = NewMuon([p], pipeline=recipe_megatron(), lr=1e-3, hook=hook)
    opt.step()
    # 4 steps in the default megatron recipe
    assert names_seen == ["DecoupledWD", "Momentum", "NewtonSchulz", "SpectralScale"]
