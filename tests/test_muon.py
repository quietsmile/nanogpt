"""Unit tests for muon.py: Polar Express orthogonality, 3D batched math, and
MultiOptimizer state_dict round-trip.

These run CPU-only (small shapes); no GPU required. The bitwise-resume integration
test for Muon lives in test_bitwise_resume.py once enabled there.
"""
import io
import torch
import pytest

from muon import Muon, MultiOptimizer, polar_express_orthogonalize


def _sv_check(U: torch.Tensor, tol: float = 0.20):
    """Verify singular values cluster near 1. Polar Express is approximate by design;
    the published 5-iter coefficients give SVs roughly in [0.85, 1.15] for non-square
    inputs from a Gaussian distribution."""
    U = U.float()
    if U.dim() == 3:
        for i in range(U.size(0)):
            svs = torch.linalg.svdvals(U[i])
            assert svs.min() > 1 - tol, f"batch {i}: min SV {svs.min().item():.3f}"
            assert svs.max() < 1 + tol, f"batch {i}: max SV {svs.max().item():.3f}"
    else:
        svs = torch.linalg.svdvals(U)
        assert svs.min() > 1 - tol, f"min SV {svs.min().item():.3f}"
        assert svs.max() < 1 + tol, f"max SV {svs.max().item():.3f}"


class TestPolarExpress:
    def test_2d_wide(self):
        torch.manual_seed(0)
        X = torch.randn(64, 256)
        U = polar_express_orthogonalize(X)
        assert U.shape == X.shape
        _sv_check(U)

    def test_2d_tall(self):
        torch.manual_seed(0)
        X = torch.randn(256, 64)
        U = polar_express_orthogonalize(X)
        assert U.shape == X.shape
        _sv_check(U)

    def test_2d_nonsquare_aspect(self):
        """Cybertron MoE down_weight aspect: 160 x 512."""
        torch.manual_seed(0)
        X = torch.randn(160, 512)
        U = polar_express_orthogonalize(X)
        _sv_check(U)

    def test_3d_batched_wide(self):
        """MoE expert weight shape: [E, in, out] with in < out (gate_weight, up_weight)."""
        torch.manual_seed(0)
        X = torch.randn(8, 64, 160)
        U = polar_express_orthogonalize(X)
        assert U.shape == X.shape
        _sv_check(U)

    def test_3d_batched_tall(self):
        """MoE down_weight shape variant [E, hidden, in] tall path."""
        torch.manual_seed(0)
        X = torch.randn(8, 160, 64)
        U = polar_express_orthogonalize(X)
        assert U.shape == X.shape
        _sv_check(U)

    def test_independence_across_batch(self):
        """Per-expert orthogonalization must not mix experts: batched vs solo
        must agree (modulo tiny bf16 tile-level rounding)."""
        torch.manual_seed(0)
        X = torch.randn(4, 32, 32)
        U_batched = polar_express_orthogonalize(X)
        for i in range(4):
            U_solo = polar_express_orthogonalize(X[i:i+1])
            # bf16 tile-level matmul can reorder reductions across batch sizes; allow
            # a small relative tolerance.
            diff = (U_batched[i:i+1].float() - U_solo.float()).norm().item()
            ref_norm = U_batched[i:i+1].float().norm().item()
            assert diff / max(ref_norm, 1e-9) < 0.02, f"expert {i} mismatch: rel diff {diff/ref_norm:.4f}"


class TestMuonStep:
    def test_step_runs_2d(self):
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.randn(64, 32))
        opt = Muon([p], lr=0.05, weight_decay=0.01)
        before = p.data.clone()
        p.grad = torch.randn_like(p)
        opt.step()
        assert not torch.equal(p.data, before)
        assert p.data.shape == before.shape

    def test_step_runs_3d(self):
        """Muon must accept 3D batched MoE expert weights."""
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.randn(8, 64, 32))
        opt = Muon([p], lr=0.05, weight_decay=0.01)
        before = p.data.clone()
        p.grad = torch.randn_like(p)
        opt.step()
        assert not torch.equal(p.data, before)

    def test_rejects_1d(self):
        p = torch.nn.Parameter(torch.randn(32))
        opt = Muon([p], lr=0.05)
        p.grad = torch.randn_like(p)
        with pytest.raises(ValueError, match="2D or 3D"):
            opt.step()

    def test_state_dict_roundtrip(self):
        """Optimizer state must round-trip through torch.save/torch.load (the realistic
        ckpt path; in-memory dict transfer aliases tensors and would mutate both sides)."""
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.randn(64, 32))
        opt = Muon([p], lr=0.05, weight_decay=0.01)
        for _ in range(3):
            p.grad = torch.randn_like(p)
            opt.step()

        # Serialize through a buffer to get a true copy.
        buf = io.BytesIO()
        torch.save(opt.state_dict(), buf)
        buf.seek(0)
        loaded = torch.load(buf, weights_only=False)

        p2 = torch.nn.Parameter(p.data.clone())
        opt2 = Muon([p2], lr=0.05, weight_decay=0.01)
        opt2.load_state_dict(loaded)

        g = torch.randn_like(p)
        p.grad = g.clone()
        p2.grad = g.clone()
        opt.step()
        opt2.step()
        assert torch.allclose(p.data, p2.data, atol=1e-5, rtol=1e-5)


class TestMultiOptimizer:
    def test_step_dispatches_to_inner(self):
        torch.manual_seed(0)
        p_a = torch.nn.Parameter(torch.randn(16))
        p_m = torch.nn.Parameter(torch.randn(64, 32))
        adam = torch.optim.AdamW([p_a], lr=1e-3)
        muon = Muon([p_m], lr=0.05)
        wrap = MultiOptimizer({"adamw": adam, "muon": muon})

        before_a, before_m = p_a.data.clone(), p_m.data.clone()
        p_a.grad = torch.randn_like(p_a)
        p_m.grad = torch.randn_like(p_m)
        wrap.step()
        assert not torch.equal(p_a.data, before_a)
        assert not torch.equal(p_m.data, before_m)

    def test_param_groups_concat(self):
        p_a = torch.nn.Parameter(torch.randn(16))
        p_m = torch.nn.Parameter(torch.randn(64, 32))
        adam = torch.optim.AdamW([p_a], lr=1e-3)
        muon = Muon([p_m], lr=0.05)
        wrap = MultiOptimizer({"adamw": adam, "muon": muon})
        groups = wrap.param_groups
        assert len(groups) == 2
        # Order: adam first (insertion order in dict)
        assert groups[0]["lr"] == pytest.approx(1e-3)
        assert groups[1]["lr"] == pytest.approx(0.05)

    def test_state_dict_roundtrip(self):
        torch.manual_seed(0)
        p_a = torch.nn.Parameter(torch.randn(16))
        p_m = torch.nn.Parameter(torch.randn(64, 32))
        adam = torch.optim.AdamW([p_a], lr=1e-3)
        muon = Muon([p_m], lr=0.05)
        wrap = MultiOptimizer({"adamw": adam, "muon": muon})

        for _ in range(3):
            p_a.grad = torch.randn_like(p_a)
            p_m.grad = torch.randn_like(p_m)
            wrap.step()

        # Serialize/deserialize for a true copy (the realistic ckpt save/load path).
        buf = io.BytesIO()
        torch.save(wrap.state_dict(), buf)
        buf.seek(0)
        loaded = torch.load(buf, weights_only=False)
        assert loaded.get("_multi") is True
        assert "adamw" in loaded and "muon" in loaded

        p_a2 = torch.nn.Parameter(p_a.data.clone())
        p_m2 = torch.nn.Parameter(p_m.data.clone())
        adam2 = torch.optim.AdamW([p_a2], lr=1e-3)
        muon2 = Muon([p_m2], lr=0.05)
        wrap2 = MultiOptimizer({"adamw": adam2, "muon": muon2})
        wrap2.load_state_dict(loaded)

        g_a = torch.randn_like(p_a)
        g_m = torch.randn_like(p_m)
        p_a.grad, p_a2.grad = g_a.clone(), g_a.clone()
        p_m.grad, p_m2.grad = g_m.clone(), g_m.clone()
        wrap.step()
        wrap2.step()
        assert torch.allclose(p_a.data, p_a2.data, atol=1e-5)
        assert torch.allclose(p_m.data, p_m2.data, atol=1e-5)

    def test_load_rejects_baseline_format(self):
        """Loading a stock AdamW state_dict (no _multi marker) must raise, to prevent
        silent state loss on resume from a baseline ckpt."""
        p = torch.nn.Parameter(torch.randn(16))
        adam = torch.optim.AdamW([p], lr=1e-3)
        wrap = MultiOptimizer({"adamw": torch.optim.AdamW([p], lr=1e-3)})
        with pytest.raises(ValueError, match="MultiOptimizer-format"):
            wrap.load_state_dict(adam.state_dict())
