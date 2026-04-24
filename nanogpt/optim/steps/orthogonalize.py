"""Orthogonalization steps — project gradient to orthogonal matrix.

Two algorithms:
  NewtonSchulz   zeroth-power via quintic polynomial iteration (Megatron style).
  PolarExpress   different coefficients, 8 steps, from arxiv 2505.16932.

Both operate on the last two dims. Supports 2D (m,n) and 3D batched (B,m,n).
Whitens on the smaller dim (transpose first if m>n).

The numerical body mirrors `muon_megatron.newton_schulz` line-for-line so
the new impl is bitwise-equivalent to v1.0.0 at fp32 in/out with the same
coefficients and matmul precision ('medium' = bf16 internal).
"""
from __future__ import annotations

import torch

from ..pipeline import OptimStep, StepContext

# Coefficient sets — verbatim from megatron-LM muon_utils.py.
_COEFFICIENT_SETS: dict[str, list[tuple[float, float, float]]] = {
    "simple": [
        (3.4445, -4.7750, 2.0315),
    ],
    "quintic": [
        (3.4445, -4.7750, 2.0315),
    ],
    "polar_express": [
        (7.2086, -15.5131, 9.0178),
        (3.9623, -2.5813, 0.4542),
        (3.9466, -2.5765, 0.4544),
        (3.8991, -2.5671, 0.4566),
        (3.7186, -2.5308, 0.4653),
        (3.1390, -2.3073, 0.4733),
        (2.1715, -1.5246, 0.3885),
        (1.8648, -1.2224, 0.3577),
    ],
    "quintic_new": [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ],
}


def _orth(x: torch.Tensor, coeffs: list[tuple[float, float, float]], steps: int,
          fp32_matmul_prec: str) -> torch.Tensor:
    """Shared iteration body (NS and Polar Express differ only in coeffs)."""
    if x.ndim < 2:
        raise ValueError(f"orth requires >=2 dims, got {x.shape}")
    if x.dtype != torch.float32:
        raise ValueError(f"orth expects fp32 input, got {x.dtype}")

    needs_transpose = x.size(-2) > x.size(-1)
    if needs_transpose:
        x = x.mT

    # Spectral-norm cap: ||X||_F <= 1 per matrix.
    X = torch.nn.functional.normalize(x, p=2, dim=(-2, -1), eps=1e-7)

    if steps % len(coeffs) != 0:
        raise ValueError(f"steps ({steps}) must be multiple of len(coeffs) ({len(coeffs)})")

    if fp32_matmul_prec == "medium":
        X = X.to(torch.bfloat16)

    for i in range(steps):
        a, b, c = coeffs[i % len(coeffs)]
        A = X @ X.mT
        B = torch.empty_like(A)
        torch.matmul(A, A, out=B).mul_(c).add_(A, alpha=b)
        X_new = torch.empty_like(X)
        torch.matmul(B, X, out=X_new).add_(X, alpha=a)
        X = X_new

    X = X.to(torch.float32)
    if needs_transpose:
        X = X.mT
    return X


class NewtonSchulz(OptimStep):
    """Newton-Schulz zeroth-power orthogonalization.

    Reads `ctx.grad` (fp32, 2D or 3D), writes orthogonalized tensor back to
    `ctx.grad`. Meta records 'ns_steps', 'ns_coeffs'.
    """

    def __init__(self, coefficient_type: str = "quintic", steps: int = 5,
                 fp32_matmul_prec: str = "medium"):
        if coefficient_type not in _COEFFICIENT_SETS:
            raise ValueError(f"unknown coefficient_type {coefficient_type!r}; "
                             f"valid: {sorted(_COEFFICIENT_SETS)}")
        self.coefficient_type = coefficient_type
        self.steps = steps
        self.fp32_matmul_prec = fp32_matmul_prec

    def __call__(self, ctx: StepContext) -> None:
        coeffs = _COEFFICIENT_SETS[self.coefficient_type]
        ctx.grad = _orth(ctx.grad, coeffs, self.steps, self.fp32_matmul_prec)
        ctx.meta["orth_steps"] = self.steps
        ctx.meta["orth_coeffs"] = self.coefficient_type


class PolarExpress(OptimStep):
    """Polar Express orthogonalization (arxiv 2505.16932).

    Different coefficient schedule than NS — 8 steps rotating through a
    distinct (a,b,c) triple each step. Same numerical framework as NS
    otherwise (`_orth` is shared).
    """

    def __init__(self, steps: int = 8, fp32_matmul_prec: str = "medium"):
        # Polar express requires steps to be multiple of 8 for full schedule.
        if steps % 8 != 0:
            raise ValueError(f"PolarExpress steps must be multiple of 8 (got {steps})")
        self.steps = steps
        self.fp32_matmul_prec = fp32_matmul_prec

    def __call__(self, ctx: StepContext) -> None:
        coeffs = _COEFFICIENT_SETS["polar_express"]
        ctx.grad = _orth(ctx.grad, coeffs, self.steps, self.fp32_matmul_prec)
        ctx.meta["orth_steps"] = self.steps
        ctx.meta["orth_coeffs"] = "polar_express"
