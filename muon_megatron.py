"""
MUON MEGATRON PORT — coexists with muon.py (NorMuon variant).
Switch via config flag muon_impl='megatron' | 'normuon'.

Muon optimizer — faithful port of megatron-LM emerging_optimizers Muon
(megatron.core.emerging_optimizers.orthogonalized_optimizers).

Source: /newcpfs/user/yuchen/llm/megatron_dots3.0_swa/megatron/core/
        emerging_optimizers/orthogonalized_optimizers/{muon,muon_utils,
        orthogonalized_optimizer}.py

Aligned with cybertron's defaults (cybertron config/base_config.py):
  coefficient_type='quintic', num_ns_steps=5, scale_mode='spectral',
  muon_matched_adamw_rms=0.2, momentum_beta=0.95, decoupled WD,
  bf16 matmul (fp32_matmul_prec='medium').

Tensor-parallel and per-head split logic from megatron is dropped (this
codebase has separate q/k/v projections and DP-only setup); 3D batched
MoE expert weights are added (megatron's Muon is 2D-only).
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch.optim.optimizer import Optimizer


# Newton-Schulz coefficient sets — verbatim from megatron muon_utils.py.
# Each entry is a list of (a, b, c) triples; one triple is consumed per NS step
# (cycling if num_ns_steps > len(set)).
_COEFFICIENT_SETS: dict[str, list[tuple[float, float, float]]] = {
    "simple": [
        (3.4445, -4.7750, 2.0315),
    ],
    "quintic": [
        # Single triple, repeated num_ns_steps times. Cybertron's default.
        (3.4445, -4.7750, 2.0315),
    ],
    "polar_express": [
        # Polar Express iteration from arxiv 2505.16932 (8 steps).
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


def newton_schulz(
    x: torch.Tensor,
    steps: int = 5,
    coefficient_type: str = "quintic",
    fp32_matmul_prec: str = "medium",
) -> torch.Tensor:
    """Newton-Schulz orthogonalization (zeroth-power approximation).

    Operates on the last two dims; supports 2D (m, n) and batched 3D (B, m, n).
    Always whitens along the smaller dimension (transpose if tall).

    Differences from megatron:
      - 3D batched supported (megatron is 2D only).
      - tp_group / tp_mode dropped (DP-only here).
    """
    if x.ndim < 2:
        raise ValueError(
            f"newton_schulz requires at least 2 dims, got {x.shape}"
        )
    if x.dtype != torch.float32:
        raise ValueError(f"newton_schulz expects fp32 input, got {x.dtype}")

    # Whiten on the smaller dim.
    needs_transpose = x.size(-2) > x.size(-1)
    if needs_transpose:
        x = x.mT

    # Spectral norm cap: ||X||_F <= 1 (per matrix in batch).
    X = torch.nn.functional.normalize(x, p=2, dim=(-2, -1), eps=1e-7)

    if coefficient_type not in _COEFFICIENT_SETS:
        raise ValueError(
            f"unknown coefficient_type {coefficient_type!r}; "
            f"valid: {sorted(_COEFFICIENT_SETS)}"
        )
    coefficient_sets = _COEFFICIENT_SETS[coefficient_type]
    if steps % len(coefficient_sets) != 0:
        raise ValueError(
            f"steps ({steps}) must be a multiple of len(coefficient_sets) "
            f"({len(coefficient_sets)}) for {coefficient_type!r}"
        )

    if fp32_matmul_prec == "medium":
        # Mirrors megatron: explicitly cast to bf16 for the matmul body, since
        # PyTorch lacks an fp32-IO bf16-compute kernel for "medium" precision.
        X = X.to(torch.bfloat16)

    for i in range(steps):
        a, b, c = coefficient_sets[i % len(coefficient_sets)]
        A = X @ X.mT
        # B = b*A + c*(A@A); X = a*X + B@X
        B = torch.empty_like(A)
        torch.matmul(A, A, out=B).mul_(c).add_(A, alpha=b)
        # X_new = a*X + B@X (cast to higher rank for batched if needed)
        X_new = torch.empty_like(X)
        torch.matmul(B, X, out=X_new).add_(X, alpha=a)
        X = X_new

    X = X.to(torch.float32)
    if needs_transpose:
        X = X.mT
    return X


def get_muon_scale_factor(
    size_out: int,
    size_in: int,
    mode: str = "spectral",
    muon_matched_adamw_rms: float = 0.2,
) -> float:
    """Per-update scale, mirrors megatron get_muon_scale_factor.

    spectral (default): muon_matched_adamw_rms * max(out, in)**0.5 — the value
                        that makes Muon's update RMS match AdamW's in expectation.
                        Lets you transfer AdamW LR with a small constant multiplier.
    shape_scaling:      max(1, out/in)**0.5
    unit_rms_norm:      (out/in)**0.5
    """
    if mode == "spectral":
        return muon_matched_adamw_rms * max(size_out, size_in) ** 0.5
    if mode == "shape_scaling":
        return max(1.0, size_out / size_in) ** 0.5
    if mode == "unit_rms_norm":
        return (size_out / size_in) ** 0.5
    raise ValueError(f"unknown scale mode {mode!r}")


class Muon(Optimizer):
    """
    Muon = SGD-momentum + Newton-Schulz orthogonalization + spectral scaling.

    Faithful to megatron's algorithm (orthogonalized_optimizer.py:step()):
      grad ← p.grad
      if wd > 0 and decoupled:
          p ← p - lr * wd * p                         (decoupled WD, BEFORE momentum)
      else:
          grad ← grad + wd * p                        (coupled)
      exp_avg ← momentum_beta * exp_avg + grad        (NB: no (1-β) factor)
      if nesterov:
          grad ← grad + momentum_beta * exp_avg
      else:
          grad ← exp_avg
      grad_ortho ← newton_schulz(grad)                (in fp32 -> bf16 -> fp32)
      scale ← spectral_scale(out, in)
      p ← p - lr * scale * grad_ortho

    Per param-group hyperparameters:
        lr, momentum_beta, use_nesterov, weight_decay, use_decoupled_weight_decay,
        coefficient_type, num_ns_steps, scale_mode, muon_matched_adamw_rms,
        fp32_matmul_prec.

    Restrictions:
        Each param must be 2D (m, n) or 3D batched (B, m, n). 1D / scalar
        params should be routed to AdamW (see configure_optimizers).
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 3e-4,
        momentum_beta: float = 0.95,
        use_nesterov: bool = True,
        weight_decay: float = 0.1,
        use_decoupled_weight_decay: bool = True,
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        muon_matched_adamw_rms: float = 0.2,
        fp32_matmul_prec: str = "medium",
    ):
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be >= 1, got {num_ns_steps}")
        if coefficient_type not in _COEFFICIENT_SETS:
            raise ValueError(f"unknown coefficient_type {coefficient_type!r}")
        defaults = dict(
            lr=lr,
            momentum_beta=momentum_beta,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            use_decoupled_weight_decay=use_decoupled_weight_decay,
            coefficient_type=coefficient_type,
            num_ns_steps=num_ns_steps,
            scale_mode=scale_mode,
            muon_matched_adamw_rms=muon_matched_adamw_rms,
            fp32_matmul_prec=fp32_matmul_prec,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum_beta = group["momentum_beta"]
            use_nesterov = group["use_nesterov"]
            wd = group["weight_decay"]
            use_decoupled = group["use_decoupled_weight_decay"]
            coeff_type = group["coefficient_type"]
            ns_steps = group["num_ns_steps"]
            scale_mode = group["scale_mode"]
            matched_rms = group["muon_matched_adamw_rms"]
            mm_prec = group["fp32_matmul_prec"]

            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                if p.dim() not in (2, 3):
                    raise ValueError(
                        f"Muon requires 2D or 3D params; got {tuple(p.shape)}. "
                        "Route 1D / scalar params to AdamW."
                    )

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad, dtype=torch.float32)
                exp_avg = state["momentum_buffer"]

                # Decoupled WD: applied directly to param BEFORE momentum.
                # Coupled WD: folded into grad.
                if wd > 0.0:
                    if use_decoupled:
                        p.data.mul_(1.0 - lr * wd)
                    else:
                        grad = grad + wd * p.data

                # Momentum: NB megatron's formula has no (1-β); exp_avg grows linearly with grad
                # until balanced by the orthogonalization step.
                grad_fp32 = grad.detach().to(torch.float32, copy=True)
                exp_avg.mul_(momentum_beta).add_(grad_fp32)

                if use_nesterov:
                    g_eff = grad_fp32.add(exp_avg, alpha=momentum_beta)
                else:
                    g_eff = exp_avg

                # Newton-Schulz orthogonalization (fp32 in, bf16 inside, fp32 out).
                g_ortho = newton_schulz(
                    g_eff,
                    steps=ns_steps,
                    coefficient_type=coeff_type,
                    fp32_matmul_prec=mm_prec,
                )

                # Spectral scale per-matrix. For 3D [B,m,n], use last two dims.
                size_out = p.size(-2)
                size_in = p.size(-1)
                scale = get_muon_scale_factor(
                    size_out, size_in,
                    mode=scale_mode,
                    muon_matched_adamw_rms=matched_rms,
                )

                p.data.add_(g_ortho.to(p.dtype), alpha=-lr * scale)

        return loss


class MultiOptimizer:
    """
    Holds multiple torch.optim.Optimizer instances behind a single interface.
    state_dict round-trips through a nested dict keyed by the optimizer name.
    """

    def __init__(self, optimizers: dict[str, Optimizer]):
        for name, opt in optimizers.items():
            if not isinstance(opt, Optimizer):
                raise TypeError(f"optimizer '{name}' is not a torch.optim.Optimizer")
        self.inner: dict[str, Optimizer] = dict(optimizers)

    @property
    def param_groups(self):
        return [g for opt in self.inner.values() for g in opt.param_groups]

    @property
    def state(self):
        merged: dict = {}
        for opt in self.inner.values():
            merged.update(opt.state)
        return merged

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for opt in self.inner.values():
            opt.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.inner.values():
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {"_multi": True, **{name: opt.state_dict() for name, opt in self.inner.items()}}

    def load_state_dict(self, state_dict: dict):
        if not state_dict.get("_multi"):
            raise ValueError(
                "load_state_dict expected a MultiOptimizer-format dict; "
                f"got keys {list(state_dict.keys())}. "
                "Likely loading a baseline (single-optimizer) checkpoint into a Muon run."
            )
        for name, opt in self.inner.items():
            if name not in state_dict:
                raise KeyError(f"missing optimizer '{name}'; have {list(state_dict.keys())}")
            opt.load_state_dict(state_dict[name])
