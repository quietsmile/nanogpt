"""
Muon optimizer (NorMuon variant with Polar Express orthogonalization and Cautious WD).

Algorithmic port of modded-nanogpt's NorMuonAndAdam (KellerJordan/modded-nanogpt
train_gpt.py, latest record). The distributed comms / param banking / Triton kernels
are intentionally dropped — only the algorithm is moved here, wrapped as a standard
torch.optim.Optimizer subclass and DDP-compatible (relies on default DDP all-reduce).

References:
  Muon                — https://kellerjordan.github.io/posts/muon/
  Polar Express       — https://arxiv.org/abs/2505.16932 (Amsel et al., 2025)
  NorMuon             — https://arxiv.org/abs/2510.05491
  Cautious optimizers — https://arxiv.org/abs/2411.16085

Use Muon for 2D matrix params (Linear weights) and 3D batched matrix params
(MoE expert weights of shape [E, in, out]). Use AdamW (or CautiousAdamW) for
non-matrix params: embeddings, lm_head, RMSNorm, biases, MoE router gate.
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


# Polar Express coefficients (5 iterations, safety_factor=2e-2, cushion=2).
# Verbatim from modded-nanogpt train_gpt.py polar_express_coeffs.
POLAR_EXPRESS_COEFFS: tuple[tuple[float, float, float], ...] = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


def polar_express_orthogonalize(X: torch.Tensor) -> torch.Tensor:
    """
    Polar Express orthogonalization. Operates on the last two dims; supports
    2D (m, n) and batched 3D (B, m, n). Runs in bf16 internally for speed and
    is self-correcting at that precision.

    Returns U with U U^T ≈ I (wide) or U^T U ≈ I (tall), same shape as X.
    """
    X = X.to(torch.bfloat16)
    is_tall = X.size(-2) > X.size(-1)

    # Frobenius-normalize so the spectral norm is at most 1 (cushion 1+2e-2).
    norm = X.norm(dim=(-2, -1), keepdim=True)
    X = X / (norm * (1 + 2e-2) + 1e-6)

    for a, b, c in POLAR_EXPRESS_COEFFS:
        if is_tall:
            A = X.transpose(-2, -1) @ X            # [..., n, n]
            B = b * A + c * (A @ A)                # [..., n, n]
            X = a * X + X @ B                      # [..., m, n]
        else:
            A = X @ X.transpose(-2, -1)            # [..., m, m]
            B = b * A + c * (A @ A)                # [..., m, m]
            X = a * X + B @ X                      # [..., m, n]
    return X


class Muon(Optimizer):
    """
    NorMuon (= Muon + Adafactor-style low-rank variance + Cautious WD), with Polar
    Express in place of Newton-Schulz.

    Per param-group hyperparameters:
        lr           — base learning rate (typically 10–60× the AdamW LR)
        momentum     — Nesterov momentum coefficient (default 0.95)
        beta2        — EMA coeff for the row/col second moment (default 0.95)
        weight_decay — decoupled WD coefficient, gated by sign-agreement (default 0.0)

    Restrictions:
        Each param must be 2D (m, n) or 3D batched (B, m, n). Non-matrix params
        should be routed to AdamW separately (see configure_optimizers in model.py).
    """

    def __init__(
        self,
        params,
        lr: float = 0.05,
        momentum: float = 0.95,
        beta2: float = 0.95,
        weight_decay: float = 0.0,
    ):
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
        defaults = dict(lr=lr, momentum=momentum, beta2=beta2, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            beta2 = group["beta2"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.dim() not in (2, 3):
                    raise ValueError(
                        f"Muon requires 2D or 3D params; got shape {tuple(p.shape)}. "
                        "Route 1D / embedding / scalar params to AdamW."
                    )

                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                    # NorMuon second moment: reduce along the smaller of the last two dims.
                    if p.size(-2) >= p.size(-1):
                        sm_shape = p.shape[:-1] + (1,)
                    else:
                        sm_shape = p.shape[:-2] + (1, p.size(-1))
                    state["second_moment"] = torch.zeros(
                        sm_shape, dtype=torch.float32, device=p.device
                    )

                state["step"] += 1

                # Nesterov momentum (FP32). g_fp32 is always a private copy.
                g_fp32 = p.grad.detach().to(torch.float32, copy=True)
                buf = state["momentum_buffer"]
                buf.lerp_(g_fp32, 1 - momentum)         # buf ← momentum·buf + (1-momentum)·g
                g_eff = (1 - momentum) * g_fp32 + momentum * buf  # Nesterov look-ahead

                # Polar Express orthogonalization (BF16).
                v = polar_express_orthogonalize(g_eff)

                # NorMuon variance reduction (rescales rows/cols by Adafactor-like EMA
                # while preserving the overall Frobenius norm).
                v_fp32 = v.float()
                red_dim = -1 if p.size(-2) >= p.size(-1) else -2
                v_mean_sq = v_fp32.square().mean(dim=red_dim, keepdim=True)
                red_dim_size = v_fp32.size(red_dim)
                v_norm = (v_mean_sq.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()

                sm = state["second_moment"]
                sm.lerp_(v_mean_sq, 1 - beta2)
                step_scale = sm.clamp_min(1e-10).rsqrt()
                scaled_sq_sum = (v_mean_sq * red_dim_size) * step_scale.square()
                v_norm_new = (
                    scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt().clamp_min_(1e-10)
                )
                final_scale = step_scale * (v_norm / v_norm_new)
                v_fp32 = v_fp32 * final_scale

                # Cautious decoupled weight decay: only decay where the proposed update
                # direction agrees in sign with the current parameter (i.e. where WD's
                # pull-toward-zero is consistent with the gradient direction).
                p_data = p.data
                if wd > 0:
                    p_fp32 = p_data.float()
                    mask = ((v_fp32 * p_fp32) >= 0).to(p_fp32.dtype)
                    p_data.sub_((p_fp32 * mask).to(p_data.dtype), alpha=lr * wd)

                # Step.
                p_data.sub_(v_fp32.to(p_data.dtype), alpha=lr)

        return loss


class MultiOptimizer:
    """
    Holds multiple torch.optim.Optimizer instances behind a single interface.
    state_dict round-trips through a nested dict keyed by the optimizer name.

    Exposes the subset of the Optimizer surface that train.py and torch.cuda.amp
    GradScaler (when enabled=False) actually use: param_groups, state, step,
    zero_grad, state_dict, load_state_dict.
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
