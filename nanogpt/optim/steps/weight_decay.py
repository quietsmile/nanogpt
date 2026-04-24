"""Weight decay steps — three flavors.

DecoupledWD    p ← p * (1 - lr*wd)             applied BEFORE momentum (AdamW style)
CoupledWD      grad ← grad + wd*p              folded into grad (SGD-with-WD style)
CautiousWD     p ← p - lr*wd*p IF sign(grad)==sign(update)   (arxiv 2411.16085)

Read group['weight_decay'] and group['lr'] from ctx.group.
"""
from __future__ import annotations

from ..pipeline import OptimStep, StepContext


class DecoupledWD(OptimStep):
    """p *= (1 - lr*wd). Applied BEFORE momentum accumulation."""

    def __call__(self, ctx: StepContext) -> None:
        wd = ctx.group.get("weight_decay", 0.0)
        if wd > 0.0:
            lr = ctx.group["lr"]
            ctx.param.data.mul_(1.0 - lr * wd)


class CoupledWD(OptimStep):
    """grad += wd * p. Folds weight decay into gradient pre-momentum."""

    def __call__(self, ctx: StepContext) -> None:
        wd = ctx.group.get("weight_decay", 0.0)
        if wd > 0.0:
            ctx.grad = ctx.grad + wd * ctx.param.data


class CautiousWD(OptimStep):
    """DecoupledWD gated by sign(grad) == sign(final update).

    Only applied once ctx.update is populated (i.e. after SpectralScale).
    Must be placed AFTER orthogonalization + scaling for the check to make
    sense. See arxiv 2411.16085.
    """

    def __call__(self, ctx: StepContext) -> None:
        wd = ctx.group.get("weight_decay", 0.0)
        if wd <= 0.0:
            return
        if ctx.update is None:
            raise RuntimeError("CautiousWD requires ctx.update to be set; "
                               "place it after SpectralScale / orth steps.")
        # Check element-wise sign match between raw grad and final update.
        raw_grad = ctx.state.get("_cautious_raw_grad")
        if raw_grad is None:
            # Fallback — use current ctx.grad. Callers should cache raw grad earlier
            # via a no-op step if sign-matching against pre-orth grad is desired.
            raw_grad = ctx.grad
        mask = (raw_grad.sign() == ctx.update.sign()).to(ctx.param.dtype)
        lr = ctx.group["lr"]
        ctx.param.data.mul_(1.0 - lr * wd * mask)
