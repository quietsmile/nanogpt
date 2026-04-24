"""Scaling steps — compute per-param multiplier for the final update.

Each sets `ctx.update = ctx.grad` and `ctx.lr_scale` so Muon's final line:
    p.data.add_(ctx.update, alpha=-ctx.group['lr'] * ctx.lr_scale)
applies the correct magnitude.

Three modes (from megatron's get_muon_scale_factor):
  Spectral   lr_scale = rms * max(out, in)**0.5    # matches AdamW RMS in expectation
  ShapeScale lr_scale = max(1, out/in)**0.5
  UnitRMS    lr_scale = (out/in)**0.5
"""
from __future__ import annotations

from ..pipeline import OptimStep, StepContext


def _last_two_dims(p) -> tuple[int, int]:
    return p.size(-2), p.size(-1)


class SpectralScale(OptimStep):
    """lr_scale = muon_matched_adamw_rms * max(out, in)**0.5.

    Default rms=0.2 matches PAI Megatron Muon baseline.
    """

    def __init__(self, muon_matched_adamw_rms: float = 0.2):
        self.muon_matched_adamw_rms = muon_matched_adamw_rms

    def __call__(self, ctx: StepContext) -> None:
        size_out, size_in = _last_two_dims(ctx.param)
        scale = self.muon_matched_adamw_rms * max(size_out, size_in) ** 0.5
        ctx.update = ctx.grad.to(ctx.param.dtype)
        ctx.lr_scale = scale
        ctx.meta["spectral_scale"] = scale


class ShapeScale(OptimStep):
    """lr_scale = max(1, out/in)**0.5."""

    def __call__(self, ctx: StepContext) -> None:
        size_out, size_in = _last_two_dims(ctx.param)
        scale = max(1.0, size_out / size_in) ** 0.5
        ctx.update = ctx.grad.to(ctx.param.dtype)
        ctx.lr_scale = scale
        ctx.meta["shape_scale"] = scale


class UnitRMSScale(OptimStep):
    """lr_scale = (out/in)**0.5. Unit RMS target."""

    def __call__(self, ctx: StepContext) -> None:
        size_out, size_in = _last_two_dims(ctx.param)
        scale = (size_out / size_in) ** 0.5
        ctx.update = ctx.grad.to(ctx.param.dtype)
        ctx.lr_scale = scale
        ctx.meta["unit_rms_scale"] = scale
