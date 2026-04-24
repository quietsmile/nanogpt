"""Optimizer pipeline — compose Muon/AdamW from small step classes.

The Muon algorithm is expressed as a list of OptimStep instances (a "recipe"):
    recipe = [DecoupledWD(), Momentum(beta=0.95, nesterov=True),
              NewtonSchulz(coefs='quintic', steps=5),
              SpectralScale(rms=0.2)]
    opt = Muon(params, pipeline=recipe, lr=1.2e-3)

Swapping algorithm parts = edit the recipe, no new optimizer class needed.
Each step is unit-testable in isolation.
"""
from .pipeline import OptimStep, StepContext
__all__ = ["OptimStep", "StepContext"]
