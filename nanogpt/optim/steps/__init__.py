"""Muon algorithm steps — each a stand-alone, unit-testable OptimStep.

Recipes compose these into Muon variants (see nanogpt.optim.recipes).
"""
from .momentum import Momentum
from .normalize import PerParamNormalize
from .orthogonalize import NewtonSchulz, PolarExpress
from .scale import ShapeScale, SpectralScale, UnitRMSScale
from .weight_decay import CautiousWD, CoupledWD, DecoupledWD

__all__ = [
    "Momentum",
    "NewtonSchulz",
    "PolarExpress",
    "PerParamNormalize",
    "ShapeScale",
    "SpectralScale",
    "UnitRMSScale",
    "CautiousWD",
    "CoupledWD",
    "DecoupledWD",
]
