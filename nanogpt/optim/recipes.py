"""Muon recipes — pre-configured pipelines for known algorithms.

A "recipe" is just a list of OptimStep instances. Feed to Muon(..., pipeline=recipe).

Built-in recipes:
  muon_megatron   PAI Megatron Muon (quintic NS, spectral scale rms=0.2, Nesterov, decoupled WD)
                  validated bitwise-equivalent to legacy muon_megatron.Muon
  muon_normuon    modded-nanogpt NorMuon (Polar Express, per-param normalize, non-Nesterov,
                  cautious WD)

Custom recipe example:
    my_muon = [DecoupledWD(), Momentum(beta=0.95, nesterov=False),
               NewtonSchulz(coefs='quintic_new', steps=5), ShapeScale()]
    opt = Muon(params, pipeline=my_muon, lr=3e-4)
"""
from __future__ import annotations

from .steps import (
    CautiousWD,
    DecoupledWD,
    Momentum,
    NewtonSchulz,
    PerParamNormalize,
    PolarExpress,
    SpectralScale,
)


def muon_megatron(
    momentum_beta: float = 0.95,
    use_nesterov: bool = True,
    coefficient_type: str = "quintic",
    num_ns_steps: int = 5,
    muon_matched_adamw_rms: float = 0.2,
    fp32_matmul_prec: str = "medium",
) -> list:
    """PAI Megatron Muon recipe (matches cybertron/base_config defaults).

    Order:
        DecoupledWD → Momentum(Nesterov) → NewtonSchulz(quintic) → SpectralScale
    """
    return [
        DecoupledWD(),
        Momentum(beta=momentum_beta, nesterov=use_nesterov),
        NewtonSchulz(
            coefficient_type=coefficient_type,
            steps=num_ns_steps,
            fp32_matmul_prec=fp32_matmul_prec,
        ),
        SpectralScale(muon_matched_adamw_rms=muon_matched_adamw_rms),
    ]


def muon_normuon(
    momentum_beta: float = 0.95,
    beta2: float = 0.95,
    polar_express_steps: int = 8,
    muon_matched_adamw_rms: float = 0.2,
    fp32_matmul_prec: str = "medium",
) -> list:
    """modded-nanogpt NorMuon + Polar Express + Cautious WD recipe.

    Order:
        Momentum(non-Nesterov) → PerParamNormalize → PolarExpress → SpectralScale
                                                                   → CautiousWD (post-scale)
    """
    return [
        Momentum(beta=momentum_beta, nesterov=False),
        PerParamNormalize(beta2=beta2),
        PolarExpress(steps=polar_express_steps, fp32_matmul_prec=fp32_matmul_prec),
        SpectralScale(muon_matched_adamw_rms=muon_matched_adamw_rms),
        CautiousWD(),
    ]
