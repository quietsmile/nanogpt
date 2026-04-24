"""LR schedule unit tests — bitwise vs train.py get_lr."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanogpt.train.schedule import CosineSchedule, WSDExpSchedule, lr_for_iter


def _legacy_cosine(it, *, learning_rate, min_lr, warmup_iters, lr_decay_iters):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def _legacy_wsdexp(it, *, learning_rate, min_lr, warmup_samples,
                    constant_samples, decay_end_samples, effective_gbs):
    consumed = (it + 1) * effective_gbs
    if consumed <= warmup_samples:
        if warmup_samples == 0:
            return learning_rate
        return learning_rate * consumed / warmup_samples
    if consumed < constant_samples:
        return learning_rate
    if consumed >= decay_end_samples:
        return min_lr
    decay_range = decay_end_samples - constant_samples
    progress = (consumed - constant_samples) / decay_range
    min_lr_ratio = max(1e-8 / learning_rate, min_lr / learning_rate)
    return 0.5 ** (-progress * math.log2(min_lr_ratio)) * learning_rate


@pytest.mark.parametrize("it", [0, 1, 100, 1999, 2000, 100000, 600000, 700000])
def test_cosine_bitwise(it):
    kw = dict(learning_rate=6e-4, min_lr=6e-5, warmup_iters=2000, lr_decay_iters=600000)
    assert CosineSchedule(**kw)(it) == _legacy_cosine(it, **kw)
    assert lr_for_iter(it, lr_decay_style="cosine", **kw) == _legacy_cosine(it, **kw)


@pytest.mark.parametrize("it", [0, 1, 500, 1000, 3000, 5000, 6000, 7484, 7485, 10000])
def test_wsdexp_bitwise_moe196(it):
    # Exact tier2_moe_196 schedule parameters
    kw = dict(
        learning_rate=1.2e-3, min_lr=1.2e-4,
        warmup_samples=32000,
        constant_samples=383232,
        decay_end_samples=479040,
        effective_gbs=64,
    )
    assert WSDExpSchedule(**kw)(it) == _legacy_wsdexp(it, **kw)
    assert lr_for_iter(it, lr_decay_style="wsd-exp", **kw) == _legacy_wsdexp(it, **kw)


def test_unknown_style_raises():
    with pytest.raises(ValueError):
        lr_for_iter(0, lr_decay_style="invalid", learning_rate=1e-3, min_lr=1e-5)


def test_wsdexp_warmup0_returns_full_lr():
    s = WSDExpSchedule(learning_rate=1e-3, min_lr=0, warmup_samples=0,
                       constant_samples=1000, decay_end_samples=2000, effective_gbs=1)
    assert s(0) == 1e-3
