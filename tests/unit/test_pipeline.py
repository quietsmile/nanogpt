"""Sanity tests for nanogpt.optim.pipeline — StepContext + OptimStep base."""
import pytest
import torch

from nanogpt.optim import OptimStep, StepContext


def test_context_construction():
    p = torch.zeros(2, 3)
    g = torch.ones(2, 3)
    ctx = StepContext(param=p, grad=g, state={}, group={"lr": 1e-3})
    assert ctx.param is p
    assert ctx.grad is g
    assert ctx.state == {}
    assert ctx.group["lr"] == 1e-3
    assert ctx.update is None
    assert ctx.lr_scale == 1.0
    assert ctx.meta == {}


def test_meta_defaults_are_independent():
    # default_factory dict should NOT be shared across instances
    ctx1 = StepContext(torch.zeros(1), torch.zeros(1), {}, {})
    ctx2 = StepContext(torch.zeros(1), torch.zeros(1), {}, {})
    ctx1.meta["x"] = 1
    assert ctx2.meta == {}


def test_optim_step_abstract():
    # Calling __call__ on raw OptimStep must raise
    step = OptimStep()
    ctx = StepContext(torch.zeros(1), torch.zeros(1), {}, {})
    with pytest.raises(NotImplementedError):
        step(ctx)


def test_optim_step_subclass_mutation():
    class Halve(OptimStep):
        def __call__(self, ctx):
            ctx.grad = ctx.grad * 0.5
            ctx.meta["halved"] = True

    ctx = StepContext(torch.zeros(4), torch.ones(4), {}, {})
    Halve()(ctx)
    assert torch.allclose(ctx.grad, torch.full((4,), 0.5))
    assert ctx.meta["halved"] is True


def test_optim_step_repr():
    class FooStep(OptimStep):
        def __init__(self, x=3.14, mode="a"):
            self.x = x
            self.mode = mode

        def __call__(self, ctx):
            pass

    assert repr(FooStep(x=2.0, mode="b")) == "FooStep(x=2.0, mode='b')"
