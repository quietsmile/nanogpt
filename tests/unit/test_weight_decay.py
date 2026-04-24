"""Weight decay variants — decoupled / coupled / cautious."""
import pytest
import torch

from nanogpt.optim import StepContext
from nanogpt.optim.steps import CautiousWD, CoupledWD, DecoupledWD


def test_decoupled_wd_scales_param():
    p = torch.nn.Parameter(torch.full((4, 4), 1.0))
    ctx = StepContext(param=p, grad=torch.zeros(4, 4), state={},
                      group={"lr": 0.1, "weight_decay": 0.5})
    DecoupledWD()(ctx)
    # p *= (1 - 0.1 * 0.5) = 0.95
    assert torch.equal(p.data, torch.full((4, 4), 0.95))


def test_decoupled_wd_zero_wd_noop():
    p = torch.nn.Parameter(torch.full((4, 4), 1.0))
    ctx = StepContext(param=p, grad=torch.zeros(4, 4), state={},
                      group={"lr": 0.1, "weight_decay": 0.0})
    DecoupledWD()(ctx)
    assert torch.equal(p.data, torch.full((4, 4), 1.0))


def test_coupled_wd_folds_into_grad():
    p = torch.nn.Parameter(torch.full((4, 4), 2.0))
    g = torch.full((4, 4), 0.5)
    ctx = StepContext(param=p, grad=g, state={}, group={"weight_decay": 0.1})
    CoupledWD()(ctx)
    # grad += 0.1 * 2.0 = 0.2, so grad = 0.5 + 0.2 = 0.7
    assert torch.equal(ctx.grad, torch.full((4, 4), 0.7))
    # Param unchanged
    assert torch.equal(p.data, torch.full((4, 4), 2.0))


def test_coupled_wd_preserves_dtype():
    p = torch.nn.Parameter(torch.ones(4, 4, dtype=torch.float32))
    g = torch.zeros(4, 4, dtype=torch.float32)
    ctx = StepContext(param=p, grad=g, state={}, group={"weight_decay": 1.0})
    CoupledWD()(ctx)
    assert ctx.grad.dtype == torch.float32


def test_cautious_wd_requires_update_set():
    p = torch.nn.Parameter(torch.ones(2, 2))
    ctx = StepContext(param=p, grad=torch.ones(2, 2), state={},
                      group={"lr": 0.1, "weight_decay": 0.1})
    with pytest.raises(RuntimeError):
        CautiousWD()(ctx)


def test_cautious_wd_masks_by_sign_match():
    p = torch.nn.Parameter(torch.ones(4))
    # sign(grad) = [+, +, -, -]   sign(update) = [+, -, +, -]
    # mask:                         [1, 0, 0, 1]
    grad = torch.tensor([0.3, 0.5, -0.2, -0.1])
    update = torch.tensor([0.1, -0.1, 0.1, -0.2])
    ctx = StepContext(param=p, grad=grad, state={}, group={"lr": 0.1, "weight_decay": 0.5})
    ctx.update = update
    CautiousWD()(ctx)
    # elements 0,3: p *= (1 - 0.1*0.5) = 0.95
    # elements 1,2: p *= 1 (unmasked, wd skipped)
    expected = torch.tensor([0.95, 1.0, 1.0, 0.95])
    assert torch.allclose(p.data, expected)
