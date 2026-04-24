"""PerParamNormalize (NorMuon) — per-elem normalization by 2nd-moment EMA RMS."""
import torch

from nanogpt.optim import StepContext
from nanogpt.optim.steps import PerParamNormalize


def test_per_param_normalize_single_step():
    g = torch.tensor([[2.0, -2.0], [4.0, 0.5]])
    step = PerParamNormalize(beta2=0.95, eps=1e-8)
    state: dict = {}
    ctx = StepContext(param=torch.empty(2, 2), grad=g.clone(), state=state, group={})
    step(ctx)
    # After 1 step with beta2=0.95: sq = (1-0.95) * g^2 = 0.05 * g^2
    expected_sq = 0.05 * g.pow(2)
    assert torch.allclose(state["second_moment"], expected_sq)
    # grad output = g / (sqrt(sq) + eps)
    expected_grad = g.float() / (expected_sq.sqrt() + 1e-8)
    assert torch.allclose(ctx.grad, expected_grad)


def test_per_param_normalize_multiple_steps():
    torch.manual_seed(5)
    grads = [torch.randn(4, 4) for _ in range(3)]
    beta2 = 0.9
    step = PerParamNormalize(beta2=beta2, eps=0.0)
    state: dict = {}
    # Reference manual EMA
    sq_ref = torch.zeros(4, 4, dtype=torch.float32)
    for g in grads:
        g32 = g.float()
        sq_ref.mul_(beta2).addcmul_(g32, g32, value=1 - beta2)
        ctx = StepContext(param=torch.empty(4, 4), grad=g.clone(), state=state, group={})
        step(ctx)
    assert torch.allclose(state["second_moment"], sq_ref)


def test_per_param_normalize_state_allocated_once():
    step = PerParamNormalize()
    state: dict = {}
    for _ in range(3):
        ctx = StepContext(param=torch.empty(2, 2), grad=torch.ones(2, 2),
                          state=state, group={})
        step(ctx)
    assert "second_moment" in state
    assert state["second_moment"].shape == (2, 2)


def test_per_param_normalize_dtype_preserved():
    step = PerParamNormalize()
    ctx = StepContext(param=torch.empty(2, 2),
                      grad=torch.ones(2, 2, dtype=torch.float32),
                      state={}, group={})
    step(ctx)
    assert ctx.grad.dtype == torch.float32
