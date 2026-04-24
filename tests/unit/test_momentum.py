"""Momentum step — bitwise vs Megatron formula exp_avg ← β*exp_avg + g."""
import torch

from nanogpt.optim import StepContext
from nanogpt.optim.steps import Momentum


def _megatron_ref(grad_seq, beta=0.95, nesterov=True):
    """Reference: iterate momentum update over a sequence of grads."""
    exp_avg = torch.zeros_like(grad_seq[0], dtype=torch.float32)
    out = []
    for g in grad_seq:
        g32 = g.detach().to(torch.float32, copy=True)
        exp_avg.mul_(beta).add_(g32)
        if nesterov:
            out.append(g32.add(exp_avg, alpha=beta))
        else:
            out.append(exp_avg.clone())
    return out, exp_avg


def test_momentum_nesterov_bitwise_3step():
    torch.manual_seed(1)
    grads = [torch.randn(8, 16, dtype=torch.float32) for _ in range(3)]
    ref_outs, ref_state = _megatron_ref(grads, beta=0.95, nesterov=True)

    step = Momentum(beta=0.95, nesterov=True)
    state: dict = {}
    for i, g in enumerate(grads):
        ctx = StepContext(param=torch.empty(8, 16), grad=g.clone(), state=state, group={})
        step(ctx)
        assert torch.equal(ctx.grad, ref_outs[i])
    assert torch.equal(state["momentum_buffer"], ref_state)


def test_momentum_no_nesterov_bitwise():
    torch.manual_seed(2)
    grads = [torch.randn(4, 8, dtype=torch.float32) for _ in range(3)]
    ref_outs, _ = _megatron_ref(grads, beta=0.9, nesterov=False)

    step = Momentum(beta=0.9, nesterov=False)
    state: dict = {}
    for i, g in enumerate(grads):
        ctx = StepContext(param=torch.empty(4, 8), grad=g.clone(), state=state, group={})
        step(ctx)
        assert torch.equal(ctx.grad, ref_outs[i])


def test_momentum_allocates_buffer_on_first_call():
    step = Momentum()
    state: dict = {}
    assert "momentum_buffer" not in state
    g = torch.randn(2, 3)
    ctx = StepContext(param=torch.empty(2, 3), grad=g, state=state, group={})
    step(ctx)
    assert state["momentum_buffer"].shape == g.shape
    assert state["momentum_buffer"].dtype == torch.float32


def test_momentum_records_meta():
    step = Momentum()
    ctx = StepContext(param=torch.empty(4, 4), grad=torch.ones(4, 4), state={}, group={})
    step(ctx)
    assert "momentum_buffer_norm" in ctx.meta
    assert ctx.meta["momentum_buffer_norm"] > 0
