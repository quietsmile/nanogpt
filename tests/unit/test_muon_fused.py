"""Fused-Muon path: orthogonalizing concat([Q, K, V]) ≡ orthogonalizing the
fused matrix once, NOT three independent orthogonalizations.

This is the fix for the nano-vs-Megatron Muon scratch regression: nano splits
attn into q_proj/k_proj/v_proj (3 separate Linears) while Megatron uses one
fused linear_qkv. Muon's NS5 must see the fused matrix to produce the same
update geometry as Megatron.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanogpt.optim.muon import Muon
from nanogpt.optim.recipes import muon_megatron


def _make_pipeline():
    return muon_megatron(
        momentum_beta=0.95,
        num_ns_steps=5,
        muon_matched_adamw_rms=0.2,
    )


def test_fused_step_matches_concat_param_step():
    """Two ways to handle Q/K/V should produce identical updates:
    (A) one big linear_qkv [out_q+out_k+out_v, in] param + standalone Muon
    (B) three split params (q,k,v) + Muon with fused_param_lists=[[q,k,v]]

    After one Muon step with the same gradients, weights must match bitwise.
    """
    torch.manual_seed(7)
    # Production-like: in_dim >= total_out_dim so SpectralScale's max(out, in) = in_dim
    # for both fused (12,12) and per-split (4,12)+(4,12)+(4,12).
    in_dim, oq, ok, ov = 12, 4, 4, 4
    # path A: single fused param
    qkv = torch.randn(oq + ok + ov, in_dim, dtype=torch.float32)
    qkv_ref = qkv.clone().requires_grad_(True)
    qkv_grad = torch.randn_like(qkv)
    qkv_ref.grad = qkv_grad.clone()
    opt_a = Muon([qkv_ref], pipeline=_make_pipeline(), lr=1e-3)
    opt_a.step()

    # path B: three split params, Muon told to fuse them
    q = qkv[:oq].clone().requires_grad_(True)
    k = qkv[oq:oq + ok].clone().requires_grad_(True)
    v = qkv[oq + ok:].clone().requires_grad_(True)
    q.grad = qkv_grad[:oq].clone()
    k.grad = qkv_grad[oq:oq + ok].clone()
    v.grad = qkv_grad[oq + ok:].clone()
    opt_b = Muon([q, k, v], pipeline=_make_pipeline(), lr=1e-3,
                 fused_param_lists=[[q, k, v]])
    opt_b.step()

    fused_after = torch.cat([q.detach(), k.detach(), v.detach()], dim=0)
    assert torch.allclose(fused_after, qkv_ref.detach(), atol=1e-7, rtol=0), \
        f"fused path diverged: max|Δ|={(fused_after - qkv_ref.detach()).abs().max():.2e}"


def test_fused_step_differs_from_split_per_param_step():
    """Sanity: the fused fix actually CHANGES behavior vs naive split (otherwise
    there's nothing to fix). Same params + same grads, running standalone Muon
    on each split param should NOT match the fused-Muon outcome.
    """
    torch.manual_seed(11)
    # Production-like: in_dim >= total_out_dim so SpectralScale's max(out, in) = in_dim
    # for both fused (12,12) and per-split (4,12)+(4,12)+(4,12).
    in_dim, oq, ok, ov = 12, 4, 4, 4
    qkv_init = torch.randn(oq + ok + ov, in_dim, dtype=torch.float32)
    qkv_grad = torch.randn(oq + ok + ov, in_dim)

    # split + fused (correct)
    q1 = qkv_init[:oq].clone().requires_grad_(True)
    k1 = qkv_init[oq:oq + ok].clone().requires_grad_(True)
    v1 = qkv_init[oq + ok:].clone().requires_grad_(True)
    q1.grad = qkv_grad[:oq].clone()
    k1.grad = qkv_grad[oq:oq + ok].clone()
    v1.grad = qkv_grad[oq + ok:].clone()
    opt1 = Muon([q1, k1, v1], pipeline=_make_pipeline(), lr=1e-3,
                fused_param_lists=[[q1, k1, v1]])
    opt1.step()

    # split + per-param (the broken nano default)
    q2 = qkv_init[:oq].clone().requires_grad_(True)
    k2 = qkv_init[oq:oq + ok].clone().requires_grad_(True)
    v2 = qkv_init[oq + ok:].clone().requires_grad_(True)
    q2.grad = qkv_grad[:oq].clone()
    k2.grad = qkv_grad[oq:oq + ok].clone()
    v2.grad = qkv_grad[oq + ok:].clone()
    opt2 = Muon([q2, k2, v2], pipeline=_make_pipeline(), lr=1e-3)
    opt2.step()

    fused_a = torch.cat([q1.detach(), k1.detach(), v1.detach()], dim=0)
    fused_b = torch.cat([q2.detach(), k2.detach(), v2.detach()], dim=0)
    diff = (fused_a - fused_b).abs().max().item()
    assert diff > 1e-4, (
        f"fused vs per-param split paths produced ~identical updates "
        f"(max|Δ|={diff:.2e}); fix has no effect"
    )


def test_fused_path_handles_per_param_momentum():
    """Multi-step run: each param's momentum buffer must be its own slice,
    verified by checking that splitting a fused param keeps state aligned."""
    torch.manual_seed(13)
    # Production-like: in_dim >= total_out_dim so SpectralScale's max(out, in) = in_dim
    # for both fused (12,12) and per-split (4,12)+(4,12)+(4,12).
    in_dim, oq, ok, ov = 12, 4, 4, 4
    qkv = torch.randn(oq + ok + ov, in_dim, dtype=torch.float32)

    # Run 3 muon steps with random grads each step on both paths.
    qkv_a = qkv.clone().requires_grad_(True)
    opt_a = Muon([qkv_a], pipeline=_make_pipeline(), lr=1e-3)

    q = qkv[:oq].clone().requires_grad_(True)
    k = qkv[oq:oq + ok].clone().requires_grad_(True)
    v = qkv[oq + ok:].clone().requires_grad_(True)
    opt_b = Muon([q, k, v], pipeline=_make_pipeline(), lr=1e-3,
                 fused_param_lists=[[q, k, v]])

    torch.manual_seed(99)
    for _ in range(3):
        g = torch.randn_like(qkv)
        qkv_a.grad = g.clone()
        q.grad = g[:oq].clone()
        k.grad = g[oq:oq + ok].clone()
        v.grad = g[oq + ok:].clone()
        opt_a.step()
        opt_b.step()

    fused_b = torch.cat([q.detach(), k.detach(), v.detach()], dim=0)
    err = (fused_b - qkv_a.detach()).abs().max().item()
    assert err < 1e-6, f"3-step trajectory drift: max|Δ|={err:.2e}"


def test_fused_step_3d_expert_weights():
    """Expert MLP gate_weight/up_weight have shape [E, hidden, in].
    Fuse along dim 1 (the hidden axis), batch dim E preserved.
    """
    torch.manual_seed(17)
    E, h_g, h_u, in_dim = 3, 4, 4, 12
    g_init = torch.randn(E, h_g, in_dim)
    u_init = torch.randn(E, h_u, in_dim)
    g_grad = torch.randn(E, h_g, in_dim)
    u_grad = torch.randn(E, h_u, in_dim)

    # Path A: single fused param of shape [E, h_g+h_u, in_dim]
    fused_init = torch.cat([g_init, u_init], dim=1).clone().requires_grad_(True)
    fused_init.grad = torch.cat([g_grad, u_grad], dim=1).clone()
    opt_a = Muon([fused_init], pipeline=_make_pipeline(), lr=1e-3)
    opt_a.step()

    # Path B: split params, fused via fused_param_lists
    g = g_init.clone().requires_grad_(True); g.grad = g_grad.clone()
    u = u_init.clone().requires_grad_(True); u.grad = u_grad.clone()
    opt_b = Muon([g, u], pipeline=_make_pipeline(), lr=1e-3,
                 fused_param_lists=[[g, u]])
    opt_b.step()

    fused_after = torch.cat([g.detach(), u.detach()], dim=1)
    err = (fused_after - fused_init.detach()).abs().max().item()
    assert err < 1e-6, f"3D fused path drift: max|Δ|={err:.2e}"


def test_per_head_split_matches_chunkwise_ns5():
    """per_head_split=True splits Q [n_heads*head_dim, in_dim] along dim 0
    into n_heads chunks of [head_dim, in_dim] and runs NS5 + scale on each.
    Should equal manually concatenating the per-chunk results.
    """
    torch.manual_seed(31)
    n_heads, head_dim, in_dim = 4, 16, 64
    out_dim = n_heads * head_dim
    q_init = torch.randn(out_dim, in_dim, dtype=torch.float32)
    q_grad = torch.randn(out_dim, in_dim)

    # Path A: Muon with per_head_split
    q1 = q_init.clone().requires_grad_(True); q1.grad = q_grad.clone()
    opt_a = Muon([q1], pipeline=_make_pipeline(), lr=1e-3,
                 per_head_split={id(q1): head_dim})
    opt_a.step()

    # Path B: manually run Muon per head
    head_results = []
    for h in range(n_heads):
        sl = slice(h * head_dim, (h + 1) * head_dim)
        chunk = q_init[sl].clone().requires_grad_(True)
        chunk.grad = q_grad[sl].clone()
        opt_h = Muon([chunk], pipeline=_make_pipeline(), lr=1e-3)
        opt_h.step()
        head_results.append(chunk.detach())
    expected = torch.cat(head_results, dim=0)

    err = (q1.detach() - expected).abs().max().item()
    assert err < 1e-7, f"per_head trajectory drift: max|Δ|={err:.2e}"


def test_per_head_differs_from_full_tensor_ns5():
    """Sanity that per-head differs from running NS5 on the whole [out, in]."""
    torch.manual_seed(37)
    n_heads, head_dim, in_dim = 4, 16, 64
    out_dim = n_heads * head_dim
    q_init = torch.randn(out_dim, in_dim, dtype=torch.float32)
    q_grad = torch.randn(out_dim, in_dim)

    # per-head
    q1 = q_init.clone().requires_grad_(True); q1.grad = q_grad.clone()
    opt_a = Muon([q1], pipeline=_make_pipeline(), lr=1e-3,
                 per_head_split={id(q1): head_dim})
    opt_a.step()

    # full-tensor (default)
    q2 = q_init.clone().requires_grad_(True); q2.grad = q_grad.clone()
    opt_b = Muon([q2], pipeline=_make_pipeline(), lr=1e-3)
    opt_b.step()

    diff = (q1.detach() - q2.detach()).abs().max().item()
    assert diff > 1e-4, f"per-head vs full-tensor produced ~identical updates (max|Δ|={diff:.2e})"


def test_no_ortho_in_pipeline_falls_back_per_param():
    """If pipeline has no ortho step, fused list silently degrades to per-param."""
    from nanogpt.optim.steps.momentum import Momentum
    pipeline = [Momentum(beta=0.9)]
    p1 = torch.randn(4, 8, requires_grad=True)
    p2 = torch.randn(4, 8, requires_grad=True)
    p1.grad = torch.randn_like(p1)
    p2.grad = torch.randn_like(p2)
    opt = Muon([p1, p2], pipeline=pipeline, lr=1e-3,
               fused_param_lists=[[p1, p2]])
    opt.step()  # should not raise
