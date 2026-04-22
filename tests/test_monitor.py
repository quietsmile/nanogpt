"""Unit tests for monitor/. CPU-only, synthetic data.

Verifies:
  1. NullMonitor (NANOGPT_MONITOR unset) is a genuine no-op — no hooks
     registered, no files written.
  2. Real Monitor writes monitor.jsonl with expected per-step records.
  3. Enabling the monitor does NOT perturb forward numerics — loss at step N
     is bitwise identical to a monitor-off run (hooks must only read + detach).
  4. MoE-specific fields appear for MoE configs (load entropy, bias stats,
     score entropy).
  5. Per-parameter-group grad norms decompose the total grad norm.
"""
import json
import math
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPT, GPTConfig          # noqa: E402
from monitor import create_monitor        # noqa: E402
from monitor.param_groups import classify  # noqa: E402


def _make_moe_model(seed=0):
    torch.manual_seed(seed)
    cfg = GPTConfig(
        block_size=32, vocab_size=256, n_layer=3, n_head=4, n_embd=64,
        dropout=0.0, bias=False, n_kv_head=2, kv_channels=16,
        use_rope=True, rotary_base=10000, use_rmsnorm=True,
        use_swiglu=True, ffn_hidden_size=128,
        qk_layernorm=True, tie_embeddings=False,
        init_std=0.02, disable_scaled_init_method=True,
        use_moe=True, moe_layer_freq=[0, 1, 1],
        num_experts=4, moe_ffn_hidden_size=32,
        moe_router_topk=2, moe_n_group=2, moe_topk_group=1,
        moe_norm_topk_prob=True,
        moe_router_score_correction_coeff=0.001,
        moe_shared_expert_hidden_size=32,
        moe_routing_type='greedy',
    )
    m = GPT(cfg)
    return m, cfg


def _batch(cfg, B=2, T=16, seed=123):
    g = torch.Generator().manual_seed(seed)
    X = torch.randint(0, cfg.vocab_size, (B, T), generator=g)
    Y = torch.randint(0, cfg.vocab_size, (B, T), generator=g)
    return X, Y


def _run_steps(model, optimizer, monitor, cfg, n_steps=4):
    losses = []
    for it in range(n_steps):
        X, Y = _batch(cfg, seed=1000 + it)
        _, loss = model(X, Y)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        monitor.step(it, loss=loss.detach().item(), grad_norm=gn,
                     lr=1e-3, samples=(it + 1) * 32)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.detach().item())
    monitor.close()
    return losses


def test_null_monitor_no_files_no_hooks():
    """When NANOGPT_MONITOR is unset, no disk write, no forward hooks."""
    os.environ.pop('NANOGPT_MONITOR', None)
    m, cfg = _make_moe_model()
    with tempfile.TemporaryDirectory() as td:
        mon = create_monitor(m, None, out_dir=td)
        # The lm_head should have zero forward hooks registered.
        assert len(m.lm_head._forward_hooks) == 0, \
            "NullMonitor must not register hooks"
        mon.step(0, loss=1.0, grad_norm=1.0, lr=1e-3, samples=8)
        mon.close()
        assert not os.path.exists(os.path.join(td, 'monitor.jsonl')), \
            "NullMonitor must not write any file"
    print("OK test_null_monitor_no_files_no_hooks")


def test_real_monitor_writes_jsonl():
    os.environ['NANOGPT_MONITOR'] = '1'
    try:
        m, cfg = _make_moe_model()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as td:
            mon = create_monitor(m, opt, out_dir=td)
            _run_steps(m, opt, mon, cfg, n_steps=3)
            path = os.path.join(td, 'monitor.jsonl')
            assert os.path.exists(path), "monitor.jsonl must be written"
            lines = open(path).readlines()
            assert len(lines) == 3, f"expected 3 records, got {len(lines)}"
            rec = json.loads(lines[0])
            for k in ('iter', 'loss', 'lr', 'grad_norm', 'gn_by_group', 'moe',
                     'final_resid_max', 'final_resid_std'):
                assert k in rec, f"missing field {k}: {rec}"
            # MoE: layers 1 and 2 present (layer 0 is dense)
            moe = rec['moe']
            assert set(moe.keys()) >= {1, 2} or set(moe.keys()) >= {'1', '2'}, \
                f"expected MoE layers 1,2, got {list(moe.keys())}"
            one = moe[1] if 1 in moe else moe['1']
            for k in ('load_entropy_norm', 'load_gini', 'dead', 'near_dead',
                     'tokens_routed', 'bias_max', 'bias_std'):
                assert k in one, f"MoE layer missing {k}: {one}"
    finally:
        os.environ.pop('NANOGPT_MONITOR', None)
    print("OK test_real_monitor_writes_jsonl")


def test_monitor_does_not_perturb_numerics():
    """Loss at each step must match bitwise between monitor-on and monitor-off."""
    def one_run(monitor_on):
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True, warn_only=True)
        m, cfg = _make_moe_model(seed=42)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        if monitor_on:
            os.environ['NANOGPT_MONITOR'] = '1'
        else:
            os.environ.pop('NANOGPT_MONITOR', None)
        with tempfile.TemporaryDirectory() as td:
            mon = create_monitor(m, opt, out_dir=td)
            losses = _run_steps(m, opt, mon, cfg, n_steps=4)
        return losses

    off = one_run(monitor_on=False)
    on = one_run(monitor_on=True)
    os.environ.pop('NANOGPT_MONITOR', None)
    for i, (a, b) in enumerate(zip(off, on)):
        assert a == b, f"step {i}: loss diverged off={a!r} on={b!r} " \
                       "— hooks must be pure reads"
    print("OK test_monitor_does_not_perturb_numerics  (loss series:", off, ")")


def test_grad_norm_by_group_decomposes_total():
    os.environ['NANOGPT_MONITOR'] = '1'
    try:
        m, cfg = _make_moe_model()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as td:
            mon = create_monitor(m, opt, out_dir=td)
            X, Y = _batch(cfg)
            _, loss = m(X, Y)
            loss.backward()
            total = torch.nn.utils.clip_grad_norm_(m.parameters(), 1e9)  # don't clip
            mon.step(0, loss=loss.detach().item(), grad_norm=total,
                     lr=1e-3, samples=8)
            opt.step()
            opt.zero_grad(set_to_none=True)
            mon.close()
            rec = json.loads(open(os.path.join(td, 'monitor.jsonl')).readline())
            by_group = rec['gn_by_group']
            # sqrt of sum-of-squares of group norms ≈ total
            recomposed = math.sqrt(sum(v * v for v in by_group.values()))
            # Accept small fp tolerance from fp32 sum-order differences
            assert abs(recomposed - float(total)) / max(float(total), 1e-8) < 1e-3, \
                f"recomposed {recomposed} vs total {float(total)}"
    finally:
        os.environ.pop('NANOGPT_MONITOR', None)
    print("OK test_grad_norm_by_group_decomposes_total "
          f"(total={float(total):.4f}, groups={list(by_group.keys())})")


def test_param_group_classification():
    assert classify('transformer.wte.weight') == 'embedding'
    assert classify('lm_head.weight') == 'lm_head'
    assert classify('transformer.h.0.ln_1.weight') == 'norm'
    assert classify('transformer.h.0.attn.q_layernorm.weight') == 'norm'
    assert classify('transformer.h.1.mlp.router.linear.weight') == 'router'
    assert classify('transformer.h.1.mlp.shared_expert.gate_proj.weight') \
        == 'shared_expert'
    assert classify('transformer.h.1.mlp.moe_w_gate') == 'routed_expert'
    assert classify('transformer.h.0.attn.q_proj.weight') == 'attn_qkv'
    assert classify('transformer.h.0.mlp.down_proj.weight') == 'ffn_down'
    print("OK test_param_group_classification")


if __name__ == '__main__':
    test_param_group_classification()
    test_null_monitor_no_files_no_hooks()
    test_real_monitor_writes_jsonl()
    test_grad_norm_by_group_decomposes_total()
    test_monitor_does_not_perturb_numerics()
    print("\nAll monitor tests passed.")
