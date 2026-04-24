"""Bitwise-equivalence: nanogpt.model.GPT == legacy model.GPT under same init.

Tests on tiny CPU configs only — end-to-end forward/backward, with and
without MoE, with and without aux loss. PAI 50-iter full-DDP validation is
left for stage-2 regression job (T7).
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _build_cfg(use_moe=False, **overrides):
    """Identical config for both legacy model.GPT and nanogpt.model.GPT."""
    base = dict(
        block_size=64, vocab_size=128, n_layer=2, n_head=2, n_embd=16,
        n_kv_head=2, kv_channels=8, dropout=0.0, bias=False, init_std=0.02,
        use_rope=True, rotary_base=10000, use_rmsnorm=True, norm_eps=1e-5,
        use_swiglu=True, ffn_hidden_size=32,
        qk_layernorm=True, tie_embeddings=False,
        disable_scaled_init_method=False,
        use_moe=use_moe,
    )
    if use_moe:
        base.update(
            moe_layer_freq=[0, 1],
            num_experts=4,
            moe_ffn_hidden_size=16,
            moe_router_topk=2,
            moe_n_group=2,
            moe_topk_group=1,
            moe_norm_topk_prob=True,
            moe_router_score_correction_coeff=0.001,
            moe_shared_expert_hidden_size=16,
            moe_routing_type="greedy",
            seq_aux_balance_alpha=0.0001,
        )
    base.update(overrides)
    return base


def _build_matched(cfg_dict, seed=0):
    """Build legacy GPT + new GPT with identical initialization."""
    import model as legacy
    from nanogpt.model import GPT, GPTConfig
    torch.manual_seed(seed)
    legacy_cfg = legacy.GPTConfig(**cfg_dict)
    new_cfg = GPTConfig(**cfg_dict)

    torch.manual_seed(seed)
    legacy_model = legacy.GPT(legacy_cfg)
    torch.manual_seed(seed)
    new_model = GPT(new_cfg)
    return legacy_model, new_model


def _params_match(legacy, new):
    """Return True if every param on legacy has an identical-named param on new,
    with bitwise-equal data."""
    legacy_sd = {n: p for n, p in legacy.named_parameters()}
    new_sd = {n: p for n, p in new.named_parameters()}
    if set(legacy_sd) != set(new_sd):
        return False, f"name mismatch: legacy∖new = {set(legacy_sd) - set(new_sd)}"
    for name in legacy_sd:
        if not torch.equal(legacy_sd[name].data, new_sd[name].data):
            return False, f"{name} not bitwise equal"
    return True, "OK"


def test_gpt_dense_forward_bitwise():
    legacy_m, new_m = _build_matched(_build_cfg(use_moe=False), seed=42)
    ok, msg = _params_match(legacy_m, new_m)
    assert ok, msg
    legacy_m.eval(); new_m.eval()
    torch.manual_seed(7)
    x = torch.randint(0, 128, (2, 32))
    with torch.no_grad():
        legacy_logits, _ = legacy_m(x)
        new_logits, _ = new_m(x)
    assert torch.equal(legacy_logits, new_logits), \
        f"max|Δ|={(legacy_logits-new_logits).abs().max().item()}"


def test_gpt_dense_loss_bitwise():
    legacy_m, new_m = _build_matched(_build_cfg(use_moe=False), seed=42)
    legacy_m.train(); new_m.train()
    torch.manual_seed(13)
    x = torch.randint(0, 128, (2, 32))
    y = torch.randint(0, 128, (2, 32))
    _, ll = legacy_m(x, targets=y)
    _, nl = new_m(x, targets=y)
    assert torch.equal(ll, nl), f"Δloss={(ll-nl).abs().item()}"


def test_gpt_moe_forward_bitwise_cpu():
    """MoE CPU path (bucket fallback). No TE lazy init on CPU."""
    legacy_m, new_m = _build_matched(_build_cfg(use_moe=True), seed=3)
    ok, msg = _params_match(legacy_m, new_m)
    assert ok, msg
    legacy_m.eval(); new_m.eval()
    torch.manual_seed(11)
    x = torch.randint(0, 128, (2, 32))
    with torch.no_grad():
        ll, _ = legacy_m(x)
        nl, _ = new_m(x)
    assert torch.equal(ll, nl), f"Δ={(ll-nl).abs().max().item()}"


def test_gpt_moe_aux_loss_bitwise_cpu():
    legacy_m, new_m = _build_matched(_build_cfg(use_moe=True), seed=5)
    legacy_m.train(); new_m.train()
    torch.manual_seed(17)
    x = torch.randint(0, 128, (2, 32))
    y = torch.randint(0, 128, (2, 32))
    _, ll = legacy_m(x, targets=y)
    _, nl = new_m(x, targets=y)
    assert torch.equal(ll, nl)


def test_primitives_rmsnorm_bitwise():
    import model as legacy
    from nanogpt.model import RMSNorm
    torch.manual_seed(1)
    legacy_norm = legacy.RMSNorm(64, eps=1e-5)
    new_norm = RMSNorm(64, eps=1e-5)
    # Copy weights
    with torch.no_grad():
        new_norm.weight.copy_(legacy_norm.weight)
    x = torch.randn(4, 16, 64)
    assert torch.equal(legacy_norm(x), new_norm(x))


def test_primitives_rope_bitwise():
    import model as legacy
    from nanogpt.model import RotaryEmbedding
    legacy_r = legacy.RotaryEmbedding(dim=16, base=10000)
    new_r = RotaryEmbedding(dim=16, base=10000)
    q = torch.randn(2, 4, 32, 16)
    k = torch.randn(2, 2, 32, 16)
    # RoPE API: forward(q, k, seq_len). position_ids path has Tensor-arange issue
    # unrelated to the refactor; both impls delegate identically to the same code.
    lq, lk = legacy_r(q, k, 32)
    nq, nk = new_r(q, k, 32)
    assert torch.equal(lq, nq)
    assert torch.equal(lk, nk)
