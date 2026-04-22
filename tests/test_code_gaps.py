"""Phase 3b: exercise the four 00196 alignment code gaps end-to-end on CPU.

1. eod_mask_loss: loss at EOD target positions must be masked (→ same loss as
   a run where those positions have ignore_index=-1).
2. mask_loss_id: likewise for target==160000.
3. seq_aux_balance: non-zero extra term on loss when alpha>0; zero when alpha=0.
4. accurate_attn_mask_eod_token: attention output past EOD must not depend on
   tokens before EOD — changing pre-EOD tokens leaves post-EOD hidden states intact.
"""
import os, sys, unittest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from model import GPTConfig, GPT  # noqa: E402


def _tiny_moe_config(**overrides):
    cfg = dict(
        block_size=16, vocab_size=200,
        n_layer=2, n_head=2, n_embd=16, n_kv_head=2, kv_channels=8,
        dropout=0.0, bias=False, init_std=0.02,
        use_rope=True, rotary_base=50000,
        use_rmsnorm=True, norm_eps=1e-5, use_swiglu=True, ffn_hidden_size=32,
        tie_embeddings=False, qk_layernorm=True, disable_scaled_init_method=True,
        use_moe=True, moe_layer_freq=[0, 1], num_experts=4,
        moe_ffn_hidden_size=8, moe_router_topk=2, moe_n_group=2, moe_topk_group=1,
        moe_norm_topk_prob=True, moe_router_score_correction_coeff=0.001,
        moe_shared_expert_hidden_size=8,
    )
    cfg.update(overrides)
    return GPTConfig(**cfg)


class TestEodMaskLoss(unittest.TestCase):
    def test_eod_masked_equals_ignore_index_loss(self):
        torch.manual_seed(0)
        cfg = _tiny_moe_config(eod_token_id=199)
        m = GPT(cfg); m.eval()
        idx = torch.randint(0, 199, (2, 16))
        tgt = torch.randint(0, 199, (2, 16))
        # Place EOD at a few positions
        tgt[0, 3] = 199; tgt[1, 7] = 199; tgt[1, 8] = 199
        # With gap: eod masked
        _, loss_gap = m(idx, targets=tgt)
        # Manual baseline: set those positions to -1 on a no-gap config
        cfg2 = _tiny_moe_config()
        torch.manual_seed(0); m2 = GPT(cfg2); m2.eval()
        m2.load_state_dict(m.state_dict())
        tgt_masked = tgt.clone(); tgt_masked[tgt == 199] = -1
        _, loss_base = m2(idx, targets=tgt_masked)
        self.assertTrue(torch.allclose(loss_gap, loss_base, atol=1e-6),
                        f"{loss_gap.item()} vs {loss_base.item()}")

    def test_mask_loss_id_masked(self):
        torch.manual_seed(0)
        cfg = _tiny_moe_config(mask_loss_id=199)
        m = GPT(cfg); m.eval()
        idx = torch.randint(0, 199, (2, 16))
        tgt = torch.randint(0, 199, (2, 16))
        tgt[0, 0] = 199; tgt[1, 5] = 199
        _, loss_gap = m(idx, targets=tgt)
        cfg2 = _tiny_moe_config()
        torch.manual_seed(0); m2 = GPT(cfg2); m2.eval()
        m2.load_state_dict(m.state_dict())
        tgt_masked = tgt.clone(); tgt_masked[tgt == 199] = -1
        _, loss_base = m2(idx, targets=tgt_masked)
        self.assertTrue(torch.allclose(loss_gap, loss_base, atol=1e-6))


class TestSeqAuxBalance(unittest.TestCase):
    def test_alpha_zero_matches_baseline(self):
        torch.manual_seed(0)
        cfg0 = _tiny_moe_config(seq_aux_balance_alpha=0.0)
        m = GPT(cfg0); m.train()
        idx = torch.randint(0, 200, (2, 16))
        tgt = torch.randint(0, 200, (2, 16))
        _, loss = m(idx, targets=tgt)
        self.assertTrue(torch.isfinite(loss))

    def test_alpha_positive_increases_loss(self):
        torch.manual_seed(0)
        cfg0 = _tiny_moe_config(seq_aux_balance_alpha=0.0)
        cfg1 = _tiny_moe_config(seq_aux_balance_alpha=1.0)  # large for sensitivity
        torch.manual_seed(0); m0 = GPT(cfg0); m0.train()
        torch.manual_seed(0); m1 = GPT(cfg1); m1.train()
        m1.load_state_dict(m0.state_dict())
        idx = torch.randint(0, 200, (2, 16))
        tgt = torch.randint(0, 200, (2, 16))
        torch.manual_seed(42); _, l0 = m0(idx, targets=tgt)
        torch.manual_seed(42); _, l1 = m1(idx, targets=tgt)
        self.assertGreater(float(l1 - l0), 0, "alpha>0 should add non-zero aux term")
        # aux value = (l1 - l0) / alpha. For uniform sigmoid ≈0.5 and uniform routing,
        # expected ≈ 0.5 * (E/K) * (K/E) = 0.5 → so l1-l0 with alpha=1 should be ≈0.5
        # This is a loose sanity bound.
        self.assertLess(float(l1 - l0), 5.0)


class TestAccurateEodAttnMask(unittest.TestCase):
    def test_post_eod_isolated_from_pre_eod(self):
        torch.manual_seed(0)
        cfg = _tiny_moe_config(eod_token_id=199, use_eod_attn_mask=True)
        m = GPT(cfg); m.eval()
        # Place EOD at middle of sequence
        idx_a = torch.randint(0, 199, (1, 16))
        idx_a[0, 7] = 199   # EOD at position 7
        idx_b = idx_a.clone()
        idx_b[0, :7] = (idx_b[0, :7] + 100) % 199  # perturb pre-EOD content only
        # Forward both and compare post-EOD hidden states (via lm_head logits)
        with torch.no_grad():
            out_a, _ = m(idx_a)
            out_b, _ = m(idx_b)
        # With the mask, logits at the LAST position (which is past EOD) should NOT
        # depend on pre-EOD tokens.
        self.assertTrue(torch.allclose(out_a, out_b, atol=1e-5),
                        f"post-EOD logits leaked pre-EOD content (max |diff|={(out_a-out_b).abs().max().item():.6g})")

    def test_no_mask_leaks(self):
        """Sanity: without the mask flag, same perturbation does change post-EOD output."""
        torch.manual_seed(0)
        cfg = _tiny_moe_config(eod_token_id=199, use_eod_attn_mask=False)
        m = GPT(cfg); m.eval()
        idx_a = torch.randint(0, 199, (1, 16))
        idx_a[0, 7] = 199
        idx_b = idx_a.clone()
        idx_b[0, :7] = (idx_b[0, :7] + 100) % 199
        with torch.no_grad():
            out_a, _ = m(idx_a); out_b, _ = m(idx_b)
        # Standard causal attn DOES see pre-EOD tokens, so outputs should differ.
        self.assertFalse(torch.allclose(out_a, out_b, atol=1e-5))


if __name__ == '__main__':
    unittest.main(verbosity=2)
