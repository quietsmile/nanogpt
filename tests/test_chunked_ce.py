"""Test ChunkedLinearCrossEntropy matches F.linear + F.cross_entropy exactly in fp32."""
import os, sys, unittest
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from model import ChunkedLinearCrossEntropy, linear_cross_entropy  # noqa: E402


class TestChunkedLinearCE(unittest.TestCase):
    def test_forward_matches_fp32(self):
        torch.default_generator.manual_seed(0)
        N, C, V = 128, 16, 200
        x = torch.randn(N, C, requires_grad=True)
        W = torch.randn(V, C, requires_grad=True) * 0.02
        targets = torch.randint(0, V, (N,))
        # Mask a few positions
        targets[5] = -1; targets[50] = -1; targets[N-1] = -1

        # Standard
        logits = F.linear(x, W)
        ref_loss = F.cross_entropy(logits, targets, ignore_index=-1)

        # Chunked
        loss = ChunkedLinearCrossEntropy.apply(x.detach(), W.detach(), targets, 32, -1)

        self.assertTrue(torch.allclose(ref_loss, loss, atol=1e-6),
                        f"{ref_loss.item()} vs {loss.item()}")

    def test_backward_matches_fp32(self):
        torch.default_generator.manual_seed(1)
        N, C, V = 64, 8, 100
        x = torch.randn(N, C)
        W = torch.randn(V, C) * 0.02
        targets = torch.randint(0, V, (N,))
        targets[7] = -1

        # Standard backward
        x_ref = x.clone().requires_grad_(True)
        W_ref = W.clone().requires_grad_(True)
        logits = F.linear(x_ref, W_ref)
        ref_loss = F.cross_entropy(logits, targets, ignore_index=-1)
        ref_loss.backward()
        ref_grad_x = x_ref.grad.clone(); ref_grad_W = W_ref.grad.clone()

        # Chunked backward
        x_chk = x.clone().requires_grad_(True)
        W_chk = W.clone().requires_grad_(True)
        chk_loss = ChunkedLinearCrossEntropy.apply(x_chk, W_chk, targets, 16, -1)
        chk_loss.backward()

        self.assertTrue(torch.allclose(ref_grad_x, x_chk.grad, atol=1e-5),
                        f"grad_x diff max {(ref_grad_x - x_chk.grad).abs().max().item():.2e}")
        self.assertTrue(torch.allclose(ref_grad_W, W_chk.grad, atol=1e-5),
                        f"grad_W diff max {(ref_grad_W - W_chk.grad).abs().max().item():.2e}")

    def test_all_masked_tokens(self):
        """Edge: if all targets are -1, total_tokens=0 → mean_loss is nan.
        Not a hot path, but shouldn't crash either."""
        x = torch.randn(8, 4)
        W = torch.randn(20, 4)
        targets = torch.full((8,), -1, dtype=torch.long)
        loss = ChunkedLinearCrossEntropy.apply(x, W, targets, 4, -1)
        # NaN is expected — 0/0. Don't require a specific value.
        self.assertTrue(torch.isnan(loss) or loss.item() == 0)

    def test_chunk_size_invariance(self):
        """Output should be identical regardless of chunk size (fp32)."""
        torch.default_generator.manual_seed(2)
        N, C, V = 100, 16, 256
        x = torch.randn(N, C); W = torch.randn(V, C) * 0.02
        targets = torch.randint(0, V, (N,))
        losses = []
        for cs in [10, 25, 50, 100, 200]:
            losses.append(ChunkedLinearCrossEntropy.apply(x, W, targets, cs, -1).item())
        for i in range(1, len(losses)):
            self.assertAlmostEqual(losses[0], losses[i], places=5,
                                   msg=f"chunk size mismatch: {losses}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
