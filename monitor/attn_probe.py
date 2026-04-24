"""Attention-map probe for learning-dynamics monitoring.

Non-invasive: temporarily replaces CausalSelfAttention.forward with a
manual-softmax variant that captures attention weights, then restores.
Run only in eval mode under no_grad — does not affect training numerics
or bitwise determinism.

Usage:
    from monitor.attn_probe import probe_attention
    snapshot = probe_attention(raw_model, probe_tokens)
    json.dump(snapshot, open('reports/attention_maps.json', 'w'))

Output schema:
    {
      'n_layer': int, 'n_head': int, 'T': int, 'T_down': int,
      'iter': int | None,
      'maps': [{layer: i, head: j, matrix: [[T_down x T_down floats]]}, ...],
      'metrics_per_layer': [{
          layer: i, sink_strength: float, mean_entropy: float,
          head_entropy: [float, ...],
      }, ...],
    }

Downsample strategy: block-mean via torch.nn.functional.adaptive_avg_pool2d
from T×T → T_down×T_down (default 32). Preserves global structure while
keeping payload small enough to embed in dashboard HTML.
"""
import math
import types

import torch
import torch.nn.functional as F


def probe_attention(model, probe_tokens, downsample=32, cap_heads=8,
                    iter_tag=None):
    """Capture per-layer, per-head attention maps for one forward pass.

    Args:
        model: the nanogpt GPT module (raw_model, i.e. post-DDP unwrap).
        probe_tokens: [B, T] long tensor of token ids on model's device.
        downsample: target side length of stored heatmaps (default 32).
        cap_heads: max heads per layer to embed in the snapshot (default 8).
        iter_tag: optional iter_num to stamp into the snapshot.

    Returns:
        dict (see module docstring).
    """
    from model import CausalSelfAttention

    captures = {}
    layer_counter = [0]
    originals = {}

    def make_probe_forward(layer_idx):
        def probe_forward(self, x, attn_mask=None, position_ids=None):
            B, T, C = x.size()
            if self.use_rope:
                q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
                v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
                if self.q_layernorm is not None:
                    q = self.q_layernorm(q)
                if self.k_layernorm is not None:
                    k = self.k_layernorm(k)
                q, k = self.rotary_emb(q, k, seq_len=T, position_ids=position_ids)
                if self.n_rep > 1:
                    k = k.unsqueeze(2).expand(B, self.n_kv_head, self.n_rep, T, self.head_dim).reshape(B, self.n_head, T, self.head_dim)
                    v = v.unsqueeze(2).expand(B, self.n_kv_head, self.n_rep, T, self.head_dim).reshape(B, self.n_head, T, self.head_dim)
            else:
                q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
                q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
                v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

            # Manual softmax in fp32 so we can capture attention weights.
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q.float() @ k.float().transpose(-2, -1)) * scale
            causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            if attn_mask is None:
                att = att.masked_fill(~causal, float('-inf'))
            else:
                att = att.masked_fill(~attn_mask, float('-inf'))
            att = F.softmax(att, dim=-1)                  # [B, H, T, T] fp32
            captures[layer_idx] = att.detach().cpu()

            y = (att @ v.float()).to(q.dtype)
            y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
            return self.resid_dropout(self.c_proj(y))
        return probe_forward

    for _, m in model.named_modules():
        if isinstance(m, CausalSelfAttention):
            originals[id(m)] = m.forward
            m.forward = types.MethodType(make_probe_forward(layer_counter[0]), m)
            layer_counter[0] += 1

    try:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            model(probe_tokens)
    finally:
        for _, m in model.named_modules():
            if isinstance(m, CausalSelfAttention):
                m.forward = originals[id(m)]
        if was_training:
            model.train()

    return _summarize(captures, downsample, cap_heads, iter_tag)


def _summarize(captures, down, cap_heads, iter_tag):
    result = {'maps': [], 'metrics_per_layer': [], 'iter': iter_tag}
    if not captures:
        return result

    for layer_idx, att in sorted(captures.items()):
        # att: [B, H, T, T]; take batch[0] for visualization
        B, H, T, _ = att.shape
        a = att[0].float()                               # [H, T, T]
        T_down = min(down, T)
        # Downsample by RAW SUM over each (block_q × block_k) tile. Each
        # pooled cell = total attention mass transferred from q-bucket-i to
        # k-bucket-j. Neither row sums nor column sums equal 1:
        #   row_sum    = block_q  (attention mass generated by one q-bucket)
        #   col_sum    = varies   (total attention received by a k-bucket)
        #   matrix_sum = T        (= sum of all original row sums = T × 1)
        # This is the "sum of probability mass" interpretation — purely
        # additive, no hidden normalization, nothing is forced to 1.
        if T > T_down:
            avg = F.adaptive_avg_pool2d(a.unsqueeze(0), (T_down, T_down))[0]
            block_q = T / T_down
            block_k = T / T_down
            pool = avg * block_q * block_k
        else:
            pool = a
        H_viz = min(H, cap_heads)
        # Dynamic precision: keep 3 sig figs across the dynamic range of the
        # pooled values. Attention max-pool at long seq_len can sit at 1.0;
        # avg-pool at long seq_len sits near 1/(T/T_down). Use log-based
        # precision so both cases keep detail without bloating payload.
        for h in range(H_viz):
            sub = pool[h]
            mx = float(sub.max().item())
            # precision = digits after the decimal needed for 3 sig figs
            if mx > 0:
                ndig = max(3, 3 - int(math.floor(math.log10(mx))))
            else:
                ndig = 3
            mat = [[round(float(x), ndig) for x in row] for row in sub.tolist()]
            result['maps'].append({
                'layer': layer_idx, 'head': h, 'matrix': mat,
            })

        # Metrics
        sink = float(a[:, :, 0].mean().item())
        log_T = math.log(T) if T > 1 else 1.0
        head_ent = []
        for h in range(H):
            p = a[h]
            ent = -(p * (p.clamp(min=1e-20)).log()).sum(dim=-1).mean().item()
            head_ent.append(round(ent / log_T, 4))
        result['metrics_per_layer'].append({
            'layer': layer_idx,
            'sink_strength': round(sink, 4),
            'mean_entropy': round(sum(head_ent) / len(head_ent) if head_ent else 0.0, 4),
            'head_entropy': head_ent,
        })
        result['T'] = T
        result['T_down'] = T_down
        result['n_head'] = H
    result['n_layer'] = len(captures)
    return result
