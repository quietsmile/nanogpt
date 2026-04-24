"""MLP + MoE — bitwise-preserved split of v1.0."""
import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class MLP(nn.Module):
    """Standard MLP with GELU (GPT-2 style)."""

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class SwiGLUMLP(nn.Module):
    """SwiGLU FFN, matching cybertron's ffn_hidden_size convention.

    The cybertron config uses:
      ffn_hidden_size = 1536  (the hidden dim of each of the two gate projections)
    The output is: swish(gate(x)) * up(x), then projected down.
    """

    def __init__(self, config):
        super().__init__()
        ffn_hidden = config.ffn_hidden_size
        self.gate_proj = nn.Linear(config.n_embd, ffn_hidden, bias=config.bias)
        self.up_proj   = nn.Linear(config.n_embd, ffn_hidden, bias=config.bias)
        self.down_proj = nn.Linear(ffn_hidden, config.n_embd, bias=config.bias)
        self.dropout   = nn.Dropout(config.dropout)

    def forward(self, x):
        # fp32 silu*up then cast to bf16 — matches TE's fused SwiGLU kernel (single bf16
        # round at end). Pure `F.silu(g)*u` under autocast rounds silu output to bf16
        # before multiplication → loses 1 ULP on ~26% of positions.
        g = self.gate_proj(x)
        u = self.up_proj(x)
        h = (F.silu(g.float()) * u.float()).to(x.dtype)
        return self.dropout(self.down_proj(h))


class ExpertMLP(nn.Module):
    """Single routed expert: SwiGLU FFN (512→160→512 for MoE 198)."""

    def __init__(self, n_embd, hidden_size, bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, hidden_size, bias=bias)
        self.up_proj   = nn.Linear(n_embd, hidden_size, bias=bias)
        self.down_proj = nn.Linear(hidden_size, n_embd, bias=bias)

    def forward(self, x):
        g = self.gate_proj(x)
        u = self.up_proj(x)
        h = (F.silu(g.float()) * u.float()).to(x.dtype)
        return self.down_proj(h)


class MoERouter(nn.Module):
    """Grouped sigmoid top-k router with expert score correction bias.

    Matches cybertron's router_scoring_func=sigmoid, n_group grouped routing,
    norm_topk_prob=True, and use_router_expert_score_correction.
    """

    def __init__(self, n_embd, num_experts, topk, n_group, topk_group,
                 norm_topk_prob, score_correction_coeff,
                 routing_type='group_limited_greedy'):
        super().__init__()
        assert num_experts % n_group == 0, "num_experts must be divisible by n_group"
        self.num_experts = num_experts
        self.topk = topk
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.score_correction_coeff = score_correction_coeff
        self.experts_per_group = num_experts // n_group
        self.routing_type = routing_type   # 'greedy' | 'group_limited_greedy'

        # Router linear (fp32 forward via moe_gating_fp32)
        self.linear = nn.Linear(n_embd, num_experts, bias=False)
        # Score correction bias: updated once per optim step via update_expert_bias(),
        # NOT via gradients. See Megatron router.py:444 and finalize_model_grads.py:270.
        self.register_buffer('e_score_correction_bias', torch.zeros(num_experts))
        # Load counts accumulated across grad_accum micro-steps; consumed + reset by
        # update_expert_bias() after the accumulation loop completes.
        self.register_buffer('local_tokens_per_expert',
                             torch.zeros(num_experts, dtype=torch.float32))
        # Per-forward diagnostic stats captured for logging. Refreshed each forward.
        # Not a parameter; training.py reads these between microbatches and aggregates.
        # Fields: score_mean, score_std, topk_margin_p5, topk_margin_median
        self._last_score_mean = 0.0
        self._last_score_std = 0.0
        self._last_topk_margin_p5 = 0.0
        self._last_topk_margin_median = 0.0
        # Per-microbatch routing stats: we snapshot counts at each forward so
        # train.py can compute per-mb max/min/mean without needing to diff the
        # accumulator. Updated from local counts delta after the scatter_add below.
        self.register_buffer('_last_mb_counts',
                             torch.zeros(num_experts, dtype=torch.float32),
                             persistent=False)
        self._prev_total = 0.0  # scalar tracking total count before this fwd

    def forward(self, x):  # x: [S, n_embd]
        S = x.shape[0]
        E = self.num_experts
        G = self.n_group
        K = self.topk

        # 1. Compute scores in fp32 (moe_gating_fp32=True). We MUST disable autocast here,
        # otherwise F.linear's output is downcast to bf16 regardless of .float() on inputs,
        # producing bf16-rounded logits that route 1-2% of tokens to wrong experts at
        # boundary cases. Matches cybertron's moe_gating_fp32=True path.
        with torch.amp.autocast('cuda', enabled=False):
            logits = F.linear(x.float(), self.linear.weight.float())  # [S, E], fp32
            scores = torch.sigmoid(logits)                            # [S, E], fp32
            # 2. Add correction bias for routing decision (NOT added to final weights)
            scores_biased = scores + self.e_score_correction_bias.float()

        # 3. Routing: flat top-K (cybertron routing_type='greedy') OR group-limited.
        if self.routing_type == 'greedy':
            # Flat top-K over all experts — matches cybertron's DeepSeekRoutingType.GREEDY
            topk_idx = scores_biased.topk(K, dim=-1).indices  # [S, K]
        else:
            # Group-limited top-K: sum top-(K//topk_group) per group → group score;
            # select topk_group groups; pick top-K among those groups' experts.
            epg = self.experts_per_group
            group_scores = scores_biased.view(S, G, epg).topk(K // self.topk_group, dim=-1).values.sum(dim=-1)
            group_idx = group_scores.topk(self.topk_group, dim=-1).indices  # [S, topk_group]
            group_mask = torch.zeros(S, G, dtype=scores_biased.dtype, device=x.device)
            group_mask.scatter_(1, group_idx, 1.0)
            score_mask = group_mask.unsqueeze(-1).expand(S, G, epg).reshape(S, E)
            scores_masked = scores_biased.masked_fill(score_mask == 0, float('-inf'))
            topk_idx = scores_masked.topk(K, dim=-1).indices  # [S, K]

        # 4. Final weights: ORIGINAL (unbiased) scores at selected positions. KEEP FP32
        # to match cybertron's path where weighted-sum of expert outputs is done in fp32.
        final_weights = scores.gather(1, topk_idx)  # [S, K], fp32
        if self.norm_topk_prob:
            final_weights = final_weights / (final_weights.sum(dim=-1, keepdim=True) + 1e-20)

        # 7. Accumulate token counts across grad_accum micro-steps. The actual bias
        #    update + all-reduce + reset happens in update_expert_bias(), called once
        #    per optim step from the training loop. This matches Megatron's behaviour:
        #    router.py:444-448 (accumulate) + finalize_model_grads.py:270-295 (update).
        if torch.is_grad_enabled():
            with torch.no_grad():
                # Keep counter in fp32 even if the module was .bfloat16()'d (mirrors
                # Megatron's _maintain_float32_expert_bias pattern).
                if self.local_tokens_per_expert.dtype != torch.float32:
                    self.local_tokens_per_expert.data = self.local_tokens_per_expert.data.float()
                # Snapshot counts BEFORE this forward's scatter so train.py can see
                # per-microbatch counts (_last_mb_counts = after - before).
                pre = self.local_tokens_per_expert.clone()
                self.local_tokens_per_expert.scatter_add_(
                    0, topk_idx.reshape(-1),
                    torch.ones(S * K, dtype=torch.float32, device=x.device)
                )
                self._last_mb_counts.copy_(self.local_tokens_per_expert - pre)
                # Additional diagnostics (fp32, cheap): score distribution stats
                # and topk margin. These are populated on every forward; train.py
                # aggregates over microbatches at log time.
                _sc = scores.detach()
                self._last_score_mean = float(_sc.mean().item())
                self._last_score_std  = float(_sc.std().item())
                _sorted, _ = scores_biased.detach().sort(dim=-1, descending=True)
                _margin = (_sorted[:, K-1] - _sorted[:, K]).abs()
                self._last_topk_margin_p5 = float(_margin.quantile(0.05).item())
                self._last_topk_margin_median = float(_margin.median().item())

        # Return also raw sigmoid scores (pre-bias, pre-group-mask) for seq_aux balance loss
        return topk_idx, final_weights, scores  # [S, K], [S, K], [S, E]

    @torch.no_grad()
    def update_expert_bias(self):
        """Apply one aux-free bias update from the accumulated token counts.

        Must be called once per optim step, AFTER the grad_accum micro-steps finish
        and BEFORE optimizer.step(). Equivalent to Megatron's
        `_update_router_expert_bias` in finalize_model_grads.py.
        """
        import torch.distributed as dist
        # Keep both buffers in fp32 (mirrors Megatron's _maintain_float32_expert_bias).
        if self.local_tokens_per_expert.dtype != torch.float32:
            self.local_tokens_per_expert.data = self.local_tokens_per_expert.data.float()
        if self.e_score_correction_bias.dtype != torch.float32:
            self.e_score_correction_bias.data = self.e_score_correction_bias.data.float()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.local_tokens_per_expert, op=dist.ReduceOp.SUM)
        mean_load = self.local_tokens_per_expert.mean()
        self.e_score_correction_bias.add_(
            (mean_load - self.local_tokens_per_expert).sign()
            * self.score_correction_coeff
        )
        self.local_tokens_per_expert.zero_()


class MoEFFN(nn.Module):
    """MoE FFN layer: grouped routing + N experts + always-on shared expert.

    Matches cybertron's MoE structure: shared expert output + weighted sum of
    top-K routed expert outputs.

    Dispatch is a single `torch._grouped_mm` call per projection (gate/up/down),
    not a Python for-loop over experts. This removes the O(num_experts) dispatch
    overhead that crushed MFU on the naive impl (see perf notes in RUNBOOK).
    """

    def __init__(self, config):
        super().__init__()
        self.router = MoERouter(
            n_embd=config.n_embd,
            num_experts=config.num_experts,
            topk=config.moe_router_topk,
            n_group=config.moe_n_group,
            topk_group=config.moe_topk_group,
            norm_topk_prob=config.moe_norm_topk_prob,
            score_correction_coeff=config.moe_router_score_correction_coeff,
            routing_type=getattr(config, 'moe_routing_type', 'group_limited_greedy'),
        )
        E = config.num_experts
        C = config.n_embd
        H = config.moe_ffn_hidden_size
        # Stacked expert weights for grouped_mm. Total params == E × ExpertMLP.
        # Layout: grouped_mm(x [M,C], w [E,C,H], offs) -> [M,H].
        self.gate_weight = nn.Parameter(torch.empty(E, C, H))
        self.up_weight   = nn.Parameter(torch.empty(E, C, H))
        self.down_weight = nn.Parameter(torch.empty(E, H, C))
        nn.init.normal_(self.gate_weight, mean=0.0, std=config.init_std)
        nn.init.normal_(self.up_weight,   mean=0.0, std=config.init_std)
        nn.init.normal_(self.down_weight, mean=0.0, std=config.init_std)

        if config.moe_shared_expert_hidden_size is not None:
            self.shared_expert = ExpertMLP(
                config.n_embd, config.moe_shared_expert_hidden_size, bias=config.bias
            )
        else:
            self.shared_expert = None
        self.num_experts = config.num_experts
        self.topk = config.moe_router_topk
        self.seq_aux_alpha = getattr(config, 'seq_aux_balance_alpha', 0.0) or 0.0

    def forward(self, x):  # x: [B, T, C]
        B, T, C = x.shape
        x_flat = x.view(B * T, C)

        # Shared expert (always active, moe_shared_expert_overlap=False → sequential)
        if self.shared_expert is not None:
            shared_out = self.shared_expert(x_flat)  # [B*T, C]
        else:
            shared_out = torch.zeros_like(x_flat)

        # Routed experts: grouped-GEMM dispatch (single CUDA op per projection)
        topk_idx, weights, raw_scores = self.router(x_flat)  # [B*T, K], [B*T, K], [B*T, E]
        S = x_flat.shape[0]
        K = self.topk
        E = self.num_experts

        # TE fused path (default on CUDA). Uses te.moe_permute_with_probs +
        # te.GroupedLinear, which avoids the bucket-padding approach below —
        # peak activation memory drops ~27 GB at mb=1 for the 9L/144-expert
        # config (see /tmp/mem_profile.py). TE requires CUDA tensors; the
        # bucket-padding fallback runs on CPU (used by unit tests) and when
        # NANO_TE_MOE=0 is set (for numerics debugging).
        import os as _os
        _use_te_moe = (
            _os.environ.get('NANO_TE_MOE', '1') == '1'
            and x.is_cuda
        )
        if _use_te_moe:
            import transformer_engine.pytorch as _te
            # Build routing_map [S, E] int32 and probs [S, E] fp32
            routing_map = torch.zeros(S, E, dtype=torch.int32, device=x.device)
            routing_map.scatter_(1, topk_idx, 1)
            probs = torch.zeros(S, E, dtype=torch.float32, device=x.device)
            probs.scatter_(1, topk_idx, weights.float())
            # Permute tokens
            permuted, _perm_probs, row_id_map = _te.moe_permute_with_probs(
                x_flat, probs, routing_map, num_out_tokens=S*K,
            )
            # Expert compute via grouped GEMM (gate/up fused)
            # Build per-expert fused weights [E, 2H, C] and [E, C, H] once per call —
            # TODO cache if perf matters. For now, reinstantiate.
            # Use te.GroupedLinear for experts (matches ref's fused grouped_gemm)
            H_ffn = self.gate_weight.shape[-1]
            # Speed: bf16 weights in GroupedLinear match ref's bf16=True yaml and
            # avoid the fp32×bf16 matmul penalty. Env NANO_TE_MOE_FP32=1 to opt out.
            import os as _os
            _gl_dtype = torch.float32 if _os.environ.get('NANO_TE_MOE_FP32', '0') == '1' else torch.bfloat16
            if not hasattr(self, '_gl_fc1'):
                self._gl_fc1 = _te.GroupedLinear(
                    num_gemms=E, in_features=C, out_features=2*H_ffn, bias=False,
                    params_dtype=_gl_dtype, device=x.device,
                )
                self._gl_fc2 = _te.GroupedLinear(
                    num_gemms=E, in_features=H_ffn, out_features=C, bias=False,
                    params_dtype=_gl_dtype, device=x.device,
                )
                # Copy nano weights into per-expert weights (gate_weight [E,C,H], up_weight [E,C,H], down_weight [E,H,C])
                with torch.no_grad():
                    for e in range(E):
                        gate_e = self.gate_weight[e].T.contiguous()  # [H, C]
                        up_e = self.up_weight[e].T.contiguous()      # [H, C]
                        fc1_e = torch.cat([gate_e, up_e], dim=0)     # [2H, C]
                        getattr(self._gl_fc1, f'weight{e}').data.copy_(fc1_e.to(_gl_dtype))
                        getattr(self._gl_fc2, f'weight{e}').data.copy_(self.down_weight[e].T.contiguous().to(_gl_dtype))
            m_splits = routing_map.sum(dim=0).tolist()
            h12 = self._gl_fc1(permuted, m_splits=m_splits)
            gate, up = h12.chunk(2, dim=-1)
            h_act = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
            out_perm = self._gl_fc2(h_act, m_splits=m_splits)
            # Unpermute with fused probs-weighted merge
            routed_out = _te.moe_unpermute(
                out_perm, row_id_map, merging_probs=probs,
                restore_shape=x_flat.shape, map_type='mask',
            )
            # Fp32 sum then cast (matches nano's non-TE path)
            out = (shared_out.float() + routed_out.float()).to(x_flat.dtype).view(B, T, C)
            # Sequence-wise balance aux loss — same formula as the bucket-path
            # branch below (modules_deepseekv2.py:374-378). Must also run on the
            # TE path so training sees the aux gradient.
            if self.training and self.seq_aux_alpha > 0:
                topk_b = topk_idx.view(B, T, K)
                counts_bE = torch.zeros(B, E, device=x.device, dtype=torch.float32)
                src = torch.ones_like(topk_b, dtype=torch.float32)
                counts_bE.scatter_add_(1, topk_b.reshape(B, T * K), src.reshape(B, T * K))
                fii = counts_bE * (float(E) / (float(T) * float(K)))
                pi = raw_scores.view(B, T, E).float()
                pii = (pi / (pi.sum(dim=-1, keepdim=True) + 1e-20)).mean(dim=1)
                aux = (fii * pii).sum(dim=1).mean()
            else:
                aux = x.new_zeros(())
            return out, aux

        # Expand tokens across topk → [S*K, C] and pair with flat (expert, weight)
        flat_tokens  = x_flat.unsqueeze(1).expand(S, K, C).reshape(S * K, C)
        flat_experts = topk_idx.reshape(-1)             # [S*K]
        # Keep weights in fp32 to match cybertron's weighted-sum precision
        flat_weights = weights.reshape(-1).float()      # [S*K], fp32

        # Sort so tokens for each expert are contiguous
        order          = flat_experts.argsort(stable=True)
        sorted_tokens  = flat_tokens[order].contiguous()
        sorted_weights = flat_weights[order]
        sorted_experts = flat_experts[order]

        # Per-expert counts
        counts = sorted_experts.bincount(minlength=E)                     # [E]
        offs   = counts.cumsum(0)                                          # [E] long
        orig_starts = torch.cat([offs.new_zeros(1), offs[:-1]])            # [E]
        # Each row's local offset within its expert bucket
        row_ids   = torch.arange(S * K, device=x.device, dtype=torch.int64)
        local_off = row_ids - orig_starts[sorted_experts]                  # [S*K]

        # Pad each bucket to max(counts) (aligned) for bmm — reliable autograd, no grouped_mm bugs
        M_per = int(counts.max().item())                                   # H2D sync, one per layer
        if M_per < 8:
            M_per = 8
        bucket = sorted_tokens.new_zeros(E, M_per, C)
        bucket[sorted_experts, local_off] = sorted_tokens
        # Three batched GEMMs: bucket [E, M_per, C] @ weight [E, C, H] → [E, M_per, H]
        gate = torch.bmm(bucket, self.gate_weight)
        up_  = torch.bmm(bucket, self.up_weight)
        # fp32 silu*up then cast — matches TE fused SwiGLU (single bf16 round)
        h = (F.silu(gate.float()) * up_.float()).to(gate.dtype)
        out_bucket = torch.bmm(h, self.down_weight)                        # [E, M_per, C]
        # Gather expert outputs in sorted (by expert) order, apply weights, then
        # scatter_add back to token positions. This matches Megatron's unpermute
        # path: `output.scatter_add_(0, sorted_indices, permuted_tokens * probs)`
        # (bf16-sum order differs from per-token K-wise sum, affects loss at ULP level).
        # Weighted sum in fp32 to match TE's moe_unpermute precision.
        out_flat = out_bucket[sorted_experts, local_off].float()           # [S*K, C] fp32
        out_flat = out_flat * sorted_weights.unsqueeze(1).float()          # fp32 * fp32
        sorted_token_ids = row_ids.div(K, rounding_mode='floor')[order]
        routed_out = torch.zeros(S, C, dtype=torch.float32, device=x.device)  # fp32 accumulator
        routed_out.scatter_add_(0, sorted_token_ids.unsqueeze(1).expand(-1, C), out_flat)
        # Cast to bf16 only at final combine
        out = (shared_out.float() + routed_out).to(x_flat.dtype).view(B, T, C)

        # Sequence-wise balance aux loss.
        # Exact match to cybertron_dots3.0_swa/cybertron/models/deepseek_v2/modules_deepseekv2.py:374-378:
        #   fii = fi.sum(0).div_(seq_len * K / E)   # [B, E] = count * E / (T*K)
        #   pii = pi.div(pi.sum(-1, keepdim=True) + 1e-20).mean(0)   # [B, E]  per-token normalized
        #   aux = (fii * pii).sum(dim=1).mean()
        if self.training and self.seq_aux_alpha > 0:
            E, K = self.num_experts, self.topk
            topk_b = topk_idx.view(B, T, K)                                   # [B, T, K]
            counts_bE = torch.zeros(B, E, device=x.device, dtype=torch.float32)
            src = torch.ones_like(topk_b, dtype=torch.float32)
            counts_bE.scatter_add_(1, topk_b.reshape(B, T * K), src.reshape(B, T * K))
            fii = counts_bE * (float(E) / (float(T) * float(K)))              # [B, E]
            pi = raw_scores.view(B, T, E).float()                             # [B, T, E]
            pii = (pi / (pi.sum(dim=-1, keepdim=True) + 1e-20)).mean(dim=1)   # [B, E]
            aux = (fii * pii).sum(dim=1).mean()                                # scalar
        else:
            aux = x.new_zeros(())
        return out, aux


