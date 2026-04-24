"""GPT + GPTConfig — bitwise-preserved split of v1.0."""
import os
import math
import inspect
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from .primitives import RMSNorm, LayerNorm, linear_cross_entropy, ChunkedLinearCrossEntropy
from .attention import CausalSelfAttention
from .mlp import MLP, SwiGLUMLP, MoEFFN, MoERouter
from .block import Block


@dataclass
class GPTConfig:
    """Configuration for both original GPT-2 and cybertron-aligned models."""
    block_size: int = 1024
    vocab_size: int = 50304   # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: better and faster

    # --- Cybertron-style options ---
    n_kv_head: Optional[int] = None  # None = MHA; set to < n_head for GQA
    kv_channels: Optional[int] = None  # per-head dimension; None → n_embd // n_head
    use_rope: bool = False            # Use RoPE instead of learned position embeddings
    rotary_base: int = 10000          # RoPE base frequency
    use_rmsnorm: bool = False         # Use RMSNorm instead of LayerNorm
    norm_eps: float = 1e-5            # Epsilon for normalization
    use_swiglu: bool = False          # Use SwiGLU FFN instead of standard GELU MLP
    ffn_hidden_size: Optional[int] = None  # SwiGLU hidden size; defaults to 4*n_embd if None
    tie_embeddings: bool = True       # Tie input and output embeddings (GPT-2 style)
    init_std: float = 0.02            # Weight initialization std
    qk_layernorm: bool = False        # Apply RMSNorm to Q and K per-head, before RoPE
    disable_scaled_init_method: bool = False  # If True, skip 1/sqrt(2*n_layer) scaling for residual projs

    # --- MoE options (DeepSeek aux-free grouped routing) ---
    use_moe: bool = False
    moe_layer_freq: Optional[list] = None  # e.g. [0,1,1,...,1]; None = all dense
    num_experts: int = 64              # routed experts per MoE layer
    moe_ffn_hidden_size: int = 128     # per-expert SwiGLU hidden size
    moe_router_topk: int = 2           # top-K experts selected per token
    moe_n_group: int = 1               # num groups for grouped routing
    moe_topk_group: int = 1            # num groups selected per token (currently only 1)
    moe_norm_topk_prob: bool = True    # normalize top-K scores to sum=1
    moe_router_score_correction_coeff: float = 0.001  # bias update step (aux-free load balance)
    moe_shared_expert_hidden_size: Optional[int] = None  # None = no shared expert
    moe_routing_type: str = 'group_limited_greedy'  # or 'greedy' (flat top-K over all experts)
    expert_model_parallel_size: int = 1  # EP: each rank holds num_experts/ep_size experts and
                                         # all-to-alls tokens to the owning rank. Matches
                                         # Megatron's expert_model_parallel_size. Must divide
                                         # num_experts and world_size.

    # --- Cybertron loss/attention extras (needed for scaling_moe_00196 alignment) ---
    eod_token_id: Optional[int] = None     # if set, loss at positions where target == this id is masked
    mask_loss_id: Optional[int] = None     # additional target id masked from loss (e.g. 160000 sentinel)
    seq_aux_balance_alpha: float = 0.0     # α for sequence-wise MoE balance aux loss; 0 = disabled
    use_eod_attn_mask: bool = False        # attention cannot cross EOD within a packed sequence
    attention_impl: str = 'sdpa'           # 'sdpa' | 'fp32_manual' | 'te' (TransformerEngine DotProductAttention, matches cybertron kernel)
    fp32_residual: bool = False            # keep residual stream in fp32; each block casts sublayer outputs
                                           # to fp32 before x + sublayer(x). Costs ~2x activation memory but
                                           # removes bf16 ULP noise per block that compounds over L layers × N steps.
    chunked_ce: bool = True                # use ChunkedLinearCrossEntropy for lm_head+CE on CUDA training
                                           # forward. Avoids materializing the full [S, V] logits tensor
                                           # (20 GB at mb=4, V=152064). Set False to debug numerics against
                                           # the standard F.linear + F.cross_entropy path.

    def __post_init__(self):
        if self.use_swiglu and self.ffn_hidden_size is None:
            # Default SwiGLU hidden size: 2/3 * 4 * n_embd, rounded to multiple of 64
            self.ffn_hidden_size = int(2 / 3 * 4 * self.n_embd)
            self.ffn_hidden_size = (self.ffn_hidden_size + 63) // 64 * 64
        if not self.use_swiglu:
            self.ffn_hidden_size = 4 * self.n_embd  # not used by MLP, but stored for reference


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        transformer_dict = dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, layer_idx=i) for i in range(config.n_layer)]),
        )
        # Propagate fp32_residual flag. Stored on ln_1 as a cheap per-block marker
        # readable from Block.forward without adding another signature arg.
        for _blk in transformer_dict['h']:
            _blk.ln_1._fp32_residual = getattr(config, 'fp32_residual', False)

        if config.use_rope:
            # No learned position embeddings when using RoPE
            transformer_dict['wpe'] = None
        else:
            transformer_dict['wpe'] = nn.Embedding(config.block_size, config.n_embd)

        if config.use_rmsnorm:
            transformer_dict['ln_f'] = RMSNorm(config.n_embd, eps=config.norm_eps)
        else:
            transformer_dict['ln_f'] = LayerNorm(config.n_embd, bias=config.bias)

        self.transformer = nn.ModuleDict(transformer_dict)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        if config.tie_embeddings:
            self.transformer.wte.weight = self.lm_head.weight

        # Init all weights
        self.apply(self._init_weights)
        # Apply special scaled init to residual projections, per GPT-2 paper.
        # cybertron sets disable_scaled_init_method=True, so skip this when requested.
        if not config.disable_scaled_init_method:
            for pn, p in self.named_parameters():
                if pn.endswith('c_proj.weight') or pn.endswith('down_proj.weight'):
                    torch.nn.init.normal_(p, mean=0.0, std=config.init_std / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            if self.config.use_rope:
                pass  # no wpe to subtract
            else:
                n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)
        # Cast embedding output to current autocast dtype so residual stream stays in bf16.
        # Without this, residual stream accumulates in fp32 (embed outputs fp32 + widest-rule adds)
        # which is MORE precision than ref's bf16 residual. When the fp32_residual
        # ablation is enabled (config.fp32_residual=True) we intentionally skip the
        # bf16 downcast so residual stays fp32 end-to-end.
        if torch.is_autocast_enabled() and not getattr(self.config, 'fp32_residual', False):
            tok_emb = tok_emb.to(torch.get_autocast_dtype('cuda'))

        if self.config.use_rope:
            x = self.transformer.drop(tok_emb)
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)  # (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)

        # Build EOD-aware attention mask + segment-local position_ids when enabled.
        # Matches Megatron's packed_seq_params / _apply_rotary_pos_emb_thd behaviour:
        # attention cannot cross EOD AND RoPE resets to position 0 at each EOD boundary.
        attn_mask = None
        position_ids = None
        eod_id = getattr(self.config, 'eod_token_id', None)
        if getattr(self.config, 'use_eod_attn_mask', False) and eod_id is not None:
            # segment_id[b, k] = number of EOD tokens strictly before position k
            # → positions 0..EOD_pos (inclusive) are in the same segment.
            is_eod = (idx == eod_id).to(torch.int32)
            seg = torch.nn.functional.pad(is_eod.cumsum(dim=1)[:, :-1], (1, 0), value=0)
            # same-segment mask & causal combined
            same_seg = seg.unsqueeze(2) == seg.unsqueeze(1)       # [B, T, T] bool
            causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=device))
            attn_mask = (same_seg & causal).unsqueeze(1)          # [B, 1, T, T]
            # Segment-local position ids: for each position k, offset is "start of
            # current segment". Since seg is monotonically non-decreasing across T,
            # prefix-max of (is_seg_start * arange) gives the per-position segment start.
            is_seg_start = torch.cat(
                [torch.ones(b, 1, dtype=torch.bool, device=device),
                 seg[:, 1:] != seg[:, :-1]],
                dim=1,
            )                                                     # [B, T] bool
            abs_pos = torch.arange(t, device=device).unsqueeze(0).expand(b, t)
            seg_start = torch.where(is_seg_start, abs_pos, torch.zeros_like(abs_pos))
            seg_start = seg_start.cummax(dim=1).values            # [B, T] int
            position_ids = abs_pos - seg_start                    # [B, T] int, resets at EOD

        aux_sum = x.new_zeros(())
        n_moe_layers = 0
        for block in self.transformer.h:
            x, aux = block(x, attn_mask=attn_mask, position_ids=position_ids)
            if getattr(block, 'is_moe', False):
                aux_sum = aux_sum + aux
                n_moe_layers += 1
        x = self.transformer.ln_f(x)

        if targets is not None:
            # Build masked targets first (same logic as before — mask EOD /
            # mask_loss_id on INPUT positions to match ref loss_mask semantics).
            t_flat = targets.view(-1)
            mask_ids = []
            if getattr(self.config, 'eod_token_id', None) is not None:
                mask_ids.append(self.config.eod_token_id)
            if getattr(self.config, 'mask_loss_id', None) is not None:
                mask_ids.append(self.config.mask_loss_id)
            if mask_ids:
                idx_flat = idx.view(-1)
                mask = torch.zeros_like(idx_flat, dtype=torch.bool)
                for mid in mask_ids:
                    mask = mask | (idx_flat == mid)
                t_flat = torch.where(mask, torch.full_like(t_flat, -1), t_flat)

            # Force lm_head compute in fp32 (matches ref fp16_lm_cross_entropy=false).
            # Use chunked linear+CE to avoid materializing the full [S, V] logits
            # tensor — 20 GB at mb=4 × seq=8192 × V=152064, otherwise backward
            # OOMs on 80 GB H100. `logits` is only computed for the separate
            # eval-time / sanity path below.
            _USE_CHUNKED_CE = getattr(self.config, 'chunked_ce', True)
            if _USE_CHUNKED_CE and targets.is_cuda:
                # Hidden fp32 copy (shape [S, C]) is small: mb=4 × seq=8192 × 512 × 4 = 0.5 GB
                x_fp32 = x.float().view(-1, x.shape[-1]).contiguous()
                W_fp32 = self.lm_head.weight.float()
                lm_loss = ChunkedLinearCrossEntropy.apply(
                    x_fp32, W_fp32, t_flat, 2048, -1)
                logits = None  # not materialized in chunked path
            else:
                with torch.amp.autocast('cuda', enabled=False):
                    logits = F.linear(x.float(), self.lm_head.weight.float())
                lm_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), t_flat, ignore_index=-1)
            # Sequence-wise MoE balance aux: stored as a tensor component of loss
            # (for backward) but kept separable so callers can log LM CE alone —
            # matches Megatron, which reports pure `lm loss` and routes aux through
            # MoEAuxLossAutoScaler independently.
            alpha = getattr(self.config, 'seq_aux_balance_alpha', 0.0) or 0.0
            if alpha > 0 and n_moe_layers > 0:
                aux_contrib = alpha * (aux_sum / n_moe_layers)
            else:
                aux_contrib = lm_loss.new_zeros(())
            loss = lm_loss + aux_contrib
            # Expose components on the module for training-loop logging. train.py
            # can subtract aux to log apples-to-apples with ref's TB `lm loss`.
            self.last_lm_loss = lm_loss.detach()
            self.last_aux_contrib = aux_contrib.detach()
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if not self.config.use_rope:
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True
        if 'dropout' in override_args:
            config_args['dropout'] = override_args['dropout']
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, eps, device_type,
                             use_muon=False, muon_lr=None, muon_momentum=0.95, muon_beta2=0.95,
                             muon_weight_decay=None, muon_impl='normuon'):
        """
        Returns either a stock AdamW (use_muon=False, the alignment baseline) or a
        MultiOptimizer holding both Muon and AdamW (use_muon=True).

        muon_impl selects which Muon variant:
          'normuon' (default) — modded-nanogpt NorMuon + Polar Express + Cautious WD
                                 (muon.py)
          'megatron'           — faithful port of Megatron Muon (quintic NS,
                                 spectral scale, matched_adamw_rms=0.2), aligned
                                 with the PAI Muon reference (muon_megatron.py)

        Param routing when use_muon=True:
          Muon  ← attention {q,k,v,c}_proj, shared_expert {gate,up,down}_proj,
                  MoE expert {gate,up,down}_weight (3D batched).
          AdamW ← embeddings (wte, lm_head), MoE router gate, all 1D params (norms),
                  any biases.
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        if not use_muon:
            # Original alignment-baseline path: single AdamW with 2D-decay / 1D-nodecay split.
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == 'cuda'
            extra_args = dict(fused=True) if use_fused else dict()
            optimizer = torch.optim.AdamW(
                optim_groups, lr=learning_rate, betas=betas, eps=eps, **extra_args
            )
            print(f"using fused AdamW: {use_fused}")
            return optimizer

        # Muon path. Name-based routing (NOT dim-based: MoE 3D goes to Muon, embedding 2D
        # goes to AdamW). Within AdamW, 2D-decay vs 1D-nodecay matches the baseline path
        # so the only single-variable change vs baseline is which params get NorMuon.
        if muon_impl == 'megatron':
            from muon_megatron import Muon, MultiOptimizer
        elif muon_impl == 'normuon':
            from muon import Muon, MultiOptimizer
        elif muon_impl == 'megatron_v2':
            # nanogpt.optim pipeline-based Muon with muon_megatron recipe.
            # Must be bitwise-equivalent to 'megatron' under det=True.
            from nanogpt.optim import Muon as _NewMuon
            from nanogpt.optim import MultiOptimizer
            from nanogpt.optim.recipes import muon_megatron as _recipe_megatron

            def Muon(params, lr, momentum_beta, weight_decay, **kw):
                return _NewMuon(
                    params,
                    pipeline=_recipe_megatron(
                        momentum_beta=momentum_beta,
                        use_nesterov=kw.get("use_nesterov", True),
                        coefficient_type=kw.get("coefficient_type", "quintic"),
                        num_ns_steps=kw.get("num_ns_steps", 5),
                        muon_matched_adamw_rms=kw.get("muon_matched_adamw_rms", 0.2),
                        fp32_matmul_prec=kw.get("fp32_matmul_prec", "medium"),
                    ),
                    lr=lr,
                    weight_decay=weight_decay,
                )
        else:
            raise ValueError(f"unknown muon_impl {muon_impl!r} "
                             "(expected 'normuon', 'megatron', or 'megatron_v2')")
        print(f"Muon impl: {muon_impl}")

        def _to_adamw(name: str, param) -> bool:
            return (
                name.endswith('wte.weight')
                or name.endswith('lm_head.weight')
                or '.router.linear' in name
                or param.dim() < 2
            )

        muon_names, muon_params = [], []
        adam_decay_names, adam_decay_params = [], []
        adam_nodecay_names, adam_nodecay_params = [], []

        for n, p in param_dict.items():
            if _to_adamw(n, p):
                if p.dim() >= 2:
                    adam_decay_names.append(n)
                    adam_decay_params.append(p)
                else:
                    adam_nodecay_names.append(n)
                    adam_nodecay_params.append(p)
            else:
                muon_names.append(n)
                muon_params.append(p)

        # Sanity: every param accounted for exactly once
        seen = set(adam_decay_names) | set(adam_nodecay_names) | set(muon_names)
        missing = set(param_dict.keys()) - seen
        if missing:
            raise RuntimeError(f"unrouted params: {sorted(missing)}")
        if any('.router.linear' in n for n in muon_names):
            raise RuntimeError("MoE router weight wrongly routed to Muon")

        muon_lr_eff = muon_lr if muon_lr is not None else (learning_rate * 33.0)
        muon_wd_eff = muon_weight_decay if muon_weight_decay is not None else weight_decay

        muon_total = sum(p.numel() for p in muon_params)
        adam_dec_total = sum(p.numel() for p in adam_decay_params)
        adam_nodec_total = sum(p.numel() for p in adam_nodecay_params)
        print(f"Muon: {len(muon_params)} tensors, {muon_total:,} params, lr={muon_lr_eff}")
        print(f"AdamW (decay):   {len(adam_decay_params)} tensors, {adam_dec_total:,} params")
        print(f"AdamW (nodecay): {len(adam_nodecay_params)} tensors, {adam_nodec_total:,} params")
        if len(muon_params) <= 4:
            print(f"  Muon param names: {muon_names}")
        if len(adam_decay_params) <= 4:
            print(f"  AdamW decay names: {adam_decay_names}")

        adam_groups = [
            {'params': adam_decay_params, 'weight_decay': weight_decay},
            {'params': adam_nodecay_params, 'weight_decay': 0.0},
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        adam_extra = dict(fused=True) if use_fused else dict()
        adamw = torch.optim.AdamW(
            adam_groups, lr=learning_rate, betas=betas, eps=eps, **adam_extra
        )
        if muon_impl in ('megatron', 'megatron_v2'):
            # Megatron port (v1 or v2 pipeline): uses momentum_beta, coefficient_type, etc. No beta2.
            muon = Muon(
                muon_params,
                lr=muon_lr_eff,
                momentum_beta=muon_momentum,
                weight_decay=muon_wd_eff,
            )
        else:  # 'normuon'
            muon = Muon(
                muon_params,
                lr=muon_lr_eff,
                momentum=muon_momentum,
                beta2=muon_beta2,
                weight_decay=muon_wd_eff,
            )
        return MultiOptimizer({'adamw': adamw, 'muon': muon})

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        N = self.get_num_params()
        cfg = self.config
        head_dim = cfg.kv_channels if cfg.kv_channels is not None else cfg.n_embd // cfg.n_head
        L, H, Q, T = cfg.n_layer, cfg.n_head, head_dim, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 989e12  # H100 GPU bfloat16 peak flops
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


