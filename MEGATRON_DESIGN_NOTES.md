# Megatron design observations during nano↔cybertron alignment

Observations about cybertron/Megatron design choices that may be sources of
imprecision or potential improvement opportunities, collected while tracing
per-sublayer diff to 0.0001 nat target.

## 1. TE `LayerNormLinear` uses a different cuBLAS kernel than `F.linear`

- `F.linear` and `te.Linear` are bitwise equal (same cuBLAS path)
- `te.LayerNormLinear` (fused LN + Linear) uses a different cuBLAS algorithm
  (likely cuBLASLt with specific tile), producing different bf16 output than
  `F.linear(te.RMSNorm(x))` despite identical math
- **Impact**: fused vs unfused LayerNormLinear give different bf16 results, so
  tensor-parallel or pipeline-parallel split/replicate of LayerNormLinear
  introduces non-bitwise deviation

## 2. `te.RMSNorm` differs from both nano's Python impl AND `torch.nn.functional.rms_norm`

- TE RMSNorm is a custom CUDA kernel with its own bf16 rounding pattern
- Not bitwise-equal to any Python-level `x * rsqrt(mean(x^2) + eps) * weight`
- `torch.rms_norm` (PyTorch built-in) also differs from TE RMSNorm
- **Impact**: cross-framework LN output mismatch, even when math is identical

## 3. Python `F.silu(x) * y` double-rounds under bf16 autocast

- `F.silu(gate)` returns bf16 under autocast bf16 (rounds once)
- Then `* up` rounds again
- TE's fused `SwiGLU` kernel does `silu(g) * u` in fp32 and rounds once
- Nano fix: `(F.silu(g.float()) * u.float()).to(bf16)` matches TE to 289/50M positions
- **Suggestion**: Python-side SwiGLU should always use fp32 intermediate when
  target dtype is bf16, to match fused TE kernel

## 4. MoE `unpermute`: fused vs non-fused precision

- Megatron's NON-fused `unpermute` does:
  `permuted_tokens * permuted_probs` → fp32 (widest rule, since probs is fp32)
  `output.scatter_add_(...)` in fp32
  cast to bf16 at end
- Megatron's FUSED path (TE `fused_unpermute`) may do this entirely in bf16
  internally for performance
- **Impact**: switching `moe_permute_fusion` flag changes per-layer bf16 output
  (potentially inconsistent across runs with different flag values)

## 5. Router non-bias-scaled weights for gate application

- `scores_for_choice = scores + e_score_correction_bias` used for topk selection
- But `final_weights` use UNBIASED `scores.gather(topk_idx)`, then normalize
- This is documented (DeepSeek-V3 aux-free formula) and matches intent
- Subtle: the bias is an additive offset, so using biased or unbiased scores
  for gate weights gives materially different loss. Megatron impl correctly
  uses unbiased.

## 6. `accurate_attn_mask_with_cp=False` with `accurate_attn_mask_eod_token=[151643]`

- Config suggests EOD-aware attention (cu_seqlens from EOD tokens)
- But `with_cp=False` means the EOD-aware path is NOT used in `default_batch_process_func`
- **Confusing**: config has `accurate_attn_mask_eod_token` set but not actually used
  unless context parallel is enabled
- Ref attention is plain causal (confirmed: EOD-aware mask gave L1=3.68e-3 vs
  plain causal L1=1.72e-5 when comparing to ref)

## 7. bf16 flash-attn-2 kernel differences across versions

- PyTorch SDPA flash backend == TE `DotProductAttention` (both bitwise equal)
- But both differ from ref's core_attention dump by L1=1.72e-5 (block 0) to
  L1=1.22e-3 (block 1)
- Likely reason: different TE/flash-attn-2/cuDNN versions between training
  and inference environments
- **Amplification**: attention output diff gets amplified ~35x through
  `ln_2 + fc1 + silu + fc2` → 6.3e-4 final block 0 diff
- Block-by-block diff grows with attn logit magnitude (softmax
  exponential sensitivity)

## 8. RoPE: nano vs TE RoPE are bitwise-equal after format fix

- TE `apply_rotary_pos_emb` defaults to `tensor_format='sbhd'`
- If you pass `bhsd` tensors, output is mathematically different (the sequence
  dim becomes the batch dim in the rotation)
- **Documentation gap**: TE RoPE requires careful tensor_format attention;
  easy to miscall producing catastrophically wrong outputs

## Summary

The dominant source of per-layer bf16 drift in deep MoE models is:
1. TE fused kernels (LayerNormLinear, SwiGLU, flash-attn-2) each produce
   slightly different bf16 output than their decomposed Python equivalents
2. Fused-vs-nonfused paths within Megatron itself may differ slightly
3. Version differences of TE/flash-attn-2/cuDNN amplify through layers

For full bf16 bitwise reproduction, all inference must use the EXACT same
TE/cuDNN kernels as training.
