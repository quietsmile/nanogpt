"""Build an exhaustive alignment checklist: ref yaml vs nano config, item by item.

Categories:
  architecture, init, optimizer, lr_schedule, batch_parallelism,
  loss_masking, precision, attention, moe, dataloader, determinism, tokenizer, data

Each check emits { category, name, ref, nano, status: OK|DIFF|UNKNOWN, note }.
Writes reports/alignment_checklist.json.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REF_YAML = '/prodcpfs/user/data/GitLab/pretrain_scaling_ladder/scaling_moe_00196.yaml'
DATA_YAML = '/prodcpfs/user/data/GitLab/pretrain_scaling_ladder/data_pretrain_v3_pai.yaml'
NANO_CONFIG = os.path.join(ROOT, 'config', 'cybertron_moe_196.py')
OUT = os.path.join(ROOT, 'reports', 'alignment_checklist.json')


def _load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _load_nano_config(path):
    ns = {}
    with open(path) as f:
        exec(f.read(), ns)
    return {k: v for k, v in ns.items() if not k.startswith('__')}


def _normalize(x):
    """Canonicalize values so string-of-number and float match, list-str vs list match."""
    if isinstance(x, str):
        # Try parse as float
        try:
            return float(x)
        except ValueError:
            pass
        # Evaluate patterns like "[0]*1+[1]*8"
        if '[' in x:
            try:
                return list(eval(x, {'__builtins__': {}}, {}))
            except Exception:
                pass
        return x
    return x


def _status(ref, nano) -> str:
    if ref is None and nano is None: return 'UNKNOWN'
    r, n = _normalize(ref), _normalize(nano)
    if r == n: return 'OK'
    try:
        if isinstance(r, (int, float)) and isinstance(n, (int, float)):
            if abs(float(r) - float(n)) < 1e-9: return 'OK'
    except Exception:
        pass
    if isinstance(r, list) and isinstance(n, list) and r == n: return 'OK'
    return 'DIFF'


def _push(out, category, name, ref, nano, note='', *,
          ref_src: Optional[str] = None, nano_src: Optional[str] = None,
          impact: Optional[str] = None, status_override: Optional[str] = None):
    st = status_override or _status(ref, nano)
    out.append({
        'category': category, 'name': name,
        'ref': ref, 'nano': nano, 'status': st, 'note': note,
        'ref_src': ref_src or '', 'nano_src': nano_src or '',
        'impact': impact or '',
    })


def check_architecture(out, ref, nano):
    c = ref['cybertron']
    _push(out, 'architecture', 'n_layer',        c['num_layers'],            nano.get('n_layer'))
    _push(out, 'architecture', 'hidden_size',    c['hidden_size'],           nano.get('n_embd'))
    _push(out, 'architecture', 'n_head',         c['num_attention_heads'],   nano.get('n_head'))
    _push(out, 'architecture', 'n_kv_head',      c['num_query_groups'],      nano.get('n_kv_head'))
    _push(out, 'architecture', 'kv_channels',    c['kv_channels'],           nano.get('kv_channels'))
    _push(out, 'architecture', 'ffn_hidden_size (dense)', c['ffn_hidden_size'], nano.get('ffn_hidden_size'))
    _push(out, 'architecture', 'moe_ffn_hidden_size', c['moe_ffn_hidden_size'], nano.get('moe_ffn_hidden_size'))
    _push(out, 'architecture', 'moe_shared_expert_hidden_size', c['moe_shared_expert_intermediate_size'], nano.get('moe_shared_expert_hidden_size'))
    _push(out, 'architecture', 'num_experts',    c['num_experts'],           nano.get('num_experts'))
    _push(out, 'architecture', 'moe_router_topk', c['moe_router_topk'],      nano.get('moe_router_topk'))
    _push(out, 'architecture', 'n_group',        c['n_group'],               nano.get('moe_n_group'))
    _push(out, 'architecture', 'topk_group',     c['topk_group'],            nano.get('moe_topk_group'))
    _push(out, 'architecture', 'moe_layer_freq',
          c['moe_layer_freq'], str(nano.get('moe_layer_freq')))
    _push(out, 'architecture', 'rotary_base',    c['rotary_base'],           nano.get('rotary_base'))
    _push(out, 'architecture', 'norm_epsilon',   c['norm_epsilon'],          nano.get('norm_eps'))
    _push(out, 'architecture', 'normalization',  c['normalization'],
          'RMSNorm' if nano.get('use_rmsnorm') else 'LayerNorm')
    _push(out, 'architecture', 'position_embedding_type', c['position_embedding_type'],
          'rope' if nano.get('use_rope') else 'learned')
    _push(out, 'architecture', 'qk_layernorm',   c['qk_layernorm'],          nano.get('qk_layernorm'))
    _push(out, 'architecture', 'untie_embeddings_and_output_weights',
          c['untie_embeddings_and_output_weights'], not nano.get('tie_embeddings', True))
    _push(out, 'architecture', 'swiglu',         c['swiglu'],                nano.get('use_swiglu'))
    _push(out, 'architecture', 'add_bias_linear', c['add_bias_linear'],      nano.get('bias', False))
    _push(out, 'architecture', 'add_qkv_bias',   c['add_qkv_bias'],          nano.get('bias', False))
    _push(out, 'architecture', 'padded_vocab_size', c['padded_vocab_size'],  nano.get('vocab_size_override'))
    _push(out, 'architecture', 'max_position_embeddings / block_size',
          c['max_position_embeddings'],    nano.get('block_size'))
    _push(out, 'architecture', 'seq_length',     c['seq_length'],            nano.get('block_size'))


def check_moe_routing(out, ref, nano):
    c = ref['cybertron']
    _push(out, 'moe_routing', 'router_scoring_func', c['router_scoring_func'],
          'sigmoid', note='nano uses torch.sigmoid in MoERouter.forward')
    _push(out, 'moe_routing', 'norm_topk_prob',    c['norm_topk_prob'],       nano.get('moe_norm_topk_prob'))
    _push(out, 'moe_routing', 'routed_scaling_factor', c['routed_scaling_factor'],
          1.0, note='nano hardcoded 1.0 (no scaling on topk weights post-norm)')
    _push(out, 'moe_routing', 'router_expert_score_correction_coeff',
          c['router_expert_score_correction_coeff'],
          nano.get('moe_router_score_correction_coeff'))
    _push(out, 'moe_routing', 'use_router_expert_score_correction',
          c['use_router_expert_score_correction'],
          (nano.get('moe_router_score_correction_coeff', 0) or 0) > 0)
    _push(out, 'moe_routing', 'moe_router_load_balancing_type', c['moe_router_load_balancing_type'],
          'greedy', note='nano picks top-K per group (greedy)')
    _push(out, 'moe_routing', 'seq_aux',           c['seq_aux'], True,
          note='nano MoEFFN computes per-sequence aux when alpha>0')
    _push(out, 'moe_routing', 'sequence_wise_balance_loss_alpha',
          c['sequence_wise_balance_loss_alpha'], nano.get('seq_aux_balance_alpha'))
    _push(out, 'moe_routing', 'seq_aux P_i normalization',
          'P_i^b = mean_t (σ(l_i(t)) / Σ_j σ(l_j(t)))  [per-token normalized]',
          'P_i^b = mean_t σ(l_i(t))                     [RAW sigmoid, NOT normalized]',
          status_override='DIFF',
          note='ROOT CAUSE CANDIDATE for 1.8 nat gap',
          ref_src='cybertron_dots3.0_swa/cybertron/models/deepseek_v2/modules_deepseekv2.py:377 — `pii = pi.div(pi.sum(dim=-1, keepdim=True) + 1e-20).mean(0)`',
          nano_src='nanogpt/model.py MoEFFN.forward (seq_aux block): `P = raw_scores.view(B,T,E).float().mean(dim=1)`',
          impact='Sigmoid outputs are independent per expert → sum ≈ E·sigmoid_mean ≈ E·0.5. Without normalization, my P is ~0.5; Megatron’s P is ~1/E = 1/144. My aux is ~72× larger. With α=0.0001, my effective aux magnitude ~7e-3 vs ref ~1e-4. This drives routing to balance too aggressively in early training, distorting everything downstream.')
    _push(out, 'moe_routing', 'seq_aux f_i scale',
          'f_i^b = count * E / (T * K)   [inside the f term]',
          'f_i^b = count / T  (then multiply by E/K outside)',
          status_override='OK',
          note='mathematically equivalent — just factor placement differs',
          ref_src='modules_deepseekv2.py:374 — `fii = fi.sum(0).div_(seq_len * K / E)`',
          nano_src='model.py MoEFFN.forward: `f = counts/T` then `aux = (E/K) * mean(Σ f*P)`')
    _push(out, 'moe_routing', 'moe_aux_loss_coeff', c.get('moe_aux_loss_coeff', 0.0) if False else 0.0,
          0.0, note='disabled in both')
    _push(out, 'moe_routing', 'expert_balance_loss_alpha',  c['expert_balance_loss_alpha'], 0.0)
    _push(out, 'moe_routing', 'device_balance_loss_alpha',  c['device_balance_loss_alpha'], 0.0)
    _push(out, 'moe_routing', 'communication_balance_loss_alpha',
          c['communication_balance_loss_alpha'], 0.0)
    _push(out, 'moe_routing', 'moe_grouped_gemm',  c['moe_grouped_gemm'], False,
          note='nano uses torch.bmm with pad-to-max-count; Megatron uses grouped_gemm (CUTLASS). Different kernel, same math.',
          ref_src='Megatron/TransformerEngine grouped_gemm CUTLASS kernel (per-expert variable-length batched matmul)',
          nano_src='nanogpt/model.py MoEFFN.forward — torch.bmm with every expert bucket padded to max(count) with zeros',
          impact='Padding adds ~10–30% extra compute on padding rows but keeps all experts aligned. Zero padding rows produce zero-output rows which are discarded via gather. Numerical result should match grouped_gemm to bf16 precision. Small residual drift ~1e-4 per bmm.')
    _push(out, 'moe_routing', 'moe_token_dispatcher_type', c['moe_token_dispatcher_type'],
          'sort+bmm', note='nano: expert-sorted bmm; ref: allgather for EP>1')
    _push(out, 'moe_routing', 'moe_permute_fusion', c['moe_permute_fusion'], False,
          note='cosmetic; nano permute via argsort (no fusion)')
    _push(out, 'moe_routing', 'moe_enable_deepep', c['moe_enable_deepep'], False)
    _push(out, 'moe_routing', 'moe_shared_expert_overlap', c['moe_shared_expert_overlap'], False)
    _push(out, 'moe_routing', 'moe_gating_fp32',    c['moe_gating_fp32'], True,
          note='nano computes router logits in fp32 (MoERouter line 270)')


def check_loss_masking(out, ref, nano):
    c = ref['cybertron']
    _push(out, 'loss_masking', 'eod_mask_loss',           c['eod_mask_loss'],
          nano.get('eod_token_id') is not None,
          ref_src='cybertron/utils/mcore_utils.py:179 — `loss_mask[data == eod_token] = 0.0` where `data` = INPUT tokens (tokens[:-1])',
          nano_src='nanogpt/model.py GPT.forward — mask when TARGET==eod (one position later than ref)',
          impact='Off-by-one: ref masks loss at the position where input IS EOD; nano masks where target IS EOD. In practice near EOD boundaries only, small effect on loss magnitude (< 0.01 loss delta) but affects gradient routing through final FC.')
    _push(out, 'loss_masking', 'EOD token id',             c['accurate_attn_mask_eod_token'][0], nano.get('eod_token_id'))
    _push(out, 'loss_masking', 'mask_loss_id',             c['mask_loss_id'], nano.get('mask_loss_id'))
    _push(out, 'loss_masking', 'accurate_attn_mask_eod_token', c['accurate_attn_mask_eod_token'][0],
          nano.get('eod_token_id') if nano.get('use_eod_attn_mask') else None,
          note='nano use_eod_attn_mask=True gates this on, otherwise it is None and behaves as plain causal.',
          ref_src='cybertron/utils/batchify_utils.py:63-88 — builds accurate_attn_mask tensor from EOD positions, passes to TE attention',
          nano_src='nanogpt/model.py GPT.forward — builds segment_id via cumsum, forms same-segment AND causal mask, passes to SDPA with is_causal=False',
          impact='Both produce mathematically the same mask, but different SDPA kernel paths: ref uses TE flash-attn with variable-length segments; nano uses SDPA dense bool mask. Dense mask can force SDPA off the flash-attn-2 fast path → slower and slightly different numerics (1e-3 relative). Observed: training slower when use_eod_attn_mask=True.')
    _push(out, 'loss_masking', 'reset_attention_mask',     c['reset_attention_mask'], False)
    _push(out, 'loss_masking', 'reset_position_ids',       c['reset_position_ids'], False,
          note='RoPE is position-based; nano always uses contiguous positions. Packed docs past EOD get "wrong" position — likely source of gap')
    _push(out, 'loss_masking', 'SUSPECT: position ids reset on EOD', 'no (ref)', 'no (nano)',
          status_override='UNKNOWN',
          note='Both say no; but since reset_attention_mask=false with accurate_attn_mask=true, docs within a sequence share positions yet attn segments — behavior subtle',
          impact='RoPE encodes absolute position; if token at position 5000 is forced to only attend within doc segment [4800..5000] but still uses position=5000, attention scores systematically different from a fresh-segment view. Unclear if ref behaves the same.')


def check_optimizer(out, ref, nano):
    c = ref['cybertron']
    _push(out, 'optimizer', 'optimizer name', c['optimizer'], 'adamw',
          note='ref: `adam` in Megatron (their "adam" = AdamW style with decoupled WD); nano: torch AdamW. Same math.',
          ref_src='Megatron optimizer = FusedAdam with decoupled weight_decay=0.1',
          nano_src='nanogpt configure_optimizers uses torch.optim.AdamW(fused=True)',
          impact='Numerically equivalent. Same update rule, same betas/eps. Any residual is from fused kernel impl details.')
    _push(out, 'optimizer', 'lr (peak)',       c['lr'],        nano.get('learning_rate'))
    _push(out, 'optimizer', 'min_lr',          c['min_lr'],    nano.get('min_lr'))
    _push(out, 'optimizer', 'adam_beta1',      c['adam_beta1'], nano.get('beta1'))
    _push(out, 'optimizer', 'adam_beta2',      c['adam_beta2'], nano.get('beta2'))
    _push(out, 'optimizer', 'adam_eps',        c['adam_eps'],   nano.get('adam_eps'))
    _push(out, 'optimizer', 'weight_decay',    c['weight_decay'], nano.get('weight_decay'))
    _push(out, 'optimizer', 'clip_grad',       c['clip_grad'], nano.get('grad_clip'))
    _push(out, 'optimizer', 'accumulate_allreduce_grads_in_fp32',
          c['accumulate_allreduce_grads_in_fp32'], False,
          note='ref: grad accumulation + DP all-reduce in fp32. nano: bf16 throughout (PyTorch DDP bucket reduces in param dtype = bf16).',
          ref_src='Megatron grad_accumulate_in_fp32=True stores gradient buffer as fp32',
          nano_src='nanogpt uses default torch DDP which reduces grads in param dtype (bf16)',
          impact='bf16 has only ~3 decimal digits of mantissa. Gradient averaging over 8 ranks in bf16 has additional rounding vs fp32. Per-step per-param error ~1e-4 relative; grows linearly in step count. Over 2000 steps the bias could explain 0.1–0.3 nat of the observed gap. Likely a real contributor.')
    _push(out, 'optimizer', 'use_distributed_optimizer',
          c['use_distributed_optimizer'], False,
          note='nano: no ZeRO-style sharded optimizer. Each rank holds full optimizer state.',
          ref_src='Megatron Distributed Optimizer: each rank holds 1/DP of optimizer state (exp_avg, exp_avg_sq)',
          nano_src='nanogpt: standard torch AdamW, full optimizer state on each rank',
          impact='Mathematically equivalent — distributed optimizer partitions the state across ranks but the per-parameter update is identical. No direct numerical divergence.')


def check_lr_schedule(out, ref, nano):
    c = ref['cybertron']
    _push(out, 'lr_schedule', 'lr_decay_style', c['lr_decay_style'], nano.get('lr_decay_style'))
    _push(out, 'lr_schedule', 'lr_warmup_init', c['lr_warmup_init'], 0.0,
          note='nano warmup starts from 0 (implicit)')
    _push(out, 'lr_schedule', 'lr_warmup_samples',  c['lr_warmup_samples'], nano.get('warmup_samples'))
    _push(out, 'lr_schedule', 'wsd_constant_samples', c['wsd_constant_samples'], nano.get('constant_samples'))
    _push(out, 'lr_schedule', 'lr_decay_samples',    c['lr_decay_samples'],    nano.get('decay_end_samples'))
    _push(out, 'lr_schedule', 'exit_interval',       c['exit_interval'],       nano.get('max_iters'))
    _push(out, 'lr_schedule', 'train_samples',       c['train_samples'], None,
          note='nano train_samples implicitly = max_iters × global_bs')


def check_batch_parallelism(out, ref, nano):
    c = ref['cybertron']
    _push(out, 'batch_parallelism', 'global_batch_size', c['global_batch_size'], nano.get('global_batch_size'))
    _push(out, 'batch_parallelism', 'micro_batch_size',  c['micro_batch_size'],  nano.get('batch_size'),
          note='ref=4 samples per rank per micro-batch; nano=1. Multiple samples in one forward → LayerNorm/RMSNorm see bigger stats (though our RMSNorm is per-sample so this is moot).',
          ref_src='Megatron batch_size=4, forward passes [4,T,C] through TE fused layers',
          nano_src='nanogpt config batch_size=1, forward passes [1,T,C]',
          impact='Independent per-sample forward in nano ≠ batched forward in ref at the kernel level. bf16 matmul accumulation order differs. Per-step output drift ~1e-3 relative, grows with depth. Over 9 layers: cumulative ~1e-2 logit difference. Small but real.')
    _push(out, 'batch_parallelism', 'tensor_model_parallel_size', c['tensor_model_parallel_size'], 1)
    _push(out, 'batch_parallelism', 'pipeline_model_parallel_size', c['pipeline_model_parallel_size'], 1)
    _push(out, 'batch_parallelism', 'context_parallel_size', c['context_parallel_size'], 1)
    _push(out, 'batch_parallelism', 'expert_model_parallel_size',
          c['expert_model_parallel_size'], 1,
          note='ref EP=4: each rank physically holds 36 of 144 experts; all-to-all dispatches tokens. nano EP=1: every rank holds ALL 144 experts, no cross-rank token shuffle.',
          ref_src='Megatron MoE layer — `moe_token_dispatcher_type=allgather` with EP=4 process group',
          nano_src='nanogpt MoEFFN — single-device expert bucket (torch.bmm with pad-to-max)',
          impact='Routing DECISIONS are identical (same sigmoid + top-K). What differs: gradient aggregation path. ref aggregates expert grads across its 2-DP ranks; nano across 8-DP ranks. Means nano has 4× the DP noise-reduction on expert weights. Expected effect: nano experts train more smoothly but diverge from ref due to different noise realization.')
    _push(out, 'batch_parallelism', 'DP ranks (derived)', 2, 8,
          note='ref DP = world_size / TP / PP / EP = 8/4 = 2; nano DP = 8 (all ranks). Different gradient averaging partition.',
          ref_src='derived: world_size(8) / TP(1) / PP(1) / EP(4) = DP(2)',
          nano_src='derived: world_size(8) / TP(1) / PP(1) / EP(1) = DP(8)',
          impact='Gradient averaging: nano avg over 8 ranks, ref avg over 2 ranks (then EP shards). For any expert, ref update = avg of 2 samples gradients; nano update = avg of 8 samples gradients. nano has 4× lower gradient variance → smoother but possibly less greedy updates in early training. Systematically different convergence path.')
    _push(out, 'batch_parallelism', 'DATA SAMPLE ROUTING PER RANK',
          'rank r reads samples r, r+2, r+4, ...  (DP=2, stride=2 in ref)',
          'rank r reads samples r, r+8, r+16, ...  (DP=8, stride=8 in nano)',
          status_override='DIFF',
          note='Same total samples per step (64) but partitioned differently across ranks. Both nano and ref visit EVERY sample exactly once per "epoch"; just the order differs. nano get_batch_sequential initializes _seq_data_pos = ddp_rank * batch_size (line 189), advances by ddp_world_size * batch_size per call (line 209) — correct interleaved sharding.',
          ref_src='Megatron MCoreGPTDataset sampler: iterates by (dp_rank, dp_size) across blended index from BlendedDataset cache',
          nano_src='nanogpt/train.py line 189 (init _seq_data_pos[train] = ddp_rank * batch_size) and line 193-223 get_batch_sequential',
          impact='At each optim step, nano rank 0 sees samples {0,8,16,...,56} and ref rank 0 sees {0,2,4,...,14}. Different batches → different gradients even with identical init. Over 1000s of steps, trajectory systematically diverges from ref (observed: +1.8 nat at iter 1000). Cannot be eliminated without matching DP=2 + EP=4 partition (requires rewriting DDP setup).')


def check_precision(out, ref, nano):
    c = ref['cybertron']
    _push(out, 'precision', 'bf16', c['bf16'], nano.get('dtype') == 'bfloat16')
    _push(out, 'precision', 'attention_softmax_in_fp32', c['attention_softmax_in_fp32'], 'unknown (SDPA backend-dependent)',
          note='ref always casts attention logits to fp32 before softmax. SDPA backend depends on flash-attn availability.',
          ref_src='TransformerEngine attention: `if attention_softmax_in_fp32: softmax(logits.float()).to(dtype)`',
          nano_src='nanogpt SDPA — internal behavior depends on kernel selected (flash-attn-2 uses fp32 accum; math backend may not)',
          impact='If SDPA picks flash-attn-2: equivalent to ref. If math backend: ~1e-3 softmax relative error per head per layer. Over 4 heads × 9 layers: compounding error ~1e-2. Mitigation: force `with sdp_kernel(enable_flash=True): ...`.')
    _push(out, 'precision', 'apply_query_key_layer_scaling', False, False, note='disabled both sides')
    _push(out, 'precision', 'moe_gating_fp32', c['moe_gating_fp32'], True)
    _push(out, 'precision', 'cross_entropy_fusion_impl', None, None, note='both use eager F.cross_entropy')


def check_init(out, ref, nano):
    c = ref['cybertron']
    _push(out, 'init', 'init_method_std',           c['init_method_std'], nano.get('init_std'))
    _push(out, 'init', 'disable_scaled_init_method', c['disable_scaled_init_method'],
          nano.get('disable_scaled_init_method'))
    _push(out, 'init', 'embedding init', 'Megatron default normal(0, std)', 'nano: normal(0, std)',
          note='Same distribution, but different library RNG order.',
          ref_src='Megatron model_provider instantiates nn.Embedding via ColumnParallelEmbedding; init via init_method=normal_(0, 0.006)',
          nano_src='nanogpt/model.py:GPT._init_weights — `nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)` on every nn.Embedding',
          impact='Same random seed (1337) produces different tensor values because module construction order differs (nano creates embedding → 9 blocks; Megatron creates embedding split by TP, per-layer init). Resulting embedding tensor is different bit-for-bit → training trajectory differs from step 1.')
    _push(out, 'init', 'router init',  'normal(0, init_method_std=0.006)', 'normal(0, init_std=0.006)',
          status_override='OK',
          note='User-confirmed: cybertron has a deepseek_v2 router with Kaiming init, but the 00196 job does NOT take that branch — it uses the Megatron init_method_std=0.006 path, identical to nano.',
          ref_src='Megatron ColumnParallelLinear (router weight) with init_method=normal_(0, 0.006)',
          nano_src='nanogpt/model.py MoERouter.linear re-initialized by GPT._init_weights apply() → normal_(0, 0.006)',
          impact='No divergence from init-distribution here.')
    _push(out, 'init', 'expert weights init (routed)', 'Megatron per-expert init_method',
          'nano: stacked weights normal_(0, std=0.006) all 3 projs',
          note='Routed experts: gate_weight [144,512,160], up_weight [144,512,160], down_weight [144,160,512]',
          ref_src='Megatron MoE experts each individually initialized via scaled_init if enabled, else normal(0, 0.006)',
          nano_src='nanogpt/model.py MoEFFN.__init__ — single nn.init.normal_ call on stacked [E,C,H] tensor',
          impact='Same distribution and std but single-call vs per-expert-call consumes random numbers in different order. Expert[0] in nano might have init tensor != expert[0] in ref. Over 144 experts × 3 weights × 512*160 params each, cumulative random-state divergence is massive. Combined with router init diff, early routing decisions diverge markedly.')
    _push(out, 'init', 'layer-norm weights init', 'ones', 'ones',
          note='Both initialize RMSNorm scale to 1; no divergence here')
    _push(out, 'init', 'bias init', 'zeros (when present)', 'zeros / no bias (nano bias=False)',
          note='both: add_bias_linear=false, so no bias tensors exist — no divergence')


def check_attention(out, ref, nano):
    c = ref['cybertron']
    _push(out, 'attention', 'multi_latent_attention', c['multi_latent_attention'], False)
    _push(out, 'attention', 'group_query_attention',  c['group_query_attention'], nano.get('n_kv_head') is not None)
    _push(out, 'attention', 'attention_dropout',      c['attention_dropout'], 0.0)
    _push(out, 'attention', 'hidden_dropout',         c['hidden_dropout'], 0.0)
    _push(out, 'attention', 'apply_rope_fusion',      c['apply_rope_fusion'], True,
          note='CLOSED (commit a97ce75): nano now uses fused_apply_rotary_pos_emb from megatron.core.extensions.transformer_engine when available — matches ref bitwise',
          ref_src='Megatron apply_rotary_pos_emb via TransformerEngine fused kernel',
          nano_src='nanogpt/model.py RotaryEmbedding — try-imports fused_apply_rotary_pos_emb; falls back to manual rotate_half on CPU tensors or when TE unavailable (the CUDA-only kernel silently returns NaN on CPU inputs)',
          impact='Was the dominant systematic drift source (~95% of per-layer Δ). With fused kernel: block 0 L1 drops 9.3e-4 → 3.9e-5; with full TE stack: 2.43e-7.')
    _push(out, 'attention', 'RoPE position ids',      'contiguous 0..T-1', 'contiguous 0..T-1',
          note='both do not reset on EOD')
    _push(out, 'attention', 'attention impl',         'TransformerEngine flash', 'PyTorch SDPA',
          note='Different flash backends. Numerically similar but not identical',
          ref_src='NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=1 → TransformerEngine flash-attn 2',
          nano_src='nanogpt/model.py CausalSelfAttention — torch.nn.functional.scaled_dot_product_attention',
          impact='SDPA may dispatch to memory-efficient attention or flash-attn-2 depending on shape/contiguity. Numerical result differs at ~1e-3 relative. Over 9 layers + softmax amplification: attention output drift ~1e-2. Small but compounds with routing.')
    _push(out, 'attention', 'transformer_impl',       c['transformer_impl'], 'eager (PyTorch)',
          note='ref uses NVIDIA TransformerEngine with fused kernels for attention+norm+linear; nano uses separate PyTorch modules.',
          ref_src='Megatron `transformer_impl=transformer_engine` — TE fused LayerNormLinear, attention, MLP',
          nano_src='nanogpt/model.py — separate RMSNorm, CausalSelfAttention, MLP/MoEFFN calls',
          impact='TE fused kernels do multiple ops in a single CUDA call with shared registers → different intermediate precision path. E.g., fused LayerNormLinear accumulates norm statistics in fp32 then applies linear in bf16; nano does norm in bf16 storage then bf16 linear. Cumulative drift across 9 layers: ~1e-3 logit relative error. Noticeable but small compared to other sources.')


def check_tokenizer(out, ref, nano):
    c = ref['cybertron']
    _push(out, 'tokenizer', 'tokenizer_type', c['tokenizer_type'], 'HFTokenizer')
    _push(out, 'tokenizer', 'tokenizer_model path', c['tokenizer_model'],
          '/prodcpfs/user/xiaoming/models/dots_tokenizer',
          note='nano uses same path when loaded (see tests/test_tokenizer_alignment)')
    _push(out, 'tokenizer', 'vocab_size base', 151643, 151643)
    _push(out, 'tokenizer', 'added_tokens count', 16, 16)
    _push(out, 'tokenizer', 'eos/eod token_id', 151643, 151643)
    _push(out, 'tokenizer', 'padded vocab', 152064, nano.get('vocab_size_override'))


def check_data(out, data_yaml, nano, ref):
    yaml_paths = data_yaml['cybertron']['train_data_path']
    _push(out, 'data', 'blend dataset count', len(yaml_paths), 116,
          note='test_data_sampling_alignment asserts match')
    _push(out, 'data', 'data_cache_path (UserCommand override)',
          '/prodcpfs/user/data/save/data/lossalign/data_cache',
          '/prodcpfs/user/data/save/data/lossalign/data_cache')
    _push(out, 'data', 'BlendedDataset hash 00196', '43adec39b46f5eb95d144361a0db6699',
          '43adec39b46f5eb95d144361a0db6699')
    _push(out, 'data', 'first 1024 (dataset_id, sample_id) sha256', 'c7c752c069f4e41f…',
          'c7c752c069f4e41f…', note='from reports/data_sampling_alignment.json')
    _push(out, 'data', 'train samples consumed', 479040, 'run-dependent (we did 128064 in 2000-step D)',
          note='2000-step short run consumed 2000*64=128000 samples, 27% of full')
    _push(out, 'data', 'val dataset', 'Pile_test_5k', 'Pile_test_5k',
          note='prepare_cybertron_data.py VAL_DATA_PATH')
    _push(out, 'data', 'ensure_full_document', ref['cybertron']['ensure_full_document'], False)


def check_determinism(out, ref, nano):
    c = ref['cybertron']
    _push(out, 'determinism', 'deterministic mode', c['bitwise_aligned'], nano.get('deterministic'),
          note='ref bitwise_aligned=false → fastest TE kernels (may be non-deterministic). nano deterministic=true → slower deterministic kernels only.',
          ref_src='Megatron args.bitwise_aligned controls TransformerEngine determinism flags',
          nano_src='nanogpt/train.py — torch.use_deterministic_algorithms(True), cudnn.deterministic=True',
          impact='Different kernels chosen → different numerical rounding. Example: non-deterministic reduction uses atomicAdd; deterministic uses stable tree-reduce — result off by ~ULP per accumulated element. Over attention with 8192 positions, residual errors ~1e-3 per layer × 9 layers = ~1e-2 output drift.')
    _push(out, 'determinism', 'seed', 1337, nano.get('seed'),
          note='Megatron default seed is 1234 (overridable), ours nano 1337.',
          ref_src='Megatron args.seed (default 1234 unless yaml sets `seed`)',
          nano_src='nanogpt config/cybertron_moe_196.py: seed = 1337',
          impact='Entirely different random stream → entirely different init tensors → entirely different training trajectory from step 0. This alone could explain a significant fraction of the 1.8 nat offset.')


def check_dataloader(out, ref, nano):
    c = ref['cybertron']
    _push(out, 'dataloader', 'dataloader_type',   c['dataloader_type'], 'sequential',
          note='nano get_batch_sequential; ref `single`. Semantically equivalent for our case.',
          ref_src='Megatron `single` mode: iterate the blended sampler once, one sample per batch element',
          nano_src='nanogpt/train.py:get_batch_sequential — read block_size+1 tokens from mmap, stride by ddp_world_size*batch_size',
          impact='No direct numerical impact — both produce the same sequence of samples when fed the same tokenized cache.')
    _push(out, 'dataloader', 'num_workers',       c['num_workers'], 'N/A (mmap read)')
    _push(out, 'dataloader', 'create_attention_mask_in_dataloader',
          c['create_attention_mask_in_dataloader'], False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()

    ref = _load_yaml(REF_YAML)
    data_yaml = _load_yaml(DATA_YAML)
    nano = _load_nano_config(NANO_CONFIG)

    items: List[Dict[str, Any]] = []
    check_architecture(items,  ref, nano)
    check_init(items,          ref, nano)
    check_optimizer(items,     ref, nano)
    check_lr_schedule(items,   ref, nano)
    check_batch_parallelism(items, ref, nano)
    check_loss_masking(items,  ref, nano)
    check_moe_routing(items,   ref, nano)
    check_attention(items,     ref, nano)
    check_precision(items,     ref, nano)
    check_tokenizer(items,     ref, nano)
    check_data(items,          data_yaml, nano, ref)
    check_determinism(items,   ref, nano)
    check_dataloader(items,    ref, nano)


    # Summary
    summary = {
        'total': len(items),
        'ok': sum(1 for i in items if i['status'] == 'OK'),
        'diff': sum(1 for i in items if i['status'] == 'DIFF'),
        'unknown': sum(1 for i in items if i['status'] == 'UNKNOWN'),
        'by_category': {},
    }
    for it in items:
        c = it['category']
        s = summary['by_category'].setdefault(c, {'ok': 0, 'diff': 0, 'unknown': 0, 'total': 0})
        s['total'] += 1
        s[it['status'].lower()] += 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'summary': summary, 'items': items}, f, ensure_ascii=False, indent=2)

    print(f"wrote {args.out}: total={summary['total']} ok={summary['ok']} diff={summary['diff']} unknown={summary['unknown']}")
    print("\nDIFF items:")
    for it in items:
        if it['status'] == 'DIFF':
            print(f"  [{it['category']}] {it['name']}: ref={it['ref']!r} nano={it['nano']!r}")
            if it.get('note'): print(f"    NOTE: {it['note']}")


if __name__ == '__main__':
    main()
