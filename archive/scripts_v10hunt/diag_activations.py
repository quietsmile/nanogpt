"""Instrument nano forward to match ref's per-sublayer activation stats.

Ref logs (per iter, averaged across 9 layers):
  act_std/decoder_input, act_*/attn_input, act_*/attn_output,
  act_*/attn_plus_residual, act_*/ffn_input, act_*/ffn_output,
  act_*/ffn_plus_residual, act_*/final_input,
  max_attn_logits/full, first_token_attn_score/full, attn_entropy_mean/full,
  sequence_wise_balance_loss, tokens_per_expert/{min,max,mean}.

We load ref iter_5988 ckpt into nano, forward on iter-5989 batch, compare
our measured stats against ref's logged 5989 values. The sublayer where
stats first diverge materially is the source of the +0.014 nat.
"""
import os, sys, re
import numpy as np
import torch
import torch.distributed as dist

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPT_DIR)


def build_nano():
    from model import GPTConfig, GPT
    cfg = GPTConfig(
        block_size=8192, vocab_size=152064, n_layer=9, n_head=4, n_embd=512,
        n_kv_head=2, kv_channels=64, dropout=0.0, bias=False, init_std=0.006,
        use_rope=True, rotary_base=50000, use_rmsnorm=True, norm_eps=1e-5,
        use_swiglu=True, ffn_hidden_size=1536,
        qk_layernorm=True, tie_embeddings=False, disable_scaled_init_method=True,
        use_moe=True, moe_layer_freq=[0] + [1] * 8, num_experts=144,
        moe_ffn_hidden_size=160, moe_router_topk=8, moe_n_group=8, moe_topk_group=1,
        moe_norm_topk_prob=True, moe_router_score_correction_coeff=0.001,
        moe_shared_expert_hidden_size=160, moe_routing_type='greedy',
        eod_token_id=151643, mask_loss_id=160000,
        seq_aux_balance_alpha=0.0001, use_eod_attn_mask=False,
    )
    return GPT(cfg), cfg


class Accum:
    def __init__(self):
        self.sum_std = 0.0; self.sum_mean = 0.0; self.max = -1e30; self.n = 0
    def add(self, x):
        xf = x.detach().float()
        self.sum_std += xf.std().item()
        self.sum_mean += xf.mean().item()
        self.max = max(self.max, xf.abs().max().item())
        self.n += 1
    def stats(self):
        n = max(self.n, 1)
        return self.sum_std/n, self.sum_mean/n, self.max


def forward_with_hooks(model, idx, tgt):
    raw = model.module if hasattr(model, 'module') else model
    stats = {k: Accum() for k in ['decoder_input', 'attn_input', 'attn_output',
                                   'attn_plus_residual', 'ffn_input', 'ffn_output',
                                   'ffn_plus_residual', 'final_input']}
    attn_logit_max = []
    attn_first_token_score = []
    attn_entropy = []
    per_layer_tokens_per_expert = []
    q_stats = []
    k_stats = []
    q_post_rope_stats = []
    k_post_rope_stats = []

    # patch Block.forward to emit hook values
    from model import Block, CausalSelfAttention, MoEFFN
    orig_block_fwd = Block.forward
    orig_attn_fwd = CausalSelfAttention.forward
    orig_moe_fwd = MoEFFN.forward

    def patched_block_fwd(self, x, attn_mask=None, position_ids=None):
        # attn
        h = self.ln_1(x)
        stats['attn_input'].add(h)
        a = self.attn(h, attn_mask=attn_mask, position_ids=position_ids)
        stats['attn_output'].add(a)
        x = x + a
        stats['attn_plus_residual'].add(x)
        # ffn
        mi = self.ln_2(x)
        stats['ffn_input'].add(mi)
        if self.is_moe:
            mo, aux = self.mlp(mi)
        else:
            mo, aux = self.mlp(mi), x.new_zeros(())
        stats['ffn_output'].add(mo)
        x = x + mo
        stats['ffn_plus_residual'].add(x)
        return x, aux

    def patched_attn_fwd(self, x, attn_mask=None, position_ids=None):
        # Compute manually to grab attn_logits / scores / entropy before SDPA
        import math
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        if self.q_layernorm is not None:
            q = self.q_layernorm(q)
            k = self.k_layernorm(k)
        q_stats.append((q.float().std().item(), q.float().abs().max().item()))
        k_stats.append((k.float().std().item(), k.float().abs().max().item()))
        q, k = self.rotary_emb(q, k, seq_len=T, position_ids=position_ids)
        q_post_rope_stats.append((q.float().std().item(), q.float().abs().max().item()))
        k_post_rope_stats.append((k.float().std().item(), k.float().abs().max().item()))
        # DEBUG probe: scale q by 0.5 to halve attn logits and see if it changes loss
        if os.environ.get('DEBUG_HALVE_Q') == '1':
            q = q * 0.5
        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(B, self.n_kv_head, self.n_rep, T, self.head_dim).reshape(B, self.n_head, T, self.head_dim)
            v = v.unsqueeze(2).expand(B, self.n_kv_head, self.n_rep, T, self.head_dim).reshape(B, self.n_head, T, self.head_dim)
        att = (q.float() @ k.float().transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        # before softmax: max logits
        attn_logit_max.append(att.abs().max().item())
        causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
        att_masked = att.masked_fill(~causal, float('-inf'))
        probs = torch.softmax(att_masked, dim=-1)
        # first-token attn score (averaged over heads+tokens, tokens after pos 0)
        attn_first_token_score.append(probs[..., 1:, 0].mean().item())
        # entropy
        ent = -(probs * torch.log(probs.clamp_min(1e-20))).sum(-1).mean().item()
        attn_entropy.append(ent)
        y = (probs @ v.float()).to(x.dtype)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)

    def patched_moe_fwd(self, x):
        # Force router to accumulate token_per_expert by enabling grad context;
        # we still use no_grad outer wrapper so nothing trains.
        # The router's counter path is gated on torch.is_grad_enabled(), so
        # we temporarily re-enable here.
        with torch.enable_grad():
            # reset before this layer's count
            self.router.local_tokens_per_expert.zero_()
            out, aux = orig_moe_fwd(self, x)
        tpe = self.router.local_tokens_per_expert.detach().clone()
        per_layer_tokens_per_expert.append(tpe)
        return out, aux

    Block.forward = patched_block_fwd
    CausalSelfAttention.forward = patched_attn_fwd
    MoEFFN.forward = patched_moe_fwd

    # Capture pre-ln_f input (= last layer's ffn_plus_residual) via hook on ln_f
    captured_final_input = {}
    def ln_f_pre_hook(module, inputs):
        captured_final_input['v'] = inputs[0].detach().float()
    h_handle = raw.transformer.ln_f.register_forward_pre_hook(ln_f_pre_hook)

    # Capture lm_head output logits
    captured_logits = {}
    def lm_head_hook(module, inputs, output):
        captured_logits['v'] = output.detach().float()
    lh_handle = raw.lm_head.register_forward_hook(lm_head_hook)

    try:
        tok_emb = raw.transformer.wte(idx)
        stats['decoder_input'].add(tok_emb)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, loss = raw(idx, targets=tgt)
        if 'v' in captured_final_input:
            stats['final_input'].add(captured_final_input['v'])
    finally:
        h_handle.remove()
        lh_handle.remove()
        Block.forward = orig_block_fwd
        CausalSelfAttention.forward = orig_attn_fwd
        MoEFFN.forward = orig_moe_fwd

    # Logits stats
    logits_stats = {}
    if 'v' in captured_logits:
        lo = captured_logits['v']
        logits_stats = {
            'logits_std': lo.std().item(),
            'logits_max': lo.abs().max().item(),
            'logits_mean': lo.mean().item(),
        }

    return stats, {
        'loss': loss.item(),
        'max_attn_logits': max(attn_logit_max) if attn_logit_max else 0,
        'first_token_attn_score': np.mean(attn_first_token_score) if attn_first_token_score else 0,
        'attn_entropy': np.mean(attn_entropy) if attn_entropy else 0,
        'tpe_max': max((t.max().item() for t in per_layer_tokens_per_expert), default=0),
        'tpe_min': min((t.min().item() for t in per_layer_tokens_per_expert), default=0),
        'tpe_mean': np.mean([t.float().mean().item() for t in per_layer_tokens_per_expert]) if per_layer_tokens_per_expert else 0,
        'q_std_mean': np.mean([s[0] for s in q_stats]) if q_stats else 0,
        'q_max_mean': np.mean([s[1] for s in q_stats]) if q_stats else 0,
        'k_std_mean': np.mean([s[0] for s in k_stats]) if k_stats else 0,
        'k_max_mean': np.mean([s[1] for s in k_stats]) if k_stats else 0,
        'q_rope_std_mean': np.mean([s[0] for s in q_post_rope_stats]) if q_post_rope_stats else 0,
        'k_rope_std_mean': np.mean([s[0] for s in k_post_rope_stats]) if k_post_rope_stats else 0,
        **logits_stats,
    }


def main():
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if world > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
    device = torch.device('cuda', local_rank)

    from megatron_to_nano import load_all_megatron_shards, convert
    arr = np.memmap(os.path.join(ROOT, 'data/cybertron_baseline/train.bin'), dtype=np.int32, mode='r')

    meg = load_all_megatron_shards('/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/iter_0005988')
    sd = convert(meg)
    model, _ = build_nano()
    model = model.to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    # iter 5989 batch = samples 5988*64 .. 5989*64-1 = 383232..383295
    all_samples = list(range(5988 * 64, 5989 * 64))
    my_samples = [all_samples[i] for i in range(rank, 64, world)]

    agg_stats = {k: Accum() for k in ['decoder_input', 'attn_input', 'attn_output',
                                       'attn_plus_residual', 'ffn_input', 'ffn_output',
                                       'ffn_plus_residual', 'final_input']}
    agg_extra = {'loss': 0, 'max_attn_logits': 0, 'first_token_attn_score': 0,
                 'attn_entropy': 0, 'tpe_max': 0, 'tpe_min': 1e18, 'tpe_mean': 0,
                 'q_std_mean': 0, 'q_max_mean': 0, 'k_std_mean': 0, 'k_max_mean': 0,
                 'q_rope_std_mean': 0, 'k_rope_std_mean': 0,
                 'logits_std': 0, 'logits_max': 0, 'logits_mean': 0, 'n': 0}

    for sid in my_samples:
        idx = torch.from_numpy(np.array(arr[sid*8192:(sid+1)*8192].astype(np.int64))).unsqueeze(0).to(device)
        tgt = torch.from_numpy(np.array(arr[sid*8192+1:(sid+1)*8192+1].astype(np.int64))).unsqueeze(0).to(device)
        stats, extra = forward_with_hooks(model, idx, tgt)
        for k, a in stats.items():
            agg_stats[k].sum_std += a.sum_std / max(a.n, 1)
            agg_stats[k].sum_mean += a.sum_mean / max(a.n, 1)
            agg_stats[k].max = max(agg_stats[k].max, a.max)
            agg_stats[k].n += 1
        for k in ['loss', 'max_attn_logits', 'first_token_attn_score', 'attn_entropy', 'tpe_mean',
                  'q_std_mean', 'q_max_mean', 'k_std_mean', 'k_max_mean', 'q_rope_std_mean', 'k_rope_std_mean',
                  'logits_std', 'logits_max', 'logits_mean']:
            agg_extra[k] += extra.get(k, 0)
        agg_extra['tpe_max'] = max(agg_extra['tpe_max'], extra['tpe_max'])
        agg_extra['tpe_min'] = min(agg_extra['tpe_min'], extra['tpe_min'])
        agg_extra['n'] += 1

    # Reduce across ranks: sum then avg
    def avg_tensor(v):
        t = torch.tensor(v, device=device, dtype=torch.float64)
        if world > 1:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return (t / world).item()

    if rank == 0:
        # Ref values at iter 5989:
        ref = {
            'loss': 3.057512,
            'act_std/decoder_input': 0.0624,
            'act_std/attn_input': 2.0093, 'act_mean/attn_input': 0.0329, 'act_max/attn_input': 284.24,
            'max_attn_logits/full': 14.011,
            'first_token_attn_score/full': 1.518e-4,
            'attn_entropy_mean/full': 4.2311,
            'act_std/attn_output': 0.6750, 'act_max/attn_output': 10.954,
            'act_std/attn_plus_residual': 2.2250, 'act_mean/attn_plus_residual': 0.0312, 'act_max/attn_plus_residual': 283.71,
            'act_std/ffn_input': 0.6363, 'act_mean/ffn_input': 0.00445, 'act_max/ffn_input': 6.661,
            'act_std/ffn_output': 0.9734, 'act_mean/ffn_output': 0.00361, 'act_max/ffn_output': 82.35,
            'act_std/ffn_plus_residual': 2.6712, 'act_mean/ffn_plus_residual': 0.0348, 'act_max/ffn_plus_residual': 316.94,
            'act_std/final_input': 2.2129, 'act_mean/final_input': 2.499e-4,
            'tokens_per_expert/max': 1275.76, 'tokens_per_expert/min': 62.39, 'tokens_per_expert/mean': 455.11,
        }
        print(f'{"hook":<36} {"nano":>14} {"ref":>14} {"Δ":>12}')
        print('-' * 80)

    nstd, nmean, nmax = {}, {}, {}
    for k in agg_stats:
        nstd[k] = avg_tensor(agg_stats[k].sum_std / max(agg_stats[k].n, 1))
        nmean[k] = avg_tensor(agg_stats[k].sum_mean / max(agg_stats[k].n, 1))
        nmax[k] = avg_tensor(agg_stats[k].max)
    ne_keys = ['loss', 'max_attn_logits', 'first_token_attn_score', 'attn_entropy', 'tpe_mean',
               'q_std_mean', 'q_max_mean', 'k_std_mean', 'k_max_mean', 'q_rope_std_mean', 'k_rope_std_mean',
               'logits_std', 'logits_max', 'logits_mean']
    ne = {k: avg_tensor(agg_extra[k] / max(agg_extra['n'], 1)) for k in ne_keys}
    ne['tpe_max'] = avg_tensor(agg_extra['tpe_max'])
    ne['tpe_min'] = agg_extra['tpe_min']
    # tpe_min: take min across ranks
    tmin = torch.tensor(agg_extra['tpe_min'], device=device, dtype=torch.float64)
    if world > 1:
        dist.all_reduce(tmin, op=dist.ReduceOp.MIN)
    ne['tpe_min'] = tmin.item()

    if rank == 0:
        for k in ['decoder_input', 'attn_input', 'attn_output', 'attn_plus_residual',
                  'ffn_input', 'ffn_output', 'ffn_plus_residual', 'final_input']:
            for stat_kind, vals in [('std', nstd), ('mean', nmean), ('max', nmax)]:
                ref_key = f'act_{stat_kind}/{k}'
                if ref_key in ref:
                    nano_v = vals[k]
                    delta = nano_v - ref[ref_key]
                    pct = 100 * delta / (abs(ref[ref_key]) + 1e-12)
                    print(f'act_{stat_kind}/{k:<25} {nano_v:>14.6f} {ref[ref_key]:>14.6f} {delta:>+12.6f}  ({pct:+.2f}%)')
        print()
        print(f'{"loss":<36} {ne["loss"]:>14.6f} {ref["loss"]:>14.6f} {ne["loss"]-ref["loss"]:>+12.6f}')
        print(f'{"max_attn_logits/full":<36} {ne["max_attn_logits"]:>14.6f} {ref["max_attn_logits/full"]:>14.6f} {ne["max_attn_logits"]-ref["max_attn_logits/full"]:>+12.6f}')
        print(f'{"first_token_attn_score/full":<36} {ne["first_token_attn_score"]:>14.6e} {ref["first_token_attn_score/full"]:>14.6e}')
        print(f'{"attn_entropy_mean/full":<36} {ne["attn_entropy"]:>14.6f} {ref["attn_entropy_mean/full"]:>14.6f} {ne["attn_entropy"]-ref["attn_entropy_mean/full"]:>+12.6f}')
        print(f'{"tokens_per_expert/max":<36} {ne["tpe_max"]:>14.6f} {ref["tokens_per_expert/max"]:>14.6f}')
        print(f'{"tokens_per_expert/min":<36} {ne["tpe_min"]:>14.6f} {ref["tokens_per_expert/min"]:>14.6f}')
        print(f'{"tokens_per_expert/mean":<36} {ne["tpe_mean"]:>14.6f} {ref["tokens_per_expert/mean"]:>14.6f}')
        print()
        print(f'{"q post-qk_ln std":<36} {ne["q_std_mean"]:>14.6f}  (per-layer avg)')
        print(f'{"q post-qk_ln max":<36} {ne["q_max_mean"]:>14.6f}')
        print(f'{"k post-qk_ln std":<36} {ne["k_std_mean"]:>14.6f}')
        print(f'{"k post-qk_ln max":<36} {ne["k_max_mean"]:>14.6f}')
        print(f'{"q post-RoPE std":<36} {ne["q_rope_std_mean"]:>14.6f}')
        print(f'{"k post-RoPE std":<36} {ne["k_rope_std_mean"]:>14.6f}')
        print()
        print(f'{"lm_head logits std":<36} {ne["logits_std"]:>14.6f}')
        print(f'{"lm_head logits max|abs|":<36} {ne["logits_max"]:>14.6f}')
        print(f'{"lm_head logits mean":<36} {ne["logits_mean"]:>14.6e}')

    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
