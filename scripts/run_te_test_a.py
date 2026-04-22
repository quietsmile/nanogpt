"""Replace nano's attention (ln_1 + QKV + attn + c_proj) and dense MLP with TE fused ops
for ALL 9 layers. Keep MoE routed experts as nano (TE MoE integration incomplete).
Run Test A. See if end-to-end loss drops materially."""
import argparse
import os, sys, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import transformer_engine.pytorch as te

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


class TEAttn(nn.Module):
    """TE-fused attention: LayerNormLinear(LN+QKV) + RMSNorm(qk_ln) + RoPE + DotProductAttention + Linear(c_proj)"""
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head; self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.kv_channels
        self.n_rep = cfg.n_head // cfg.n_kv_head
        q_dim = cfg.n_head * self.head_dim
        k_dim = cfg.n_kv_head * self.head_dim
        v_dim = cfg.n_kv_head * self.head_dim
        # Fused LN+QKV using TE
        self.ln_qkv = te.LayerNormLinear(
            in_features=cfg.n_embd,
            out_features=q_dim + k_dim + v_dim,
            eps=cfg.norm_eps, bias=False, normalization='RMSNorm',
            params_dtype=torch.float32, device='cuda',
        )
        self.q_size = q_dim
        self.k_size = k_dim
        self.v_size = v_dim
        # qk_layernorm via TE RMSNorm
        self.q_ln = te.RMSNorm(self.head_dim, eps=cfg.norm_eps, params_dtype=torch.float32, device='cuda')
        self.k_ln = te.RMSNorm(self.head_dim, eps=cfg.norm_eps, params_dtype=torch.float32, device='cuda')
        # TE RoPE — precompute freqs
        inv_freq = 1.0 / (cfg.rotary_base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        t = torch.arange(cfg.block_size, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer('rope_freqs', torch.cat([freqs, freqs], dim=-1).view(cfg.block_size, 1, 1, self.head_dim), persistent=False)
        # TE attention
        self.dpa = te.DotProductAttention(
            num_attention_heads=cfg.n_head, kv_channels=self.head_dim,
            num_gqa_groups=cfg.n_kv_head, attention_dropout=0.0,
            qkv_format='bshd', attn_mask_type='causal',
        )
        # c_proj using TE Linear
        self.c_proj = te.Linear(q_dim, cfg.n_embd, bias=False,
                                 params_dtype=torch.float32, device='cuda')

    def load_from_nano(self, nano_attn, meg_ln_w):
        # ln_qkv.layer_norm_weight ← ln_1.weight (already loaded in nano from converter)
        self.ln_qkv.layer_norm_weight.data = meg_ln_w.float().cuda()
        # ln_qkv.weight ← concat nano q_proj.weight, k_proj.weight, v_proj.weight as [Q_all; K_all; V_all]
        qw = nano_attn.q_proj.weight.data.float()
        kw = nano_attn.k_proj.weight.data.float()
        vw = nano_attn.v_proj.weight.data.float()
        self.ln_qkv.weight.data = torch.cat([qw, kw, vw], dim=0).cuda()
        self.q_ln.weight.data = nano_attn.q_layernorm.weight.data.float().cuda()
        self.k_ln.weight.data = nano_attn.k_layernorm.weight.data.float().cuda()
        self.c_proj.weight.data = nano_attn.c_proj.weight.data.float().cuda()

    def forward(self, x, attn_mask=None, position_ids=None):
        B, T, C = x.size()
        qkv = self.ln_qkv(x)  # [B, T, q+k+v]
        q = qkv[..., :self.q_size].view(B, T, self.n_head, self.head_dim)
        k = qkv[..., self.q_size:self.q_size+self.k_size].view(B, T, self.n_kv_head, self.head_dim)
        v = qkv[..., self.q_size+self.k_size:].view(B, T, self.n_kv_head, self.head_dim)
        # qk_layernorm
        q = self.q_ln(q); k = self.k_ln(k)
        # TE RoPE (bshd format)
        from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb
        freqs = self.rope_freqs[:T]
        q = apply_rotary_pos_emb(q, freqs, tensor_format='bshd')
        k = apply_rotary_pos_emb(k, freqs, tensor_format='bshd')
        # Attention
        out = self.dpa(q, k, v)  # [B, T, n_head*head_dim]
        # c_proj
        return self.c_proj(out)


class TEDenseMLP(nn.Module):
    """TE fused LayerNormMLP for dense SwiGLU MLP."""
    def __init__(self, cfg):
        super().__init__()
        self.te_mlp = te.LayerNormMLP(
            hidden_size=cfg.n_embd,
            ffn_hidden_size=cfg.ffn_hidden_size,
            eps=cfg.norm_eps, bias=False, normalization='RMSNorm',
            activation='swiglu',
            params_dtype=torch.float32, device='cuda',
        )

    def load_from_meg(self, meg, layer_idx=0):
        self.te_mlp.layer_norm_weight.data = meg[f'decoder.layers.{layer_idx}.mlp.linear_fc1.layer_norm_weight'].float().cuda()
        self.te_mlp.fc1_weight.data = meg[f'decoder.layers.{layer_idx}.mlp.linear_fc1.weight'].float().cuda()
        self.te_mlp.fc2_weight.data = meg[f'decoder.layers.{layer_idx}.mlp.linear_fc2.weight'].float().cuda()

    def forward(self, x):
        return self.te_mlp(x)


def build_te_model(cfg, meg, device):
    from model import GPT
    model = GPT(cfg).to(device)
    # Load nano weights first
    from megatron_to_nano import convert
    sd = convert(meg)
    model.load_state_dict(sd, strict=False)

    # Now replace attention + block 0 MLP with TE versions
    for i, block in enumerate(model.transformer.h):
        te_attn = TEAttn(cfg).to(device)
        # ln_qkv weight: layer_norm_weight is in block.ln_1.weight
        te_attn.load_from_nano(block.attn, block.ln_1.weight.data)
        block.attn = te_attn
        # Replace ln_1 with identity since TE LayerNormLinear fuses it
        block.ln_1 = nn.Identity()

        if i == 0:
            # Dense MLP → TE LayerNormMLP
            te_mlp_b0 = TEDenseMLP(cfg).to(device)
            te_mlp_b0.load_from_meg(meg, layer_idx=0)
            block.mlp = te_mlp_b0
            block.ln_2 = nn.Identity()
            # Override forward to handle TEDenseMLP returning non-tuple
            # Actually block.forward calls `block.mlp(mlp_in)` and for MoE expects tuple (out, aux)
            # For dense it should return out directly
            # Block's forward has: if self.is_moe: mlp_out, aux = self.mlp(...) else: mlp_out, aux = self.mlp(mlp_in), x.new_zeros(())
            # is_moe False for block 0, so it's mlp(mlp_in) returning one tensor. OK.
        # For MoE blocks 1-8, keep nano MoEFFN but replace ln_2 with TE RMSNorm
        else:
            te_ln2 = te.RMSNorm(cfg.n_embd, eps=cfg.norm_eps, params_dtype=torch.float32, device=device)
            te_ln2.weight.data = block.ln_2.weight.data.float().cuda()
            block.ln_2 = te_ln2

    return model


def forward_on_batch(model, sample_indices, data_mmap):
    device = next(model.parameters()).device
    raw = model.module if hasattr(model, 'module') else model
    raw.eval()
    losses = []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for sid in sample_indices:
            idx = torch.from_numpy(np.array(data_mmap[sid*8192:(sid+1)*8192].astype(np.int64))).unsqueeze(0).to(device)
            tgt = torch.from_numpy(np.array(data_mmap[sid*8192+1:(sid+1)*8192+1].astype(np.int64))).unsqueeze(0).to(device)
            _, loss = model(idx, targets=tgt)
            losses.append(loss.item())
    return sum(losses) / len(losses) if losses else 0.0


def main():
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if world > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
    device = torch.device('cuda', local_rank)

    from megatron_to_nano import load_all_megatron_shards

    arr = np.memmap(os.path.join(ROOT, 'data/cybertron_baseline/train.bin'), dtype=np.int32, mode='r')

    import re
    ref_losses = {}
    for line in open('/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/logs/rank-0-1-scaling_moe_00196-run.log'):
        m = re.search(r'iteration\s+(\d+)/.*lm loss:\s*([\d.eE+-]+)', line)
        if m: ref_losses[int(m.group(1))] = float(m.group(2))

    pairs = [(1497, 1498), (2994, 2995), (4491, 4492), (5988, 5989)]

    if rank == 0:
        print(f'[TE-replace Test A] 4 matched pairs')
        print(f'{"ckpt":>6}{"batch":>7}{"nano_fwd":>12}{"ref_logged":>12}{"Δ":>10}')

    _, cfg = build_nano()
    results = []
    for ckpt_iter, next_iter in pairs:
        ref_dir = f'/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/iter_{ckpt_iter:07d}'
        meg = load_all_megatron_shards(ref_dir)
        model = build_te_model(cfg, meg, device)
        all_samples = list(range((next_iter - 1) * 64, next_iter * 64))
        my_samples = [all_samples[i] for i in range(rank, 64, world)]
        local_loss = forward_on_batch(model, my_samples, arr)
        t = torch.tensor(local_loss, device=device)
        if world > 1:
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
        ref_loss = ref_losses.get(next_iter)
        if rank == 0:
            delta = t.item() - ref_loss
            print(f'{ckpt_iter:>6}{next_iter:>7}{t.item():>12.6f}{ref_loss:>12.6f}{delta:>+10.6f}')
            results.append(delta)
        del model
        torch.cuda.empty_cache()

    if rank == 0 and results:
        import statistics
        print(f'\n[TE-replace Test A] avg Δ = {statistics.mean(results):+.6f}')
        print(f'stddev = {statistics.stdev(results):.6f}')

    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
