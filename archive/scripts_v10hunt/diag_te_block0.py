"""Integrate TE LayerNormLinear for block-0's pre-attention and pre-MLP.
Load ref weights into TE modules, forward on samples 28-31, compare block 0 output.

If block 0's L∞ drops from 0.125 → ~0, TE-fused kernels confirmed as the
source of the remaining +0.014 nat residual.
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class TEBlock0Attn(nn.Module):
    """Replaces nano's block0 (ln_1 + q/k/v/c_proj + ln_2 + gate/up/down + q_layernorm/k_layernorm)
    with TE's fused LayerNormLinear(QKV) + qk_layernorm + attn + c_proj + LayerNormMLP."""
    def __init__(self, cfg, orig_block):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.kv_channels
        self.n_rep = cfg.n_head // cfg.n_kv_head
        qkv_dim = (cfg.n_head + 2 * cfg.n_kv_head) * self.head_dim  # q+k+v
        # Fused pre-attn LN + QKV linear
        self.ln_qkv = te.LayerNormLinear(
            in_features=cfg.n_embd,
            out_features=qkv_dim,
            eps=cfg.norm_eps,
            bias=False,
            normalization='RMSNorm',
            params_dtype=torch.float32,
            device='cuda',
        )
        self.q_layernorm = orig_block.attn.q_layernorm  # keep nano's RMSNorm
        self.k_layernorm = orig_block.attn.k_layernorm
        self.rotary_emb = orig_block.attn.rotary_emb
        self.c_proj = orig_block.attn.c_proj
        # Fused pre-mlp LN + MLP (SwiGLU)
        # TE LayerNormMLP with activation='swiglu' does: ln(x) @ linear_fc1 → [gate; up] → silu(gate)*up → linear_fc2
        self.ln_mlp = te.LayerNormMLP(
            hidden_size=cfg.n_embd,
            ffn_hidden_size=cfg.ffn_hidden_size,
            eps=cfg.norm_eps,
            bias=False,
            normalization='RMSNorm',
            activation='swiglu',
            params_dtype=torch.float32,
            device='cuda',
        )

    def load_from_ref(self, meg):
        # LN weight → TE LayerNormLinear's layer_norm_weight
        self.ln_qkv.layer_norm_weight.data = meg['decoder.layers.0.self_attention.linear_qkv.layer_norm_weight'].float().cuda()
        # De-interleave Megatron GQA QKV into nano's [Q_all, K_all, V_all] layout via split_qkv
        from megatron_to_nano import split_qkv
        q, k, v = split_qkv(meg['decoder.layers.0.self_attention.linear_qkv.weight'])
        # TE's Linear weight shape is [out, in]. We want output = [Q_all | K_all | V_all]
        qkv_contig = torch.cat([q, k, v], dim=0).float().cuda()
        self.ln_qkv.weight.data = qkv_contig

        # TE LayerNormMLP for dense MLP
        self.ln_mlp.layer_norm_weight.data = meg['decoder.layers.0.mlp.linear_fc1.layer_norm_weight'].float().cuda()
        # Megatron's linear_fc1 is [gate; up] fused — TE LayerNormMLP with activation='swiglu' expects same layout
        self.ln_mlp.fc1_weight.data = meg['decoder.layers.0.mlp.linear_fc1.weight'].float().cuda()
        self.ln_mlp.fc2_weight.data = meg['decoder.layers.0.mlp.linear_fc2.weight'].float().cuda()

    def forward(self, x, attn_mask=None, position_ids=None):
        # x: [B, T, C] in nano's format; TE expects [T, B, C] or [B, T, C]
        B, T, C = x.shape
        # TE LayerNormLinear accepts bshd implicit — just feed x
        qkv = self.ln_qkv(x)  # [B, T, Q+K+V]
        q_end = self.n_head * self.head_dim
        k_end = q_end + self.n_kv_head * self.head_dim
        q = qkv[..., :q_end].view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = qkv[..., q_end:k_end].view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = qkv[..., k_end:].view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        if self.q_layernorm is not None:
            q = self.q_layernorm(q)
            k = self.k_layernorm(k)
        q, k = self.rotary_emb(q, k, seq_len=T)
        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(B, self.n_kv_head, self.n_rep, T, self.head_dim).reshape(B, self.n_head, T, self.head_dim)
            v = v.unsqueeze(2).expand(B, self.n_kv_head, self.n_rep, T, self.head_dim).reshape(B, self.n_head, T, self.head_dim)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, -1)
        x = x + self.c_proj(attn)
        # Fused pre-MLP LN + MLP
        mlp_out = self.ln_mlp(x)  # [B, T, C]
        x = x + mlp_out
        return x, x.new_zeros(())  # (x, aux) to match nano Block interface


def main():
    CKPT_DIR = '/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/iter_0005988'
    DUMP_DIR = '/newcpfs/user/yuchen/karpathy/cybertron_dump/dumps'
    SAMPLE_IDS = [5988*64 + s for s in [28, 29, 30, 31]]

    device = torch.device('cuda:0')
    from megatron_to_nano import load_all_megatron_shards, convert
    meg = load_all_megatron_shards(CKPT_DIR)
    sd = convert(meg)
    model, cfg = build_nano()
    model = model.to(device)
    model.load_state_dict(sd, strict=False)

    # Replace block 0 with TE-fused version
    orig_block0 = model.transformer.h[0]
    te_block0 = TEBlock0Attn(cfg, orig_block0).to(device)
    te_block0.load_from_ref(meg)
    # Swap
    model.transformer.h[0] = te_block0
    model.eval()

    arr = np.memmap(os.path.join(ROOT, 'data/cybertron_baseline/train.bin'), dtype=np.int32, mode='r')
    idx_np = np.stack([arr[s*8192:(s+1)*8192].astype(np.int64) for s in SAMPLE_IDS])
    idx = torch.from_numpy(idx_np).to(device)

    captured = {f'block{i}': None for i in range(9)}
    captured['ln_f'] = None
    handles = []
    for i, block in enumerate(model.transformer.h):
        def make_hook(idx):
            def fn(m, inp, o):
                x = o[0] if isinstance(o, tuple) else o
                captured[f'block{idx}'] = x.detach().transpose(0, 1).cpu().contiguous()
            return fn
        handles.append(block.register_forward_hook(make_hook(i)))
    handles.append(model.transformer.ln_f.register_forward_hook(
        lambda m, inp, o: captured.update({'ln_f': o.detach().transpose(0, 1).cpu().contiguous()})))

    tgt_np = np.stack([arr[s*8192+1:(s+1)*8192+1].astype(np.int64) for s in SAMPLE_IDS])
    tgt = torch.from_numpy(tgt_np).to(device)

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        _, loss = model(idx, targets=tgt)
    for h in handles: h.remove()
    print(f'nano+TE-block0 loss: {loss.item():.6f}')

    # Compare every block's output vs ref
    print('\n=== Per-block diff with TE block 0 ===')
    print(f'{"block":<10}{"L∞":>10}{"L1":>12}{"rel_mean":>12}')
    for i in range(9):
        ref = torch.load(f'{DUMP_DIR}/decoder.layers.{i}-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt',
                         weights_only=False, map_location='cpu')
        ref = ref[0] if isinstance(ref, tuple) else ref
        a = captured[f'block{i}'].float()
        b = ref.float()
        d = (a - b).abs()
        rel = d.mean() / b.abs().mean().clamp_min(1e-8)
        print(f'block{i:<5}{d.max().item():>10.4e}{d.mean().item():>12.4e}{rel.item():>12.4e}')
    ref_f = torch.load(f'{DUMP_DIR}/decoder.final_layernorm-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt',
                       weights_only=False, map_location='cpu')
    ref_f = ref_f[0] if isinstance(ref_f, tuple) else ref_f
    a = captured['ln_f'].float()
    b = ref_f.float()
    d = (a - b).abs()
    rel = d.mean() / b.abs().mean().clamp_min(1e-8)
    print(f'{"ln_f":<10}{d.max().item():>10.4e}{d.mean().item():>12.4e}{rel.item():>12.4e}')


if __name__ == '__main__':
    main()
