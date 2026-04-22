"""Test: replace nano's MoEFFN with TE-based (GroupedLinear + moe_permute/unpermute)
for layer 1 (first MoE layer), forward on samples 28-31, compare block 1 output vs ref.

If L1 drops meaningfully, confirms MoE kernel diff is the remaining source.
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


class TEMoEFFN(nn.Module):
    def __init__(self, cfg, orig_moeffn):
        super().__init__()
        self.cfg = cfg
        self.num_experts = cfg.num_experts  # 144
        self.topk = cfg.moe_router_topk  # 8
        self.ffn_h = cfg.moe_ffn_hidden_size  # 160
        self.hidden = cfg.n_embd  # 512
        # Reuse nano's router + shared_expert
        self.router = orig_moeffn.router
        self.shared_expert = orig_moeffn.shared_expert
        # TE GroupedLinear: num_gemms=144, 512→320 (fc1=[gate;up]) then 160→512 (fc2)
        self.fc1 = te.GroupedLinear(
            num_gemms=self.num_experts,
            in_features=self.hidden,
            out_features=2 * self.ffn_h,  # gate+up fused
            bias=False,
            params_dtype=torch.float32,
            device='cuda',
        )
        self.fc2 = te.GroupedLinear(
            num_gemms=self.num_experts,
            in_features=self.ffn_h,
            out_features=self.hidden,
            bias=False,
            params_dtype=torch.float32,
            device='cuda',
        )

    def load_from_nano(self, orig):
        # orig.gate_weight: [E, C, H], up_weight: [E, C, H], down_weight: [E, H, C]
        # TE GroupedLinear stores weights as per-gemm [out_features, in_features]
        # For fc1, we want [2*H, C] per expert with [gate; up] concat
        # TE's convention: we register weight{i} parameters with [out, in] shape
        E, C, H = orig.gate_weight.shape  # [144, 512, 160]
        for e in range(E):
            # fc1 expert e: weight = concat([gate_e.T, up_e.T], dim=0) → [2H, C]
            gate_e = orig.gate_weight[e].T  # [H, C] (transpose since orig is [C, H])
            up_e = orig.up_weight[e].T      # [H, C]
            fc1_e = torch.cat([gate_e, up_e], dim=0).float()  # [2H, C]
            # fc2 expert e: weight = down_e.T → [C, H]
            fc2_e = orig.down_weight[e].T.float()  # [C, H]
            # TE stores as weight0, weight1, ... (one per gemm)
            getattr(self.fc1, f'weight{e}').data = fc1_e.cuda()
            getattr(self.fc2, f'weight{e}').data = fc2_e.cuda()

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(B * T, C)  # [S, C] where S = B*T
        # Shared expert (always on)
        shared_out = self.shared_expert(x_flat)
        # Router: get topk_idx [S, K], weights [S, K], raw_scores [S, E]
        topk_idx, weights, raw_scores = self.router(x_flat)
        S = x_flat.size(0)
        K = self.topk
        E = self.num_experts
        # Build routing_map [S, E] bool
        routing_map = torch.zeros(S, E, dtype=torch.bool, device=x.device)
        routing_map.scatter_(1, topk_idx, True)
        # Build probs [S, E] fp32 with scattered weights
        probs = torch.zeros(S, E, dtype=torch.float32, device=x.device)
        probs.scatter_(1, topk_idx, weights.float())

        # Count tokens per expert (m_splits for GroupedLinear)
        m_splits = routing_map.sum(dim=0).tolist()  # [E], each = # tokens routed to expert e
        num_out_tokens = int(sum(m_splits))
        # Permute (returns: permuted_tokens, row_id_map, permuted_probs)
        permuted, row_id_map, permuted_probs = te.moe_permute_with_probs(
            x_flat, probs, routing_map, num_out_tokens=num_out_tokens,
        )
        # fc1: [N, C] → [N, 2H]
        h = self.fc1(permuted, m_splits=m_splits)
        # SwiGLU
        gate, up = h.chunk(2, dim=-1)
        h = F.silu(gate) * up  # [N, H]
        # fc2: [N, H] → [N, C]
        out_permuted = self.fc2(h, m_splits=m_splits)
        # Unpermute with weighted sum (use merging_probs = permuted_probs from permute step)
        out = te.moe_unpermute(out_permuted, row_id_map, merging_probs=permuted_probs,
                                restore_shape=x_flat.shape)  # [S, C]
        return (shared_out + out).view(B, T, C), x.new_zeros(())


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

    # Swap layer 1's MoEFFN with TE-based version
    orig = model.transformer.h[1].mlp
    te_moe = TEMoEFFN(cfg, orig).to(device)
    te_moe.load_from_nano(orig)
    model.transformer.h[1].mlp = te_moe
    model.eval()

    arr = np.memmap(os.path.join(ROOT, 'data/cybertron_baseline/train.bin'), dtype=np.int32, mode='r')
    idx_np = np.stack([arr[s*8192:(s+1)*8192].astype(np.int64) for s in SAMPLE_IDS])
    idx = torch.from_numpy(idx_np).to(device)
    tgt_np = np.stack([arr[s*8192+1:(s+1)*8192+1].astype(np.int64) for s in SAMPLE_IDS])
    tgt = torch.from_numpy(tgt_np).to(device)

    captured = {}
    def make_hook(k):
        def fn(m, inp, o):
            x = o[0] if isinstance(o, tuple) else o
            captured[k] = x.detach().transpose(0, 1).cpu().contiguous()
        return fn
    handles = [model.transformer.h[i].register_forward_hook(make_hook(f'b{i}')) for i in range(9)]

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        _, loss = model(idx, targets=tgt)
    for h in handles: h.remove()
    print(f'nano+TE-MoE-layer1 loss: {loss.item():.6f}')

    print('\n=== Per-block diff (TE layer-1 MoE) ===')
    for i in range(9):
        ref = torch.load(f'{DUMP_DIR}/decoder.layers.{i}-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt',
                         weights_only=False, map_location='cpu')
        ref = ref[0] if isinstance(ref, tuple) else ref
        a = captured[f'b{i}'].float(); b = ref.float()
        d = (a - b).abs()
        print(f'block{i}: L∞={d.max().item():.4e} L1={d.mean().item():.4e} rel={d.mean().item()/b.abs().mean().clamp_min(1e-8).item():.4e}')


if __name__ == '__main__':
    main()
