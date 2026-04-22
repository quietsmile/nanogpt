"""Compare nano's per-layer forward outputs against ref's bitwise dumps.
Ref's rank-7 mbs0 saw samples [28, 29, 30, 31] of iter-5989 global batch.
We forward nano on these 4 samples and diff per-block outputs."""
import os, sys
import numpy as np
import torch

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


def tensor_diff(a, b, name):
    a = a.float()
    b = b.float()
    assert a.shape == b.shape, f'{name}: {a.shape} vs {b.shape}'
    d = (a - b).abs()
    # Relative to ref's scale
    rel = d.mean() / b.abs().mean().clamp_min(1e-8)
    print(f'  {name} shape={list(a.shape)}: L∞={d.max():.4e} L1={d.mean():.4e} rel_mean={rel:.4e}')


def main():
    DUMP_DIR = '/newcpfs/user/yuchen/karpathy/cybertron_dump/dumps'
    CKPT_DIR = '/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/iter_0005988'

    # Ref mbs0 sees samples 28,29,30,31 of iter-5989 batch
    SAMPLE_IDS = [5988*64 + s for s in [28, 29, 30, 31]]

    device = torch.device('cuda:0')
    from megatron_to_nano import load_all_megatron_shards, convert
    meg = load_all_megatron_shards(CKPT_DIR)
    sd = convert(meg)
    model, _ = build_nano()
    model = model.to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    arr = np.memmap(os.path.join(ROOT, 'data/cybertron_baseline/train.bin'), dtype=np.int32, mode='r')
    # Build batch of 4 samples
    idx_np = np.stack([arr[s*8192:(s+1)*8192].astype(np.int64) for s in SAMPLE_IDS])  # [4, 8192]
    idx = torch.from_numpy(idx_np).to(device)  # [4, 8192]

    # Hook nano to capture per-sublayer tensors
    from model import Block, CausalSelfAttention
    captured = {'embed': None, 'blocks': [None]*9, 'ln_f': None, 'lm_head': None,
                'L0_ln1_out': None, 'L0_attn_out': None, 'L0_x_plus_attn': None,
                'L0_ln2_out': None, 'L0_mlp_out': None}

    def hook_block(i):
        def fn(mod, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            captured['blocks'][i] = x.detach().transpose(0, 1).cpu().contiguous()
        return fn

    handles = []
    for i, block in enumerate(model.transformer.h):
        handles.append(block.register_forward_hook(hook_block(i)))
    def hf_hook(mod, inp, out):
        captured['ln_f'] = out.detach().transpose(0, 1).cpu().contiguous()
    handles.append(model.transformer.ln_f.register_forward_hook(hf_hook))

    # Sub-layer hooks on block 0
    b0 = model.transformer.h[0]
    handles.append(b0.ln_1.register_forward_hook(
        lambda m, i, o: captured.update({'L0_ln1_out': o.detach().transpose(0,1).cpu().contiguous()})))
    handles.append(b0.attn.register_forward_hook(
        lambda m, i, o: captured.update({'L0_attn_out': o.detach().transpose(0,1).cpu().contiguous()})))
    handles.append(b0.ln_2.register_forward_hook(
        lambda m, i, o: captured.update({'L0_ln2_out': o.detach().transpose(0,1).cpu().contiguous()})))
    handles.append(b0.mlp.register_forward_hook(
        lambda m, i, o: captured.update({'L0_mlp_out': (o[0] if isinstance(o, tuple) else o).detach().transpose(0,1).cpu().contiguous()})))

    # Targets for loss
    tgt_np = np.stack([arr[s*8192+1:(s+1)*8192+1].astype(np.int64) for s in SAMPLE_IDS])
    tgt = torch.from_numpy(tgt_np).to(device)

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        tok_emb = model.transformer.wte(idx)  # [B=4, T=8192, H=512]
        captured['embed'] = tok_emb.detach().transpose(0, 1).cpu().contiguous()
        # full forward
        _, loss = model(idx, targets=tgt)

    for h in handles:
        h.remove()

    print(f'nano loss on 4-sample batch: {loss.item():.6f}')

    # Load ref dumps (mbs0) and diff
    def load_ref(name):
        t = torch.load(f'{DUMP_DIR}/{name}-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt',
                       weights_only=False, map_location='cpu')
        if isinstance(t, tuple):
            t = t[0]
        return t  # [T, B, H] or [T, B, V] for lm_head

    print('\n=== Per-sublayer diff ===')
    print('nano captures in [T, B, H] to match ref')

    tensor_diff(captured['embed'], load_ref('embedding'), 'embedding')
    for i in range(9):
        tensor_diff(captured['blocks'][i], load_ref(f'decoder.layers.{i}'), f'block[{i}]')
    tensor_diff(captured['ln_f'], load_ref('decoder.final_layernorm'), 'ln_f')

    # Per-sublayer stats for block 0 (internal, no ref comparison)
    print('\n=== Nano block 0 sub-layer stats (for reference) ===')
    for k in ['L0_ln1_out', 'L0_attn_out', 'L0_ln2_out', 'L0_mlp_out']:
        v = captured[k]
        print(f'  {k}: shape={list(v.shape)} std={v.float().std():.4f} max={v.float().abs().max():.4f}')


if __name__ == '__main__':
    main()
