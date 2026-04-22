"""Load the converted Megatron→nano ckpt and run a forward pass.

Part A (default): tiny forward on CPU with seq_len=256, checks load correctness + finite logits.
Part B (--full): full seq_len=8192 forward on GPU, reads real sample from train.bin.
                 Emits per-layer activation statistics to compare against ref TB values.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from model import GPTConfig, GPT  # noqa: E402

CKPT_PATH = '/home/claudeuser/nanogpt/reports/megatron_to_nano_ckpt.pt'


def build_model(use_eod_attn_mask: bool = False):
    cfg = GPTConfig(
        block_size=8192, vocab_size=152064,
        n_layer=9, n_head=4, n_embd=512, n_kv_head=2, kv_channels=64,
        use_rope=True, rotary_base=50000, use_rmsnorm=True, norm_eps=1e-5,
        use_swiglu=True, ffn_hidden_size=1536, qk_layernorm=True,
        tie_embeddings=False, init_std=0.006, disable_scaled_init_method=True,
        use_moe=True, moe_layer_freq=[0] + [1] * 8, num_experts=144,
        moe_ffn_hidden_size=160, moe_router_topk=8, moe_n_group=8, moe_topk_group=1,
        moe_norm_topk_prob=True, moe_router_score_correction_coeff=0.001,
        moe_shared_expert_hidden_size=160,
        eod_token_id=151643, mask_loss_id=160000,
        seq_aux_balance_alpha=0.0, use_eod_attn_mask=use_eod_attn_mask,
    )
    m = GPT(cfg)
    return m, cfg


def load_and_report(model):
    sd = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    model_sd = model.state_dict()
    # Missing vs extra
    nano_keys = set(model_sd.keys())
    ck_keys = set(sd.keys())
    missing = nano_keys - ck_keys
    extra = ck_keys - nano_keys
    print(f"state_dict: nano={len(nano_keys)} ckpt={len(ck_keys)}")
    print(f"  missing in ckpt (nano expects but not in converted): {len(missing)}")
    for k in sorted(missing):
        # Skip known nano-only buffers (RoPE cached tables, etc.)
        if any(s in k for s in ('rotary_emb.', 'cos_cached', 'sin_cached', 'inv_freq', '.bias')):
            continue
        print(f"    {k}  (shape in nano: {tuple(model_sd[k].shape)})")
    print(f"  extra in ckpt (ckpt has but nano doesn't): {len(extra)}")
    for k in sorted(extra):
        print(f"    {k}")
    # Shape mismatches
    mismatches = []
    for k in nano_keys & ck_keys:
        if tuple(model_sd[k].shape) != tuple(sd[k].shape):
            mismatches.append((k, tuple(model_sd[k].shape), tuple(sd[k].shape)))
    print(f"  shape mismatches: {len(mismatches)}")
    for k, s_nano, s_ck in mismatches[:10]:
        print(f"    {k}: nano={s_nano}  ckpt={s_ck}")
    # Actually load (strict=False to tolerate RoPE buffers + e_score_correction_bias)
    result = model.load_state_dict(sd, strict=False)
    print(f"  load_state_dict missing: {len(result.missing_keys)}")
    for k in result.missing_keys[:8]:
        print(f"    {k}")
    print(f"  load_state_dict unexpected: {len(result.unexpected_keys)}")
    for k in result.unexpected_keys[:8]:
        print(f"    {k}")
    return result


def tiny_forward(model, seq_len=256, batch=1):
    model.eval()
    torch.manual_seed(42)
    idx = torch.randint(0, 152064, (batch, seq_len))
    with torch.no_grad():
        logits, _ = model(idx)
    print(f"tiny forward seq_len={seq_len}:")
    print(f"  logits shape: {tuple(logits.shape)}")
    print(f"  logits[0, -1, :5]: {logits[0, -1, :5].tolist()}")
    print(f"  logits mean/std/max/min: {logits.mean():.4f} / {logits.std():.4f} / "
          f"{logits.max():.4f} / {logits.min():.4f}")
    print(f"  logits finite: {torch.isfinite(logits).all().item()}")
    # Hash
    import hashlib
    h = hashlib.sha256(logits.numpy().tobytes()).hexdigest()[:16]
    print(f"  logits sha256 (first 16): {h}")
    return logits


def full_forward(model, device='cuda', dtype=torch.bfloat16):
    """Forward one full 8192-token sample from train.bin, collect per-layer activation stats."""
    model = model.to(device).to(dtype)
    model.eval()

    data_bin = os.path.join(ROOT, 'data', 'cybertron_baseline', 'train.bin')
    if not os.path.exists(data_bin):
        print(f"(train.bin not available at {data_bin}; skipping real-sample forward)")
        return
    arr = np.memmap(data_bin, dtype=np.uint16, mode='r')
    # Take first sample (block_size+1 = 8193 tokens)
    seq = np.array(arr[:8193].astype(np.int64))
    idx = torch.from_numpy(seq[:8192]).unsqueeze(0).to(device)
    tgt = torch.from_numpy(seq[1:8193]).unsqueeze(0).to(device)

    # Hook registration for per-layer stats
    stats = {}
    def hook(name):
        def fn(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            stats[name] = {
                'mean': o.float().mean().item(),
                'std': o.float().std().item(),
                'max': o.float().max().item(),
                'min': o.float().min().item(),
                'abs_max': o.float().abs().max().item(),
            }
        return fn
    handles = []
    for L, block in enumerate(model.transformer.h):
        handles.append(block.attn.register_forward_hook(hook(f'attn_output/l{L}')))
        handles.append(block.mlp.register_forward_hook(hook(f'mlp_output/l{L}')))

    with torch.no_grad():
        logits, loss = model(idx, targets=tgt)
    for h in handles: h.remove()

    print(f"full forward seq=8192:")
    print(f"  loss: {loss.item():.4f}")
    print(f"  logits[0, -1, :5]: {logits[0, -1, :5].tolist()}")
    print(f"  per-layer activation stats:")
    for k in sorted(stats.keys()):
        s = stats[k]
        print(f"    {k}: mean={s['mean']:+.4f} std={s['std']:.4f} abs_max={s['abs_max']:.3f}")

    out = {'loss': loss.item(), 'logits_first10': logits[0, -1, :10].float().tolist(),
           'stats': stats}
    out_path = os.path.join(ROOT, 'reports', 'megatron_weight_forward.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true', help='run full 8192 forward on GPU (needs CUDA)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    model, cfg = build_model()
    print(f"Model built: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    load_and_report(model)

    if args.full and args.device == 'cuda':
        full_forward(model, device='cuda')
    else:
        tiny_forward(model)


if __name__ == '__main__':
    main()
