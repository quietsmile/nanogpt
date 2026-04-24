"""Compare nano's per-layer activations against ref's bitwise_dump output.

Ref dumps (from cybertron via hook_fwd_bwd_to_module) are at paths like:
  {dump_dir}/{module_name}-iter{N}-mbs{M}-forward-{input|output}-tp0.1-pp0.1-ep0.1.pt

We load these tensors and compare against nano's forward on the same sample.
For each hook point, report L∞/L1/L2 diff and first diverging position.
"""
import argparse
import glob
import os
import sys

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


def load_ref_dump(dump_dir, module_name, iter_num=5989, mbs=0):
    """Find and load ref's dump for a given module/iter/mbs."""
    patt = f'{module_name}-iter{iter_num}-mbs{mbs}-forward-output-*.pt'
    files = sorted(glob.glob(os.path.join(dump_dir, patt)))
    if not files:
        return None
    # Load the last-rank file (that's what bitwise_dump saves)
    return torch.load(files[0], weights_only=False, map_location='cpu')


def tensor_stats(a, b, name=''):
    if isinstance(a, (tuple, list)):
        for i, (ai, bi) in enumerate(zip(a, b)):
            tensor_stats(ai, bi, f'{name}[{i}]')
        return
    if not (torch.is_tensor(a) and torch.is_tensor(b)):
        print(f'  {name}: skip (non-tensor: {type(a).__name__}, {type(b).__name__})')
        return
    if a.shape != b.shape:
        print(f'  {name}: SHAPE MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}')
        return
    af, bf = a.float(), b.float()
    diff = (af - bf).abs()
    rel = diff / bf.abs().clamp_min(1e-6)
    print(f'  {name}: shape={tuple(a.shape)} L∞={diff.max().item():.4e} L1={diff.mean().item():.4e} rel_max={rel.max().item():.4e}')


def nano_forward_hooks(model, idx, tgt, hook_modules):
    """Forward through nano, capturing outputs of the specified submodules."""
    captured = {}
    handles = []
    for name, mod in model.named_modules():
        if name in hook_modules:
            def make_hook(n):
                def fn(module, inputs, output):
                    captured[n] = output.detach().cpu() if torch.is_tensor(output) else output
                return fn
            handles.append(mod.register_forward_hook(make_hook(name)))
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits, loss = model(idx, targets=tgt)
    captured['__loss__'] = loss.item()
    for h in handles:
        h.remove()
    return captured


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump-dir', required=True, help='Path to ref dumps dir')
    ap.add_argument('--iter', type=int, default=5989)
    ap.add_argument('--ckpt-dir', default='/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/iter_0005988')
    args = ap.parse_args()

    device = torch.device('cuda:0')
    from megatron_to_nano import load_all_megatron_shards, convert
    meg = load_all_megatron_shards(args.ckpt_dir)
    sd = convert(meg)
    model, _ = build_nano()
    model = model.to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    import numpy as np
    arr = np.memmap(os.path.join(ROOT, 'data/cybertron_baseline/train.bin'), dtype=np.int32, mode='r')
    # Ref iter 5989 mbs 0 = first sample in global batch = sample 5988*64 + 0
    sid = (args.iter - 1) * 64
    idx = torch.from_numpy(np.array(arr[sid*8192:(sid+1)*8192].astype(np.int64))).unsqueeze(0).to(device)
    tgt = torch.from_numpy(np.array(arr[sid*8192+1:(sid+1)*8192+1].astype(np.int64))).unsqueeze(0).to(device)

    # List available dumps
    files = sorted(glob.glob(os.path.join(args.dump_dir, f'*-iter{args.iter}-mbs0-forward-output-*.pt')))
    print(f'Found {len(files)} ref dump files at iter {args.iter}, mbs 0')
    for f in files[:20]:
        print(f'  {os.path.basename(f)}')
    print()

    # For each ref dump, attempt mapping + comparison
    # Megatron module name → nano module name mapping
    meg_to_nano = {
        'embedding': 'transformer.wte',
        'decoder.final_layernorm': 'transformer.ln_f',
        'output_layer': 'lm_head',
    }
    for i in range(9):
        meg_to_nano[f'decoder.layers.{i}'] = f'transformer.h.{i}'

    nano_hooks = set(meg_to_nano.values())
    captured = nano_forward_hooks(model, idx, tgt, nano_hooks)
    print(f'nano loss: {captured["__loss__"]:.6f}')

    print('\n=== Per-layer comparison ===')
    for meg_name, nano_name in meg_to_nano.items():
        ref_tensor = load_ref_dump(args.dump_dir, meg_name, iter_num=args.iter)
        if ref_tensor is None:
            print(f'{meg_name}: NO DUMP')
            continue
        nano_out = captured.get(nano_name)
        if nano_out is None:
            print(f'{meg_name}: nano hook missing')
            continue
        print(f'{meg_name} ↔ {nano_name}:')
        tensor_stats(nano_out, ref_tensor, f'  out')


if __name__ == '__main__':
    main()
