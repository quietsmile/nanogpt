"""Per-layer isolated diff on TE-replaced model, compared to ref dumps."""
import os, sys
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPT_DIR)

from run_te_test_a import build_nano, build_te_model


def load_ref(dump_dir, name, mbs=0):
    p = f'{dump_dir}/{name}-iter5988-mbs{mbs}-forward-output-tp0.1-pp0.1-ep3.4.pt'
    t = torch.load(p, weights_only=False, map_location='cpu')
    if isinstance(t, tuple):
        t = t[0]
    return t


def main():
    DUMP = '/newcpfs/user/yuchen/karpathy/cybertron_dump/dumps'
    CKPT = '/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/iter_0005988'
    device = torch.device('cuda:0')

    from megatron_to_nano import load_all_megatron_shards
    meg = load_all_megatron_shards(CKPT)
    _, cfg = build_nano()
    model = build_te_model(cfg, meg, device)
    model.eval()

    # Load ref per-block outputs
    ref = {}
    ref['embed'] = load_ref(DUMP, 'embedding')
    for i in range(9):
        ref[f'b{i}'] = load_ref(DUMP, f'decoder.layers.{i}')

    print(f'\n=== TE-replaced model: ISOLATED PER-LAYER DIFF ===')
    print(f'{"layer":<8}{"L∞":>12}{"L1":>12}{"rel_mean":>12}')
    for i in range(9):
        inp_ref = ref['embed'] if i == 0 else ref[f'b{i-1}']
        inp_bth = inp_ref.transpose(0, 1).contiguous().to(device)  # [B, T, H] bf16
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            block = model.transformer.h[i]
            out, _ = block(inp_bth)
        out_tb = out.transpose(0, 1).contiguous().cpu()
        a, b = out_tb.float(), ref[f'b{i}'].float()
        d = (a - b).abs()
        rel = d.mean() / b.abs().mean().clamp_min(1e-8)
        print(f'block{i:<4}{d.max().item():>12.4e}{d.mean().item():>12.4e}{rel.item():>12.4e}')


if __name__ == '__main__':
    main()
