"""Load Megatron iter_0001497 weights into nano, forward on samples at step 1498, measure loss.

If loss ≈ ref step 1498 loss (~3.44), forward + loss computation is correct.
If loss deviates significantly, a forward/loss bug accumulates across iters.

Does 64 samples (full global batch at that step) in chunks; returns mean.
"""
import os, sys, json
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from model import GPTConfig, GPT

CKPT = '/root/nanogpt/reports/megatron_to_nano_ckpt.pt'
DATA = '/root/nanogpt/data/cybertron_baseline/train.bin'
BLOCK = 8192
GLOBAL_BS = 64
ITER = 1498  # step at which weights we loaded were checkpointed at 1497 → next step is 1498


def main():
    cfg = GPTConfig(
        block_size=BLOCK, vocab_size=152064,
        n_layer=9, n_head=4, n_embd=512, n_kv_head=2, kv_channels=64,
        use_rope=True, rotary_base=50000, use_rmsnorm=True, norm_eps=1e-5,
        use_swiglu=True, ffn_hidden_size=1536, qk_layernorm=True,
        tie_embeddings=False, init_std=0.006, disable_scaled_init_method=True,
        use_moe=True, moe_layer_freq=[0]+[1]*8, num_experts=144,
        moe_ffn_hidden_size=160, moe_router_topk=8, moe_n_group=8, moe_topk_group=1,
        moe_norm_topk_prob=True, moe_router_score_correction_coeff=0.001,
        moe_shared_expert_hidden_size=160,
        # Match ref: eod_mask_loss=True + mask_loss_id=160000 (no seq_aux needed for forward-only loss)
        eod_token_id=151643, mask_loss_id=160000, seq_aux_balance_alpha=0.0, use_eod_attn_mask=True,
    )
    m = GPT(cfg).cuda().bfloat16()
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)
    m.load_state_dict(sd, strict=False)
    m.eval()
    print(f'loaded Megatron weights into nano ({sum(p.numel() for p in m.parameters())/1e6:.2f}M)', flush=True)

    # Samples at step 1498 (iter_num 1497 is the ckpt, so next step reads _seq_data_pos = 1497*8 = 11976 on rank 0 with DP=8)
    # Global batch 64 covers samples [start, start+64) where start = (ITER-1) * GLOBAL_BS in continuous consumption
    arr = np.memmap(DATA, dtype=np.uint16, mode='r')
    n_samples = (len(arr) - 1) // BLOCK
    start_sample = (ITER - 1) * GLOBAL_BS  # 1497 * 64 = 95808
    print(f'global batch at iter {ITER}: samples [{start_sample}, {start_sample+GLOBAL_BS})', flush=True)

    losses = []
    chunk = 8
    for c0 in range(0, GLOBAL_BS, chunk):
        idx_list, tgt_list = [], []
        for s in range(chunk):
            sid = (start_sample + c0 + s) % n_samples
            off = sid * BLOCK
            tok = np.array(arr[off:off + BLOCK + 1].astype(np.int64))
            idx_list.append(torch.from_numpy(tok[:BLOCK]))
            tgt_list.append(torch.from_numpy(tok[1:BLOCK+1]))
        idx = torch.stack(idx_list, dim=0).cuda()
        tgt = torch.stack(tgt_list, dim=0).cuda()
        with torch.no_grad():
            _, loss = m(idx, targets=tgt)
        losses.append(loss.item())
        print(f'  samples [{c0}:{c0+chunk}]: loss={loss.item():.4f}', flush=True)

    mean = sum(losses)/len(losses)
    print(f'\nmean loss over 64 samples: {mean:.4f}')
    print(f'ref lm loss at iter 1497: 3.4435 (step 1498 interpolated ~3.35)')
    print(f'|diff|: {abs(mean-3.35):.4f}')
    out = {'nano_mean_loss_iter_1498': mean, 'per_chunk': losses,
           'ref_iter_1497_loss': 3.4435, 'expected_iter_1498': 3.35}
    json.dump(out, open('/root/nanogpt/reports/loss_at_resume.json','w'), indent=2)


if __name__ == '__main__':
    main()
