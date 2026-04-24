"""Verify nano forward loss on specific sample ranges to pinpoint the 2-nat bug.

If loss depends wildly on which sample, it's likely a stateful bug (e.g., MoE router state,
KV cache, BatchNorm running stats). If loss is uniformly wrong, it's pure fwd math.
"""
import os, sys, json, numpy as np, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from model import GPTConfig, GPT

CKPT = '/root/nanogpt/reports/megatron_to_nano_ckpt.pt'
DATA = '/root/nanogpt/data/cybertron_baseline/train.bin'
BLOCK = 8192


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
        eod_token_id=None, mask_loss_id=None, seq_aux_balance_alpha=0.0, use_eod_attn_mask=False,
    )
    m = GPT(cfg).cuda().bfloat16()
    m.load_state_dict(torch.load(CKPT, map_location='cpu', weights_only=False), strict=False)
    m.eval()

    arr = np.memmap(DATA, dtype=np.uint16, mode='r')

    def loss_for(start, n=1):
        losses = []
        for s in range(n):
            off = (start + s) * (BLOCK + 1)
            tok = np.array(arr[off:off + BLOCK + 1].astype(np.int64))
            idx = torch.from_numpy(tok[:BLOCK]).unsqueeze(0).cuda()
            tgt = torch.from_numpy(tok[1:BLOCK+1]).unsqueeze(0).cuda()
            with torch.no_grad():
                _, loss = m(idx, targets=tgt)
            losses.append(loss.item())
        return losses

    # Replicate earlier: sample 0
    l0 = loss_for(0)[0]
    print(f'sample 0: loss={l0:.4f}  (previous run: 2.80)', flush=True)

    # Sample 0 via batched [8,8192] (how loss_at_resume_test did it)
    idx_list, tgt_list = [], []
    for s in range(8):
        off = s * (BLOCK + 1)
        tok = np.array(arr[off:off + BLOCK + 1].astype(np.int64))
        idx_list.append(torch.from_numpy(tok[:BLOCK]))
        tgt_list.append(torch.from_numpy(tok[1:BLOCK+1]))
    idx = torch.stack(idx_list, dim=0).cuda()
    tgt = torch.stack(tgt_list, dim=0).cuda()
    with torch.no_grad():
        _, loss_batched8 = m(idx, targets=tgt)
    print(f'samples [0:8] batched: loss={loss_batched8.item():.4f}', flush=True)

    # Per-sample loss for each of 0..7
    print('per-sample (batch=1 forward):', flush=True)
    for s in range(8):
        ls = loss_for(s)[0]
        print(f'  sample {s}: {ls:.4f}', flush=True)

    # Peek logits on sample 0 at a specific position to see top-k
    off = 0
    tok = np.array(arr[off:off + BLOCK + 1].astype(np.int64))
    idx = torch.from_numpy(tok[:BLOCK]).unsqueeze(0).cuda()
    tgt = torch.from_numpy(tok[1:BLOCK+1]).unsqueeze(0).cuda()
    with torch.no_grad():
        logits, _ = m(idx, targets=tgt)
    print(f'\nlogits shape: {tuple(logits.shape)}', flush=True)
    # Check position 100: show predicted top-5 and target
    for pos in [10, 100, 500, 1000, 5000, 8000]:
        top5 = logits[0, pos].topk(5)
        tgt_id = tgt[0, pos].item()
        tgt_logit = logits[0, pos, tgt_id].item()
        print(f'pos {pos}: target={tgt_id} tgt_logit={tgt_logit:.3f} top5_ids={top5.indices.tolist()} top5_vals={[f"{v:.2f}" for v in top5.values.tolist()]}', flush=True)


if __name__ == '__main__':
    main()
