"""Compare nano's MoERouter output against Megatron-core's topk_routing_with_score_function
given the same logits + expert_bias. If Megatron's routing decisions differ from nano's on
the SAME inputs, we've found the source of the +0.014 nat residual.
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from megatron_to_nano import load_all_megatron_shards, convert
from model import GPTConfig, GPT

from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function


def build_nano():
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


def main():
    device = torch.device('cuda:0')
    model, cfg = build_nano()
    model = model.to(device)
    meg = load_all_megatron_shards('/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/iter_0005988')
    sd = convert(meg)
    model.load_state_dict(sd, strict=False)
    model.eval()

    arr = np.memmap('/root/nanogpt/data/cybertron_baseline/train.bin', dtype=np.int32, mode='r')
    sid = 5988 * 64
    idx = torch.from_numpy(np.array(arr[sid*8192:(sid+1)*8192].astype(np.int64))).unsqueeze(0).to(device)

    # Forward to layer 8's router input
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        x = model.transformer.wte(idx)
        for bi in range(9):
            block = model.transformer.h[bi]
            x = x + block.attn(block.ln_1(x))
            if bi == 8:
                mlp_in = block.ln_2(x)
                break
            mlp_in_b = block.ln_2(x)
            if block.is_moe:
                mo, _ = block.mlp(mlp_in_b)
            else:
                mo = block.mlp(mlp_in_b)
            x = x + mo

        router = model.transformer.h[8].mlp.router
        x_flat = mlp_in.view(-1, 512)

        # Call the REAL model router (which now has the autocast-disabled fix)
        import torch.nn.functional as F
        topk_idx_nano, final_weights_nano, raw_scores_nano = router(x_flat)
        # Also compute logits/scores for debugging
        with torch.amp.autocast('cuda', enabled=False):
            logits = F.linear(x_flat.float(), router.linear.weight.float())
            scores_nano = torch.sigmoid(logits)
            scores_biased_nano = scores_nano + router.e_score_correction_bias.float()

        # Megatron-core path (exact reference). Disable autocast so dtypes stay as specified.
        with torch.amp.autocast('cuda', enabled=False):
            routing_probs_meg, routing_map_meg = topk_routing_with_score_function(
                logits=logits.float(),
                topk=8,
                use_pre_softmax=False,
                num_groups=None, group_topk=None,
                scaling_factor=None,
                score_function='sigmoid',
                expert_bias=router.e_score_correction_bias.float(),
                fused=False,
            )

        # Derive top_indices (sorted) from Megatron routing_map (bool [S, E])
        # Then compare to nano's topk_idx (also sort for fair compare)
        idx_nano_sorted, _ = topk_idx_nano.sort(dim=-1)
        # Get sorted indices of True positions from routing_map
        s_idx, _ = torch.nonzero(routing_map_meg, as_tuple=True)  # flat indices
        # This is complex; just check the number of experts selected per row and whether sets match
        n_selected_meg = routing_map_meg.sum(dim=-1)
        assert (n_selected_meg == 8).all(), f'Meg selected != 8 on some rows: {n_selected_meg.unique().tolist()}'
        # Convert nano topk to bool [S, E] form
        mask_nano = torch.zeros_like(routing_map_meg, dtype=torch.bool)
        mask_nano.scatter_(1, topk_idx_nano, True)
        # Exact set match per row
        exact_match = (mask_nano == routing_map_meg).all(dim=-1)
        diffs_per_token = (mask_nano != routing_map_meg).sum(dim=-1).float() / 2  # each differing expert contributes 1 each side
        print(f'Layer 8 / 8192 tokens:')
        print(f'  exact topk-set match: {exact_match.sum().item()}/{x_flat.size(0)} ({100*exact_match.float().mean().item():.2f}%)')
        print(f'  mean differing experts per token: {diffs_per_token.mean().item():.3f}')

        matches = exact_match
        if matches.any():
            w_meg_at_nano_idx = routing_probs_meg.gather(1, topk_idx_nano)
            wd = (final_weights_nano[matches] - w_meg_at_nano_idx[matches]).abs().max().item()
            print(f'  for matching tokens ({matches.sum().item()}): max weight diff = {wd:.3e}')
        if (~matches).any():
            mism = (~matches).nonzero(as_tuple=False).squeeze(1)
            tok = mism[0].item()
            nano_set = set(topk_idx_nano[tok].tolist())
            meg_set = set(routing_map_meg[tok].nonzero(as_tuple=False).squeeze(1).tolist())
            e_nano = list(nano_set - meg_set)[0]
            e_meg = list(meg_set - nano_set)[0]
            print(f'\n  First mismatch at token {tok}: nano-only={nano_set - meg_set}, meg-only={meg_set - nano_set}')
            print(f'  logits[{tok}, {e_nano}] = {logits[tok, e_nano].item():.10f}')
            print(f'  logits[{tok}, {e_meg}]  = {logits[tok, e_meg].item():.10f}')
            print(f'  scores_biased_nano[{tok}, {e_nano}] = {scores_biased_nano[tok, e_nano].item():.10f}')
            print(f'  scores_biased_nano[{tok}, {e_meg}]  = {scores_biased_nano[tok, e_meg].item():.10f}')
            # Recompute with strict fp32
            with torch.amp.autocast('cuda', enabled=False):
                sig = torch.sigmoid(logits[tok].float())
                sb = sig + router.e_score_correction_bias.float()
            print(f'  strict-fp32 scores_biased[{tok}, {e_nano}] = {sb[e_nano].item():.10f}')
            print(f'  strict-fp32 scores_biased[{tok}, {e_meg}]  = {sb[e_meg].item():.10f}')
            # Check actual dtype of nano scores
            print(f'  nano scores_nano dtype: {scores_nano.dtype}')
            print(f'  nano scores_biased_nano dtype: {scores_biased_nano.dtype}')


if __name__ == '__main__':
    main()
