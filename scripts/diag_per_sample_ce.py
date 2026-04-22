"""Compare per-sample CE between nano and ref for all 16 dumped samples
(iter 5989's ep=3 ranks: dp3 + dp7 × mbs0 + mbs1)."""
import torch, torch.nn.functional as F
import sys, numpy as np
sys.path.insert(0, "/root/nanogpt"); sys.path.insert(0, "/root/nanogpt/scripts")

DUMP = "/newcpfs/user/yuchen/karpathy/cybertron_dump/dumps_allranks"

# Load all 16 ref logits (iter 5988 = iter 5989's forward in our run)
ref_logits_all = {}  # key = (dp, mbs, b), value = [T, V]
for dp in [3, 7]:
    for mbs in [0, 1]:
        path = f'{DUMP}/output_layer-iter5988-mbs{mbs}-forward-output-tp0.1-pp0.1-ep3.4-dp{dp}.8.pt'
        t = torch.load(path, weights_only=False, map_location='cuda')
        if isinstance(t, tuple): t = t[0]
        # [T, B, V]
        print(f'dp{dp} mbs{mbs} logits shape={t.shape}')
        for b in range(4):
            ref_logits_all[(dp, mbs, b)] = t[:, b, :]

# Also get embedding output to identify tokens via embedding row matching
# Actually simpler: match samples by loading nano data.bin and checking first 10 tokens
# from embedding output. But embedding output is bf16 hidden state, not token ids.
#
# We can match by rebuilding nano's data pool for iter 5989 and reverse-engineering
# which samples went to which (dp, mbs, b).
# iter 5989 consumes samples [5989*64, 5989*64+64) = [383296, 383360) in nano data.
# Each ep rank gets 16 samples. We need to match exactly which 16 are on ep=3.

# Actually the token IDs are embedded. Let me decode via embedding-weight inverse matching.
# We have ref ckpt - load embedding weight.
from megatron_to_nano import load_all_megatron_shards
meg = load_all_megatron_shards("/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/iter_0005988")
embed_w = meg['embedding.word_embeddings.weight'].cuda()  # [V, C]
V, C = embed_w.shape

# Build hash map
print(f'Building hash map for {V} embedding rows...')
row_to_id = {}
w_bytes = embed_w.contiguous().view(torch.uint8).view(V, C * 2)  # bf16 = 2 bytes
for v in range(V):
    key = bytes(w_bytes[v].tolist())
    row_to_id[key] = v

# For each dumped sample, recover token ids via embedding output
# Identify which nano sample (= sample_idx) each dumped b corresponds to
data = np.memmap('/root/nanogpt/data/cybertron_baseline/train.bin', dtype=np.int32, mode='r')
block = 8192

sample_map = {}  # (dp, mbs, b) → nano_sample_idx
for dp in [3, 7]:
    for mbs in [0, 1]:
        emb_path = f'{DUMP}/embedding-iter5988-mbs{mbs}-forward-output-tp0.1-pp0.1-ep3.4-dp{dp}.8.pt'
        emb = torch.load(emb_path, weights_only=False, map_location='cuda')  # [T, B, C] bf16
        ref_bytes = emb.contiguous().view(torch.uint8).view(emb.shape[0], emb.shape[1], C * 2)
        for b in range(4):
            # Use first 10 tokens to match in nano
            tok_ids = []
            for s in range(10):
                key = bytes(ref_bytes[s, b].tolist())
                tid = row_to_id.get(key, -1)
                tok_ids.append(tid)
            # Find nano sample whose first 10 tokens match.
            # Filename says iter5988 but dumped samples may be from EITHER iter 5988
            # or 5989 depending on sampler state on resume. Search a wider range.
            matched = -1
            for s_idx in list(range(383232, 383296)) + list(range(383296, 383360)):
                nano_first10 = data[s_idx*block : s_idx*block + 10].tolist()
                if nano_first10 == tok_ids:
                    matched = s_idx
                    break
            sample_map[(dp, mbs, b)] = matched
            print(f'  dp{dp} mbs{mbs} b{b} → nano sample idx {matched} (first10={tok_ids})')

# Now for each matched sample, compute ref CE and nano CE
from model import GPTConfig, GPT
cfg = GPTConfig(block_size=8192, vocab_size=152064, n_layer=9, n_head=4, n_embd=512,
    n_kv_head=2, kv_channels=64, dropout=0.0, bias=False, init_std=0.006, use_rope=True,
    rotary_base=50000, use_rmsnorm=True, norm_eps=1e-5, use_swiglu=True, ffn_hidden_size=1536,
    qk_layernorm=True, tie_embeddings=False, disable_scaled_init_method=True, use_moe=True,
    moe_layer_freq=[0]+[1]*8, num_experts=144, moe_ffn_hidden_size=160, moe_router_topk=8,
    moe_n_group=8, moe_topk_group=1, moe_norm_topk_prob=True,
    moe_router_score_correction_coeff=0.001, moe_shared_expert_hidden_size=160,
    moe_routing_type="greedy", eod_token_id=151643, mask_loss_id=160000,
    seq_aux_balance_alpha=0.0, use_eod_attn_mask=False)
from megatron_to_nano import convert
sd = convert(meg)
model = GPT(cfg).cuda()
model.load_state_dict(sd, strict=False)
model.eval()

EOD = 151643
MASK_ID = 160000
ref_total_sum, ref_total_n = 0.0, 0
nano_total_sum, nano_total_n = 0.0, 0

print(f'\n{"dp":>3s} {"mbs":>4s} {"b":>3s} {"sample_idx":>12s} {"ref_ce":>10s} {"nano_ce":>10s} {"Δ":>10s} {"n_unmasked":>12s}')
for key in sorted(sample_map.keys()):
    dp, mbs, b = key
    s_idx = sample_map[key]
    if s_idx < 0: continue
    X = torch.from_numpy(data[s_idx*block : s_idx*block + block].astype(np.int64)).cuda()
    Y = torch.from_numpy(data[s_idx*block + 1 : s_idx*block + 1 + block].astype(np.int64)).cuda()
    ref_logits = ref_logits_all[key]

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        X_b = X.unsqueeze(0)
        Y_b = Y.unsqueeze(0)
        nano_logits, _ = model(X_b, targets=Y_b)
    nano_logits = nano_logits[0]  # [T, V]

    ce_ref = F.cross_entropy(ref_logits.float(), Y, reduction='none')
    ce_nano = F.cross_entropy(nano_logits.float(), Y, reduction='none')
    mask = ((X != EOD) & (X != MASK_ID)).float()
    n = int(mask.sum().item())
    ref_mean = (ce_ref * mask).sum().item() / n
    nano_mean = (ce_nano * mask).sum().item() / n
    print(f'{dp:>3d} {mbs:>4d} {b:>3d} {s_idx:>12d} {ref_mean:>10.4f} {nano_mean:>10.4f} {nano_mean-ref_mean:>+10.4f} {n:>12d}')
    ref_total_sum += (ce_ref * mask).sum().item()
    ref_total_n += n
    nano_total_sum += (ce_nano * mask).sum().item()
    nano_total_n += n
    del X, Y, nano_logits
    torch.cuda.empty_cache()

print(f'\n=== 16-sample token-weighted avg ===')
print(f'  ref  = {ref_total_sum / ref_total_n:.6f}')
print(f'  nano = {nano_total_sum / nano_total_n:.6f}')
print(f'  Δ    = {(nano_total_sum - ref_total_sum) / nano_total_n:+.6f}')
print(f'ref logged iter 5989 lm_loss = 3.057512')
