"""Compare nano's backward grad at module OUTPUTS (dL/dy) to ref's dumps.
Going backwards from loss: lm_head output → block 1 MoE output → block 1 attn output.

If ratio is uniform (2x everywhere), it's a scaling bug.
If ratio varies, more complex."""
import torch, torch.nn.functional as F
import sys, numpy as np
sys.path.insert(0, "/root/nanogpt"); sys.path.insert(0, "/root/nanogpt/scripts")
from megatron_to_nano import load_all_megatron_shards, convert
from model import GPTConfig, GPT

DUMP = "/newcpfs/user/yuchen/karpathy/cybertron_dump/dumps_allranks"
def load_out(p):
    t = torch.load(p, weights_only=False, map_location="cuda")
    if isinstance(t, tuple): return t[0] if t else None
    return t

meg = load_all_megatron_shards("/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/iter_0005988")
sd = convert(meg)
cfg = GPTConfig(block_size=8192, vocab_size=152064, n_layer=9, n_head=4, n_embd=512,
    n_kv_head=2, kv_channels=64, dropout=0.0, bias=False, init_std=0.006, use_rope=True,
    rotary_base=50000, use_rmsnorm=True, norm_eps=1e-5, use_swiglu=True, ffn_hidden_size=1536,
    qk_layernorm=True, tie_embeddings=False, disable_scaled_init_method=True, use_moe=True,
    moe_layer_freq=[0]+[1]*8, num_experts=144, moe_ffn_hidden_size=160, moe_router_topk=8,
    moe_n_group=8, moe_topk_group=1, moe_norm_topk_prob=True,
    moe_router_score_correction_coeff=0.001, moe_shared_expert_hidden_size=160,
    moe_routing_type="greedy", eod_token_id=151643, mask_loss_id=160000,
    seq_aux_balance_alpha=0.0001, use_eod_attn_mask=False)
model = GPT(cfg).cuda()
model.load_state_dict(sd, strict=False)
model.train()

# Run forward+backward on dp=3 mbs=0 (4 samples: 383244..383247)
sample_indices = [383244, 383245, 383246, 383247]
data = np.memmap('/root/nanogpt/data/cybertron_baseline/train.bin', dtype=np.int32, mode='r')
block = 8192

# Register hooks to capture dL/dy at key modules
captured = {}
hook_handles = []
def make_hook(name):
    def h(module, grad_input, grad_output):
        go = grad_output[0] if isinstance(grad_output, tuple) else grad_output
        if go is None: return
        captured.setdefault(name, []).append(go.detach().clone())
    return h
# Register on LM head, block 1 mlp, block 1 self_attn
# Note block 1 corresponds to model.transformer.h[1]
hook_handles.append(model.transformer.h[1].mlp.register_full_backward_hook(make_hook('block1.mlp')))
hook_handles.append(model.transformer.h[1].attn.register_full_backward_hook(make_hook('block1.attn')))
# lm_head uses F.linear directly (not nn.Module call) so use .weight.register_hook instead
# for comparing lm_head OUTPUT grad, we can capture the final_hidden's grad via model.transformer.ln_f
hook_handles.append(model.transformer.ln_f.register_full_backward_hook(make_hook('final_ln')))

EOD = 151643
# Count total unmasked tokens in 4 samples (match ref's mb semantic)
n_tok_total = sum((torch.from_numpy(data[s*block : s*block + block].astype(np.int64)) != EOD).sum().item()
                  for s in sample_indices)
print(f"n_tok_total (4 samples, unmasked): {n_tok_total}")

# Process each sample, contribute sum_ce/n_tok_total to mb_loss (not /num_mb, matching ref WITHOUT /num_mb)
for s_idx in sample_indices:
    X = torch.from_numpy(data[s_idx*block : s_idx*block + block].astype(np.int64)).unsqueeze(0).cuda()
    Y = torch.from_numpy(data[s_idx*block + 1 : s_idx*block + 1 + block].astype(np.int64)).unsqueeze(0).cuda()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(X, targets=Y)
        ce = F.cross_entropy(logits.float().view(-1, logits.size(-1)), Y.view(-1),
                              ignore_index=-1, reduction='none')
        mask = (X.view(-1) != EOD).float()
        sample_contrib = (ce * mask).sum() / n_tok_total
        sample_contrib.backward()
    del X, Y, logits
    torch.cuda.empty_cache()
for h in hook_handles: h.remove()

# Ref dL/dy dumps
def load_ref(name, mbs=0, dp=3):
    p = f"{DUMP}/{name}-iter5988-mbs{mbs}-backward-output-tp0.1-pp0.1-ep3.4-dp{dp}.8.pt"
    return load_out(p)

ref_lm = load_ref('output_layer')  # [T, B, V]
ref_mlp = load_ref('decoder.layers.1.mlp')  # [T, B, C]
ref_attn = load_ref('decoder.layers.1.self_attention')  # [T, B, C]

print(f"\n=== dL/dy norms ===")
def compare(name, nano_g, ref_g):
    nn = nano_g.float().norm().item()
    rn = ref_g.float().norm().item()
    cos = torch.nn.functional.cosine_similarity(
        nano_g.float().flatten().unsqueeze(0),
        ref_g.float().flatten().unsqueeze(0)).item()
    print(f"  {name:25s}: nano={nn:.4e}  ref={rn:.4e}  ratio={nn/rn:.4f}  cos={cos:.6f}")

for k in captured:
    print(f"  captured: {k} has {len(captured[k])} tensors, first shape={captured[k][0].shape}")
def combine(tensors):
    # Each shape [1, T, C]; concat on batch dim → [4, T, C]; transpose → [T, 4, C] to match ref
    out = torch.cat(tensors, dim=0)  # [4, T, C]
    return out.transpose(0, 1).contiguous()  # [T, 4, C]

if 'block1.mlp' in captured:
    compare("block1.mlp dL/dy", combine(captured['block1.mlp']), ref_mlp)
if 'block1.attn' in captured:
    compare("block1.attn dL/dy", combine(captured['block1.attn']), ref_attn)
# final_ln dL/dy is at ln_f output; compare to output_layer forward-input grad?
# Actually ref_lm is dL/dy at output_layer OUTPUT = dL/dlogits, not same module. Skip lm_head for now.
print("(lm_head separately shown to be 2.0× in diag_lm_head_grad.py)")
