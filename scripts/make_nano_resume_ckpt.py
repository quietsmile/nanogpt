"""Wrap ref iter_0 weights + config into nano-format ckpt.pt for init_from='resume'."""
import sys, os, torch
sys.path.insert(0, '/home/claudeuser/nanogpt')
from model import GPTConfig, GPT

SRC = '/home/claudeuser/nanogpt/reports/megatron_iter0_ckpt.pt'
DST = '/home/claudeuser/nanogpt/reports/nano_ckpt_iter0.pt'

# Nano config matching 00196
cfg = GPTConfig(
    block_size=8192, vocab_size=152064, n_layer=9, n_head=4, n_embd=512, n_kv_head=2, kv_channels=64,
    dropout=0.0, bias=False, init_std=0.006,
    use_rope=True, rotary_base=50000, use_rmsnorm=True, norm_eps=1e-5, use_swiglu=True, ffn_hidden_size=1536,
    qk_layernorm=True, tie_embeddings=False, disable_scaled_init_method=True,
    use_moe=True, moe_layer_freq=[0]+[1]*8, num_experts=144, moe_ffn_hidden_size=160,
    moe_router_topk=8, moe_n_group=8, moe_topk_group=1, moe_norm_topk_prob=True,
    moe_router_score_correction_coeff=0.001, moe_shared_expert_hidden_size=160,
    moe_routing_type='greedy',
)
# Build model on meta to harvest state_dict shape, then load real weights
with torch.device('meta'):
    m = GPT(cfg)
nano_keys = set(m.state_dict().keys())
meg_sd = torch.load(SRC, map_location='cpu', weights_only=False)
print(f'nano expects {len(nano_keys)} keys; converted has {len(meg_sd)}')

# model_args dict mirroring train.py's expected fields
model_args = {
    'n_layer': cfg.n_layer, 'n_head': cfg.n_head, 'n_embd': cfg.n_embd, 'block_size': cfg.block_size,
    'bias': cfg.bias, 'vocab_size': cfg.vocab_size, 'dropout': cfg.dropout,
    'n_kv_head': cfg.n_kv_head, 'kv_channels': cfg.kv_channels,
    'use_rope': cfg.use_rope, 'rotary_base': cfg.rotary_base,
    'use_rmsnorm': cfg.use_rmsnorm, 'norm_eps': cfg.norm_eps,
    'use_swiglu': cfg.use_swiglu, 'ffn_hidden_size': cfg.ffn_hidden_size,
    'tie_embeddings': cfg.tie_embeddings, 'init_std': cfg.init_std,
    'qk_layernorm': cfg.qk_layernorm, 'disable_scaled_init_method': cfg.disable_scaled_init_method,
    'use_moe': cfg.use_moe, 'moe_layer_freq': cfg.moe_layer_freq,
    'num_experts': cfg.num_experts, 'moe_ffn_hidden_size': cfg.moe_ffn_hidden_size,
    'moe_router_topk': cfg.moe_router_topk, 'moe_n_group': cfg.moe_n_group,
    'moe_topk_group': cfg.moe_topk_group, 'moe_norm_topk_prob': cfg.moe_norm_topk_prob,
    'moe_router_score_correction_coeff': cfg.moe_router_score_correction_coeff,
    'moe_shared_expert_hidden_size': cfg.moe_shared_expert_hidden_size,
}

ckpt = {
    'model': meg_sd,
    'model_args': model_args,
    'optimizer': {},   # train.py's resume path tolerates missing optim keys
    'iter_num': 0,
    'best_val_loss': 1e9,
    'config': {},
}
torch.save(ckpt, DST)
sz = os.path.getsize(DST) / 1e9
print(f'wrote {DST} ({sz:.2f} GB)')
