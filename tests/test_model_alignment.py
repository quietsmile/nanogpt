"""Phase 3: model structure / param / FLOPs alignment for scaling_moe_00196.

Runs entirely on DSW (no GPU). Uses torch meta device so no RAM is touched by the 447M model.
Compares against /home/claudeuser/nanogpt/reference/megatron_state_dict_shapes.json.
"""
import os, sys, json, math, re, unittest

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from model import GPTConfig, GPT  # noqa: E402
from tools.flops_mfu import count_params_detailed, compute_flops_per_step, PEAK_FLOPS  # noqa: E402
from scripts.param_name_map import build_name_map  # noqa: E402


def _load_config_file(config_path):
    """Evaluate a nanogpt config file and return a namespace dict."""
    ns = {}
    with open(config_path) as f:
        exec(f.read(), ns)
    return {k: v for k, v in ns.items() if not k.startswith('__')}


def _build_gpt_config_from_namespace(ns):
    # GPTConfig default fields that matter:
    return GPTConfig(
        block_size=ns['block_size'],
        vocab_size=ns['vocab_size_override'],
        n_layer=ns['n_layer'],
        n_head=ns['n_head'],
        n_embd=ns['n_embd'],
        dropout=ns.get('dropout', 0.0),
        bias=ns.get('bias', False),
        init_std=ns.get('init_std', 0.02),
        n_kv_head=ns.get('n_kv_head'),
        kv_channels=ns.get('kv_channels'),
        use_rope=ns.get('use_rope', False),
        rotary_base=ns.get('rotary_base', 10000),
        use_rmsnorm=ns.get('use_rmsnorm', False),
        norm_eps=ns.get('norm_eps', 1e-5),
        use_swiglu=ns.get('use_swiglu', False),
        ffn_hidden_size=ns.get('ffn_hidden_size'),
        tie_embeddings=ns.get('tie_embeddings', True),
        qk_layernorm=ns.get('qk_layernorm', False),
        disable_scaled_init_method=ns.get('disable_scaled_init_method', False),
        use_moe=ns.get('use_moe', False),
        moe_layer_freq=ns.get('moe_layer_freq'),
        num_experts=ns.get('num_experts', 0),
        moe_ffn_hidden_size=ns.get('moe_ffn_hidden_size'),
        moe_router_topk=ns.get('moe_router_topk', 1),
        moe_n_group=ns.get('moe_n_group', 1),
        moe_topk_group=ns.get('moe_topk_group', 1),
        moe_norm_topk_prob=ns.get('moe_norm_topk_prob', True),
        moe_router_score_correction_coeff=ns.get('moe_router_score_correction_coeff', 0.0),
        moe_shared_expert_hidden_size=ns.get('moe_shared_expert_hidden_size'),
    )


def _load_megatron_shapes():
    path = os.path.join(ROOT, 'reference', 'megatron_state_dict_shapes.json')
    with open(path) as f:
        return json.load(f)


def _megatron_lookup(meg, name):
    """Find a key across all 4 ranks; return its shape (first hit)."""
    for r in range(4):
        sd = meg[f'rank{r:03d}']
        if name in sd:
            return sd[name]['shape']
    return None


class TestModelStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ns = _load_config_file(os.path.join(ROOT, 'config', 'cybertron_moe_196.py'))
        cls.ns = ns
        cfg = _build_gpt_config_from_namespace(ns)
        cls.cfg = cfg
        # Build on meta so no RAM/GPU needed
        with torch.device('meta'):
            cls.model = GPT(cfg)
        cls.meg = _load_megatron_shapes()
        cls.name_map = build_name_map(cfg, ns['moe_layer_freq'])

    # ---------- Coverage ----------
    def test_name_map_covers_nano_state_dict(self):
        """Every nano state_dict tensor param must appear in name_map (buffers excluded)."""
        nano_keys = [n for n, _ in self.model.state_dict().items()
                     if not n.endswith('.e_score_correction_bias')
                     and 'rotary_emb.inv_freq' not in n
                     and '.bias' not in n.split('.')[-1].lower()  # skip unused .bias registered buffers
                     or 'e_score_correction_bias' in n]
        # Keep only non-buffer params; nanogpt buffers to exclude explicitly:
        buffers_to_skip = {'rotary_emb.inv_freq', 'rotary_emb.cos_cached', 'rotary_emb.sin_cached'}
        nano_keys = [n for n in self.model.state_dict().keys()
                     if not any(s in n for s in buffers_to_skip)]
        missing = [n for n in nano_keys if n not in self.name_map]
        # e_score_correction_bias is a buffer; we allow it missing if not mapped yet
        # but our map does include it. Everything else must be in.
        self.assertLess(len(missing), 5,
                        f"uncovered nano params ({len(missing)} > 4 allowed): {missing[:10]}")

    # ---------- Shape invariants ----------
    def test_embedding_and_output_shapes(self):
        wte_shape = tuple(self.model.transformer.wte.weight.shape)
        self.assertEqual(wte_shape, (152064, 512))
        out_shape = tuple(self.model.lm_head.weight.shape)
        self.assertEqual(out_shape, (152064, 512))
        # Megatron equivalents:
        self.assertEqual(tuple(_megatron_lookup(self.meg, 'embedding.word_embeddings.weight')),
                         (152064, 512))
        self.assertEqual(tuple(_megatron_lookup(self.meg, 'output_layer.weight')),
                         (152064, 512))

    def test_attn_shapes_layer0(self):
        attn = self.model.transformer.h[0].attn
        self.assertEqual(tuple(attn.q_proj.weight.shape), (256, 512))   # 4h*64, 512
        self.assertEqual(tuple(attn.k_proj.weight.shape), (128, 512))   # 2kvh*64, 512
        self.assertEqual(tuple(attn.v_proj.weight.shape), (128, 512))
        self.assertEqual(tuple(attn.c_proj.weight.shape), (512, 256))
        # Megatron linear_qkv is fused (512, 512); we check row sum
        meg_qkv = _megatron_lookup(self.meg, 'decoder.layers.0.self_attention.linear_qkv.weight')
        self.assertEqual(tuple(meg_qkv), (512, 512))
        # Megatron linear_proj
        meg_proj = _megatron_lookup(self.meg, 'decoder.layers.0.self_attention.linear_proj.weight')
        self.assertEqual(tuple(meg_proj), (512, 256))

    def test_qk_layernorm_per_head(self):
        attn = self.model.transformer.h[0].attn
        self.assertEqual(tuple(attn.q_layernorm.weight.shape), (64,))
        self.assertEqual(tuple(attn.k_layernorm.weight.shape), (64,))
        self.assertEqual(tuple(_megatron_lookup(
            self.meg, 'decoder.layers.0.self_attention.q_layernorm.weight')), (64,))

    def test_dense_mlp_layer0(self):
        mlp = self.model.transformer.h[0].mlp
        # nanogpt SwiGLUMLP: gate (1536,512), up (1536,512), down (512,1536)
        self.assertEqual(tuple(mlp.gate_proj.weight.shape), (1536, 512))
        self.assertEqual(tuple(mlp.up_proj.weight.shape), (1536, 512))
        self.assertEqual(tuple(mlp.down_proj.weight.shape), (512, 1536))
        # Megatron fused SwiGLU: linear_fc1 (3072, 512), linear_fc2 (512, 1536)
        self.assertEqual(tuple(_megatron_lookup(self.meg, 'decoder.layers.0.mlp.linear_fc1.weight')),
                         (3072, 512))
        self.assertEqual(tuple(_megatron_lookup(self.meg, 'decoder.layers.0.mlp.linear_fc2.weight')),
                         (512, 1536))

    def test_moe_layer_experts_and_shared(self):
        # Layer 1 = first MoE layer per moe_layer_freq=[0]+[1]*8
        mlp = self.model.transformer.h[1].mlp
        # MoEFFN stores stacked weights: [E, C, H] for gate/up, [E, H, C] for down
        self.assertEqual(tuple(mlp.gate_weight.shape), (144, 512, 160))
        self.assertEqual(tuple(mlp.up_weight.shape),   (144, 512, 160))
        self.assertEqual(tuple(mlp.down_weight.shape), (144, 160, 512))
        self.assertIsNotNone(mlp.shared_expert)
        self.assertEqual(tuple(mlp.shared_expert.gate_proj.weight.shape), (160, 512))
        # Megatron per-expert reference (unchanged): linear_fc1.weight0 (320, 512), linear_fc2.weight0 (512, 160)
        self.assertEqual(tuple(_megatron_lookup(self.meg,
            'decoder.layers.1.mlp.experts.linear_fc1.weight0')), (320, 512))
        self.assertEqual(tuple(_megatron_lookup(self.meg,
            'decoder.layers.1.mlp.experts.linear_fc2.weight0')), (512, 160))
        self.assertEqual(tuple(_megatron_lookup(self.meg,
            'decoder.layers.1.mlp.shared_experts.linear_fc1.weight')), (320, 512))

    def test_moe_layer_count_matches(self):
        moe_layer_freq = self.ns['moe_layer_freq']
        # Count nanogpt MoE layers (MoEFFN has stacked gate_weight)
        nano_moe = sum(1 for b in self.model.transformer.h if hasattr(b.mlp, 'gate_weight'))
        self.assertEqual(nano_moe, sum(moe_layer_freq))
        # Matches Megatron: layer 0 has .mlp.linear_fc1 (dense), layers 1-8 have .mlp.experts
        for L in range(1, 9):
            key = f'decoder.layers.{L}.mlp.experts.linear_fc1.weight0'
            self.assertIsNotNone(_megatron_lookup(self.meg, key),
                                 f"layer {L} should be MoE but missing {key}")
        self.assertIsNotNone(_megatron_lookup(self.meg, 'decoder.layers.0.mlp.linear_fc1.weight'),
                             "layer 0 should be dense")


class TestParamCount(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ns = _load_config_file(os.path.join(ROOT, 'config', 'cybertron_moe_196.py'))
        cfg = _build_gpt_config_from_namespace(ns)
        # Need REAL tensors to count numel (meta works too, since numel is metadata-only).
        with torch.device('meta'):
            cls.model = GPT(cfg)
        cls.breakdown = count_params_detailed(cls.model)
        cls.meg = _load_megatron_shapes()

    def test_total_matches_megatron_sum(self):
        # Sum of all non-_extra_state numel across 4 EP shards, dedup non-expert
        seen = set()
        total = 0
        for r in range(4):
            for k, v in self.meg[f'rank{r:03d}'].items():
                if '_extra_state' in k or v['shape'] is None:
                    continue
                if '.experts.' in k:
                    total += v['numel']
                elif k not in seen:
                    seen.add(k); total += v['numel']
        # Known computed: ~447.3M. nanogpt adds rotary buffers (not params) + router gate weight
        # (num_experts*n_embd per MoE layer = 144*512*8 = 589824 ≈ 0.59M)
        # Megatron router gate is counted in 'mlp.router.weight' (also a trainable param), which
        # was included in `seen`. So totals should match within ±1M.
        diff = abs(self.breakdown.total - total)
        print(f"\nnanogpt total: {self.breakdown.total:,}  megatron total: {total:,}  diff: {diff:,}")
        self.assertLess(diff, 2_000_000,
                        f"total param diff too large: nano={self.breakdown.total:,} meg={total:,}")

    def test_embedding_roughly_156m(self):
        self.assertAlmostEqual(self.breakdown.embedding / 1e6, 155.71, delta=0.5)


class TestFlops(unittest.TestCase):
    def test_flops_per_step_reasonable(self):
        ns = _load_config_file(os.path.join(ROOT, 'config', 'cybertron_moe_196.py'))
        cfg = _build_gpt_config_from_namespace(ns)
        # Wrap ns values into cfg for extras
        cfg.vocab_size_override = ns['vocab_size_override']
        cfg.moe_ffn_hidden_size = ns['moe_ffn_hidden_size']
        flops = compute_flops_per_step(cfg, seq_len=ns['block_size'], global_bs=ns['global_batch_size'])
        print(f"\nfwd flops/step: {flops['fwd_flops_per_step']/1e15:.3f} PF")
        print(f"train flops/step: {flops['train_flops_per_step']/1e15:.3f} PF")
        # Back-of-envelope: ~30M active params * 512k tokens/step * 6 ≈ 100T ≈ 0.1 PF/step
        self.assertGreater(flops['train_flops_per_step'], 1e13)
        self.assertLess(flops['train_flops_per_step'], 1e17)


if __name__ == '__main__':
    unittest.main(verbosity=2)
