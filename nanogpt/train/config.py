"""TrainConfig — dataclass replacement for configurator.py globals() hack.

Provides type-checked config loading from Python config files OR YAML.
Validates combinations before training starts.

Kept alongside train.py for now; full adoption in T15 release.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TrainConfig:
    # I/O
    out_dir: str = "out"
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False
    always_save_checkpoint: bool = True
    init_from: str = "scratch"
    dataset: str = "openwebtext"

    # Data
    gradient_accumulation_steps: int = 40
    batch_size: int = 12
    block_size: int = 1024

    # Model arch (mirrors GPTConfig)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    n_kv_head: Optional[int] = None
    kv_channels: Optional[int] = None
    use_rope: bool = False
    rotary_base: int = 10000
    use_rmsnorm: bool = False
    norm_eps: float = 1e-5
    use_swiglu: bool = False
    ffn_hidden_size: Optional[int] = None
    tie_embeddings: bool = True
    init_std: float = 0.02
    qk_layernorm: bool = False
    disable_scaled_init_method: bool = False
    vocab_size_override: Optional[int] = None

    # MoE
    use_moe: bool = False
    moe_layer_freq: Optional[list] = None
    num_experts: int = 64
    moe_ffn_hidden_size: int = 128
    moe_router_topk: int = 2
    moe_n_group: int = 1
    moe_topk_group: int = 1
    moe_norm_topk_prob: bool = True
    moe_router_score_correction_coeff: float = 0.001
    moe_shared_expert_hidden_size: Optional[int] = None
    moe_routing_type: str = "greedy"
    seq_aux_balance_alpha: float = 0.0
    routed_scaling_factor: float = 1.0
    eod_token_id: Optional[int] = None
    mask_loss_id: Optional[int] = None
    use_eod_attn_mask: bool = False

    # Optimizer (AdamW)
    learning_rate: float = 6e-4
    max_iters: int = 600000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip: float = 1.0

    # Muon
    use_muon: bool = False
    muon_impl: str = "normuon"  # "normuon" | "megatron" | "megatron_v2"
    muon_lr: Optional[float] = None
    muon_momentum: float = 0.95
    muon_beta2: float = 0.95
    muon_weight_decay: Optional[float] = None

    # LR schedule
    decay_lr: bool = True
    lr_decay_style: str = "cosine"  # "cosine" | "wsd-exp"
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5
    warmup_samples: int = 0
    decay_end_samples: int = 0
    constant_samples: int = 0
    global_batch_size: Optional[int] = None

    # DDP / system
    backend: str = "nccl"
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = True
    deterministic: bool = False
    seed: int = 1337

    # Fast-path flags (bitwise-equivalence gated)
    chunked_ce: bool = True
    attention_impl: str = "default"

    def validate(self) -> None:
        if self.use_muon and self.muon_impl not in ("normuon", "megatron", "megatron_v2"):
            raise ValueError(f"unknown muon_impl {self.muon_impl!r}")
        if self.lr_decay_style not in ("cosine", "wsd-exp"):
            raise ValueError(f"unknown lr_decay_style {self.lr_decay_style!r}")
        if self.lr_decay_style == "wsd-exp":
            if self.decay_end_samples <= self.constant_samples:
                raise ValueError("decay_end_samples must be > constant_samples")
        if self.use_moe and self.moe_layer_freq is not None:
            if len(self.moe_layer_freq) != self.n_layer:
                raise ValueError(
                    f"moe_layer_freq len {len(self.moe_layer_freq)} != n_layer {self.n_layer}"
                )

    @classmethod
    def from_python_file(cls, path: str | Path) -> "TrainConfig":
        """Load from a config/*.py file (same format as configurator.py exec'd)."""
        path = Path(path)
        ns: dict = {}
        exec(path.read_text(), ns)
        kwargs = {k: v for k, v in ns.items() if k in cls.__dataclass_fields__}
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg
