# Megatron/cybertron Alignment Progress

> 将 nanogpt 对齐到 Megatron ref（scaling_moe_00196 / PAI job `dlc1q9arre48b0kx`），bf16 ULP 级别。

## 最终结果（7485 步 full run）

| 阶段 | final Δ_loss | max\|Δ\| |
|---|---|---|
| v1 原始 buggy | +1.250 nat | 2.568 |
| v2 biasfix | +0.649 | 2.208 |
| v4 fullfix | +0.634 | 1.187 |
| v5 from iter_10 | +0.600 | 1.174 |
| **v6 对齐版** | **+0.003 nat** (final 50-iter avg) | **0.147** |

整体改善 **400×**，从 1.25 nat → 3 mnat。

7485 步训练中 rolling-20-iter avg Δ 全程在 **±0.01 nat** 内（nano iter N → ref iter N+1）：

| iter | nano avg | ref avg | Δ |
|---|---|---|---|
| 0 (bitwise) | 11.9430 | 11.9430 | +0.0001 |
| 500 | 4.7010 | 4.7045 | −0.003 |
| 1000 | 3.7801 | 3.7805 | −0.0004 |
| 3000 | 3.194 | 3.185 | +0.009 |
| 5000 | 3.057 | 3.051 | +0.006 |
| 7000 | 2.871 | 2.865 | +0.006 |
| 7485 | 2.854 | 2.862 | −0.008 |

## 发现 & 修复的 bugs（按严重度排序）

### 1. use_sequential 字符串匹配 bug（最严重）

```python
use_sequential = dataset.endswith('_cybertron') or dataset.endswith('_sequential')
```

dataset `cybertron_baseline` 以 `_baseline` 结尾，所以 `use_sequential=False`，train.py 回退到原版 `get_batch` 用 **`torch.randint` 随机采样**。所有之前的 train.py 运行都是从 train.bin 里乱跳读片段，不是 Megatron 对齐的顺序数据。

Fix: `or 'cybertron' in dataset`。

### 2. Resume path 配置未透传

train.py 的 resume 路径原本只从 ckpt 恢复 arch keys。`moe_routing_type`、`use_eod_attn_mask`、`eod_token_id`、`mask_loss_id`、`seq_aux_balance_alpha` 全部 fallback 到默认值。

Fix: 通过 `globals().get()` 从当前 config 透传非架构 keys。

### 3. LR off-by-one

seed ckpt 把 `iter_num` 设为 Adam step=1（来自 PAI zero-grad patch），但应该用完成训练步数=0。导致 nano iter 1 的 LR=4.8e-6，ref iter 1 的 LR=2.4e-6，差 2×。

Fix: `seed_from_meg_ckpt.py` 用 Megatron ckpt 里的 `iteration` 字段。

### 4. DDP find_unused_parameters

原设 True（防 MoE unused expert 报错），但 grouped_mm 下所有 experts 都被触碰。`find_unused_parameters=True` 每 iter 多一次 autograd graph 遍历。

Fix: 设为 False。MFU 有改善但受限于 lm_head logits memory。

### 5. Adam optim state 未加载

对齐训练必须同时加载 ref 的 `exp_avg` / `exp_avg_sq` / `step`。不加载导致第一步 Adam bias correction 错位（step=0→1 vs step=1→2 差 2×）。

Fix: `scripts/recover_megatron_optim.py` 从 Megatron `distrib_optim.pt` 反推出 per-param state。

### 6. MoE aux-free bias 更新频率

ref 每 optim step 更新 1 次（`finalize_model_grads.py` hook）。nano 原版每 forward 更新 1 次，grad_accum=8 下频率 8×。

Fix: 拆成 forward 只累加 `local_tokens_per_expert`，`update_expert_bias()` 在 optim.step() 前调一次。

### 7. Code gaps（loss 层面）

- `eod_mask_loss=True`：target==151643 处 ignore_index
- `mask_loss_id=160000`：target==160000 处 ignore_index
- `sequence_wise_balance_loss_alpha=0.0001`：按序列 aggregation 的 aux loss

### 8. MoE routing_type 错误

ref yaml `moe_router_load_balancing_type: greedy`（flat top-K sigmoid），nano 默认 `group_limited_greedy`。

Fix: 加 `moe_routing_type` config field 分叉两条路径。

### 9. uint16 token 截断

Qwen vocab=152064 > 65535，`uint16` 存 token 会回绕。

Fix: `prepare_cybertron_data.py` + `train.py` 的 memmap dtype 改为 `int32`。

### 10. Sample stride off-by-one

prepare 按 block_size=8192 切 sample，train 按 block_size+1=8193 读，target 错位。

Fix: train.py 改 `stride=block_size`，target 取 `[s+1, s+block_size+1]`。

### 11. 权重转换 converter（`scripts/megatron_to_nano.py`）

- QKV interleaved GQA layout → 独立 q/k/v 矩阵
- SwiGLU fused `[gate; up]` → 独立 gate_proj / up_proj
- EP=4 下的 144 experts shard 合并为 nano 的 stacked tensor

## 当前配置（v6）

`config/cybertron_moe_196_from0.py`:

- 9L / 512d / 4h / 2kvh, ffn=1536, moe_ffn=160, 144 experts top-8 sigmoid
- block_size=8192, GBS=64, mb=1, grad_accum=64 (//=8 DP → 8 per rank)
- RoPE base=50000, RMSNorm eps=1e-5, SwiGLU, init_std=0.006
- AdamW β=(0.9, 0.95), eps=1e-15, wd=0.1（decay 2D params only）
- WSD LR: peak=1.2e-3, warmup 500 iter, constant to 5987, decay to 7485 (min_lr=1.2e-4)
- bf16 autocast, deterministic=True
- `use_eod_attn_mask=False`、`moe_routing_type='greedy'`、`eod_token_id=151643`、`mask_loss_id=160000`

## 性能

- 8× H100 80GB, bf16, MFU 12-13%
- mb=1 受限于 lm_head logits tensor 内存（`[mb×T, V=152064]` 在 mb≥2 会 OOM）
- 7485 步全跑 ~4.5h

## 残留 diff — 已定位为 forward kernel diff

**Test A（`scripts/diag_weights_vs_forward.py --mode A`）定位 0.014 nat 的 pure forward-kernel-diff**：

加载 ref iter N 的 ckpt 到 nano，在 iter N+1 的 batch（ref 用同一 ckpt 实际 forward 过的 batch）上跑 nano forward，和 ref 的 logged loss 比：

| ref ckpt | batch iter | nano_fwd | ref_logged | Δ |
|---|---|---|---|---|
| 1497 | 1498 | 3.3978 | 3.3873 | +0.0105 |
| 2994 | 2995 | 3.3709 | 3.3553 | +0.0157 |
| 4491 | 4492 | 3.0715 | 3.0572 | +0.0143 |
| 5988 | 5989 | 3.0725 | 3.0575 | +0.0150 |

**avg Δ = +0.0139 nat, stddev = 0.0023**

同权重、同 batch、只换 forward kernel → nano SDPA 系统性比 TE flash-attn-2 高 0.014 nat。方向在 4 个独立 ckpt 上 100% 一致。

这个 +0.014 nat 是 **2×** 训练中 20-iter rolling avg 残留 diff（+0.007 nat）—— 意味着 trained weights 其实几乎相同，中段漂移完全由 forward kernel 差异贡献（且 training 过程还略微对冲了一部分）。

### 根因候选（按优先级）

1. **PyTorch SDPA flash backend vs TE flash-attn-2 的 Q@K^T / Attn@V bf16 tiling 顺序** — 最可能
2. MoE dense×mask vs ref allgather dispatcher 的 bf16 scatter/gather 累加顺序
3. Adam 内部 bf16 update 顺序（PyTorch fused AdamW vs apex FusedAdam）

### 要压到 ±0.001 nat 的可能路径

- 把 nano attention 换成 TE flash-attn-2 同一套 kernel（最直接）
- 或在 attention 里用 manual fp32 matmul 绕开 bf16 累加
- 或整个训练转 fp32

## 关键文件

- `train.py` — 训练主循环（DDP + MoE aux-free bias update hook）
- `model.py` — 架构 + `MoERouter` + `MoEFFN` grouped_mm
- `config/cybertron_moe_196_from0.py` — 对齐配置
- `scripts/megatron_to_nano.py` — 权重 layout 转换
- `scripts/recover_megatron_optim.py` — 从 `distrib_optim.pt` 反推 per-param Adam state
- `scripts/optim_megatron_to_nano.py` — optim state layout 映射
- `scripts/seed_from_meg_ckpt.py` — 打包成 nano 格式 ckpt + 正确的 iter_num / Adam step
- `prepare_cybertron_data.py` — 从 Megatron BlendedDataset 抽 token 流到 train.bin (int32, stride=block_size)
- `dashboard/build_local.py` + `dashboard/refresh_runs.py` — 多 run loss trajectory 对比 HTML

## Dashboard

7485 步 loss trajectory 叠图 + 5 个 run 下拉切换：http://47.84.144.221:8882/dashboard/alignment_report.html

选 `nano-196-aligned-20260421_200000` 看 v6 对齐版。
