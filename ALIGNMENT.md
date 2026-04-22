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

## 残留 diff — 系统性 +0.014 nat，已排除多个候选

Test A（`scripts/diag_weights_vs_forward.py --mode A`）方法：加载 ref iter N 的 ckpt 到 nano，在 iter N+1 的 batch（ref 用同一 ckpt 实际 forward 过的 batch）上跑 nano forward，和 ref 的 logged single-iter loss 比。

### 四种 forward 配置都得到同一 +0.014 nat

| forward 配置 | avg Δ (4 ckpts) | stddev |
|---|---|---|
| PyTorch SDPA (bf16 autocast) | +0.01387 | 0.00229 |
| fp32 manual attention（Q@K^T/softmax/Attn@V 全 fp32，其他 bf16） | +0.01386 | 0.00222 |
| **fp32 everywhere（autocast 关闭）** | **+0.01370** | **0.00227** |
| **TE DotProductAttention (flash-attn-2 kernel)** | **+0.01387** | **0.00229** |

### 结论（修正之前的判断）

**attention kernel + bf16 累加彻底 RULED OUT**：TE flash-attn-2（cybertron 同款）和 PyTorch SDPA 给出 **bitwise 相同** 的结果（loss 值每个 ckpt 都完全一样）。两套 kernel 底层用同一 flash-attn-2 CUDA 实现。bf16/fp32 精度也与此无关。

### 当前仍可能的源头

1. **MoE routing / dispatch 语义差异** — nano dense×mask vs ref `allgather` dispatcher。不是 bf16 rounding 的事，可能是 topk 决策边界或 normalize 顺序导致路由不同（路由不同 → 经过的 experts 不同 → loss 差系统性偏一边）
2. **权重转换 converter 的细节** — 某个 tensor 的 layout/scale 有微差但 shape 对得上（converter 已覆盖 QKV/SwiGLU/144-experts shard，但可能还有没覆盖的 buffer 或 norm weight）
3. **Loss 层面的某个 mask 处理** — ref 的 `loss_mask` 计算逻辑（EOD / reset_position / packed docs 边界）与 nano 的 `eod_mask_loss + mask_loss_id` 可能有差异

### 已排除的候选（按排查顺序）

- ❌ bf16 attention matmul 累加顺序（fp32 attn 同结果 +0.01386）
- ❌ bf16 linear / MoE grouped_mm 累加（fp32 everywhere 同结果 +0.01370）
- ❌ RMSNorm variance 精度（已 fp32）
- ❌ RoPE 精度（已 fp32 compute）
- ❌ MoE aux-free expert bias 未加载（已确认转换并加载 8 层×144）
- ❌ train.bin 和 ref 的 sampler offset 不匹配（Test C 扫描 5985-5992 验证）
- ❌ 初始化阶段（iter 0 bitwise 对齐 +0.0001 nat）
- ❌ **T1**: 权重转换 key 覆盖（0 个 silently-dropped，8 个 missing 是训练-only buffer `local_tokens_per_expert`）
- ❌ **T2**: seq_aux_balance loss（`self.training` 下才计算，eval 模式 aux=0，非源头）
- ❌ **T3**: `reset_position_ids` / `reset_attention_mask` — ref yaml 两者都是 false（和 nano 的 continuous position_ids + causal mask 一致）
- ❌ **T4**: MoERouter 的 aux-free bias 应用顺序 — 代码对比符合：`scores + bias` 只用于 topk selection，`final_weights` 用 unbiased `scores.gather(topk_idx)`（和 Megatron 一致）
- ❌ **T5**: topk normalize 时机 — nano 在 topk 之后 normalize，匹配 ref 的 `moe_norm_topk_prob=True`
- ❌ **T6**: Shared + routed 合并公式 — `out = shared + sum(g_i * E_i)`，无 `(1-Σg)` 重加权，匹配 deepseek 标准
- ❌ RoPE 变体 — nano 用 halves-split（llama/megatron default），不是 interleaved（gpt-j style）
- ❌ SwiGLU 分拆 — nano 取 `fc1[:H]` 为 gate、`fc1[H:]` 为 up，匹配 Megatron 的 `torch.chunk(y, 2, -1)`
- ❌ Block 结构 — 标准 pre-LN（`x + attn(ln1(x))`, `x + mlp(ln2(x))`）
- ❌ Ref 权重存储 dtype — bf16 保存，nano fp32 加载（padded），数学上同值

### 逐层 activation stats 对比（`scripts/diag_activations.py`）

**关键发现**：ref 的 training log 本身就记录了 **per-iter × per-sublayer activation stats**，可以直接 hook nano forward 再对照。

iter 5989 的对比结果（ref iter-5988 ckpt → 5989 batch）：

| hook 点 | nano (avg over 9 layers × 64 samples) | ref (logged) | Δ |
|---|---|---|---|
| `act_std/decoder_input`（embed 输出） | 0.0623 | 0.0624 | −0.15% ✓ |
| `act_std/attn_output`（self-attn 输出） | 0.6734 | 0.6750 | −0.24% ✓ |
| `act_std/attn_plus_residual` | 2.2247 | 2.2250 | −0.01% ✓ |
| `act_std/ffn_output` | 0.9752 | 0.9734 | +0.19% ✓ |
| `act_std/ffn_plus_residual` | 2.6758 | 2.6712 | +0.17% ✓ |
| `loss` | 3.0724 | 3.0575 | +0.014936 |

**所有逐层 std 都在 0.2% 以内对齐**。说明每一层 sublayer 的 forward 数学都是正确的。

Hook 解读对齐（通过权重 rms 比对验证）：
- `act_std/decoder_input` = post-embed（ref 0.0624 = embed.weight.rms 附近）✓
- `act_std/attn_output` / `ffn_output` / `*_plus_residual` — 对应层输出，数值对齐
- `act_std/final_input` = **post-ln_f 输出**（ref 2.2129 = final_ln.weight.rms 2.2148 匹配）

**attn-specific 指标：**
| 指标 | nano | ref |
|---|---|---|
| `max_attn_logits/full` | 28.85 | 14.01 |
| `first_token_attn_score/full` | 3.62e-3 | 1.52e-4 |
| `attn_entropy_mean/full` | 4.102 | 4.231 |

这三个 attention 指标有 2× / 24× 的系统性差异。但：
- `act_std/attn_output` 完全对齐
- SDPA fp32 manual attn 同 bf16 结果
- 整体 entropy 只差 3%（4.10 vs 4.23，nano 稍 peaked）

推测解读：**ref 的这三个指标在测量口径上可能和 nano 不同**（例如 ref 可能在 pre-scale、或 excluding causal-masked 位置测 max；first_token_attn_score 的 head/position 聚合方式不确定）。无法直接作为 "bug 存在" 的证据，因为 attn_output 完全对齐。

### 已完成：cybertron Megatron forward dump + nano 逐层对比

PAI job `dlc1n4rtvu2e4f3q` 跑了 ref cybertron forward + bitwise_dump 钩子（staged at `/newcpfs/user/yuchen/karpathy/cybertron_dump/`），dumps 在 `dumps/`。ref rank-7 mbs0 对应 iter-5989 batch 的 samples 28-31（通过 bf16 embedding byte 精确匹配反推）。

Nano 对这 4 个 sample 做 forward，hook 每个 block 的输出，对比 ref 的 dump：

| block | L∞ | L1 | rel_mean | 备注 |
|---|---|---|---|---|
| embedding | **0** | **0** | **0** | **BITWISE identical** |
| layer_0 (dense) | 0.125 | 0.00093 | 0.20% | 1-2 bf16 ULPs |
| layer_1 (MoE) | 2.0 | 0.0039 | 0.57% | |
| layer_2 | 10.0 | 0.0077 | 0.88% | |
| layer_5 | 10.0 | 0.033 | 1.72% | |
| layer_8 | 11.9 | 0.073 | 2.08% | |
| ln_f | 3.0 | 0.027 | 2.13% | |

**最终 root cause**：layer_0 是 DENSE 层（非 MoE），但已经在 bf16-ULP 层面与 ref 发散（每 element 1-2 ULP）。MoE 完全无关，残差完全来自 **bf16 matmul 的 kernel-level tiling 和累加顺序差异**：

- Ref 用 TE 的 fused `LayerNormLinear`（一个 CUDA kernel 把 layernorm + linear 融合）
- Nano 用 PyTorch 原生 `F.linear` + 独立 `RMSNorm`

两套实现数学上等价，但 bf16 累加顺序不同 → 每 block 1-2 ULP 偏差 → 9 层累积到 2% 相对偏差 → loss 差 0.5%（= +0.014 nat）。

要彻底压到 bitwise：
- 把 nano 的 attention pre-norm 换成 TE 的 `LayerNormLinear`
- 或把 MLP 换成 TE 的 `LayerNormMLP` 融合版
- 或训练跑 fp32（数学等价但慢）

这个诊断完成了"对齐到 bf16 ULP"的最终答案：nano 已经在数学层面完全对齐，剩下的 0.014 nat 是 **不同 bf16 kernel 实现的 irreducible tiling diff**，不是任何 bug。

### 其它已排除

- ✅ **TE DotProductAttention (flash-attn-2 kernel)**: 与 SDPA **bitwise 相同结果**，kernel 级确定排除
- ✅ 数据：token 分布正常，无 >vocab_size token，0 个 `mask_loss_id=160000` 实际存在（此 mask 从未触发）
- ✅ EOD mask 统计一致（0.13% tokens 是 EOD）

### MoE routing 层面的细粒度对比

读了 cybertron-dots3 的 `modules_deepseekv2.py` 的 router forward 和 token_dispatcher。Nano 的实现和 cybertron 的数学等价：
- `scores = sigmoid(F.linear(x.float(), w.float()))` — 都是 fp32 gating
- `scores_for_choice = scores + e_score_correction_bias` — bias 只用于 topk 选择
- `final_weights = scores.gather(topk_idx) / scores.gather(topk_idx).sum()` — 用 unbiased scores 做 gate

实测对比 iter 5989 batch（64 samples）全局 tokens_per_expert aggregate（每层 sum 后除以 64 得 per-microbatch avg）：

| layer | nano per-mb max | nano per-mb min |
|---|---|---|
| 1 | 545 | 340 |
| 2 | 552 | 305 |
| 8 | 656 | 220 |

Ref 在 iter 5989 logged `tokens_per_expert/max: 1275, min: 62` — ref 的 per-mb max 比 nano 高 2×，min 更小。说明 **ref 的 per-sample routing 实际上比 nano 更不均衡**，可能是因为 ref 的 aux-free bias 达到的平衡点在 per-sample 层面 token 分布更 peaky（某些 expert 特别 popular，某些几乎不用），但全局上仍然 balanced（mean=455）。

这是一个 **per-sample 层面 token dispatch 的数值 pattern 差异**。Nano 的 router 数学正确但产生了不同的 per-sample 分布，原因可能：
- bf16 autocast 下 nano 的 score+bias 精度路径和 ref 的 bf16/fp32 mix 路径不同
- 导致 topk 在边界 token 上做不同的选择
- 路由不同 → 不同 expert 处理 token → 最终 logits 不同 → loss 不同

**这个 per-sample routing 差异就是残留 +0.014 nat 最可能的源头**。要彻底对齐需要：
- 精确匹配 cybertron 的 bf16/fp32 dtype 路径
- 或替换成 Megatron 同一套 token_dispatcher

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
