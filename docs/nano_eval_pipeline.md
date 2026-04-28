# nano → eval-pipeline 接入指南

本文档说明把 nano v2.0.0 训练出的 ckpt 接入 ml-research-agents eval-service 评测全链路（除 train/val loss 之外，跑下游 V4L 72-task benchmark）的方法。

参考资料：
- 评测全链路手册：`/prodcpfs/user/ruofeng/ml-research-agents/docs/eval_pipeline_handbook.md`
- 统一 CLI：`/prodcpfs/user/ruofeng/ml-research-agents/scripts/eval_guide.py`
- Skill：`/prodcpfs/user/ruofeng/ml-research-agents/.claude/skills/eval-guide/SKILL.md`

---

## 1. 评测全链路三层

```
nano ckpt.pt  ─┐
               │  Step A: HF 转换（待实现 converter）
               ▼
HF 目录结构  ──┐  /prodcpfs/.../{NAME}_hf/iter_NNNNNNN/
               │     • config.json (architectures: XdgMoEForCausalLM)
               │     • model.safetensors + index.json
               │     • configuration_xdgmoe.py / modeling_xdgmoe.py
               │     • tokenizer.json + tokenizer_config.json
               │
               │  Step B: eval_guide.py pre-check + submit
               ▼
eval-service ──┐  Stage 1 parity (静态) → Stage 2 (forward 数值校验)
   scheduler   │  → Worker 拉到 Ray actor → sglang 加载 → OC 跑 72 task
               │
               │  Step C: 解析结果
               ▼
last.csv      ──  /prodcpfs/user/lien/eval_results/pretrain_v4_lite/{NAME}/{ITER}/summary/last.csv
                  • V4L 聚合行: "--- PreTrainV4_Lite ---" 这一行的最后一列
                  • PASS 标准: n_real ≥ 50 of 72 任务（model column 非 `-`）
```

## 2. 已验证 (走通) 步骤

### 2a. pre-check 一个已有 HF ckpt

```bash
cd /prodcpfs/user/ruofeng/ml-research-agents
python3 scripts/eval_guide.py pre-check \
  /prodcpfs/user/data/save/data/scaling/exp12_scaling_moe_00196_ef_3.0_muon_base_1ep_hf/iter_0007000
```

输出：
```
[1/4] CKPT config
  - architectures:           ['XdgMoEForCausalLM']
  - model_type:              xdgmoe
  - kv_lora_rank:            None
[2/4] Infra fit (combo × kv joint)
  → PASS: no kv_lora_rank → non-MLA / safe
[3/4] Backend / cluster recommendation
  → backend: flashinfer
  → cluster: ds_scaling
[4/4] Gate status: GO-WITH-CAVEAT (port_status=UNKNOWN → 需 PARITY_BOOTSTRAP)
```

### 2b. dry-run submit 看 payload

```bash
python3 scripts/eval_guide.py submit \
  /prodcpfs/user/data/save/data/scaling/exp12_scaling_moe_00196_ef_3.0_muon_base_1ep_hf 7000 \
  --direction muon --dry-run
```

dry-run 输出的 JSON payload 可以拿来直接 curl POST，绕过 CLI。

### 2c. 读取已有 eval 结果

```bash
EVAL_DIR=/prodcpfs/user/lien/eval_results/pretrain_v4_lite/exp12_scaling_moe_00196_ef_3.0_muon_base_1ep/7000
head -2 ${EVAL_DIR}/summary/last.csv
# 输出:
# dataset,version,metric,mode,exp12_scaling_moe_00196_ef_3.0_muon_base_1ep-0007000
# --- PreTrainV4_Lite ---,-,-,-,19.1855
```

V4L 聚合分数 = 最后一行最后一列（这个例子 19.19）。

## 3. 待实现：nano ckpt.pt → HF 转换 (Step A)

nano 保存格式：`out_dir/ckpt.pt = {'model': state_dict, 'model_args': dict, 'optimizer': ..., 'iter_num': N}`

需要的 HF 目录结构（以 muon_base 为参考）：

```
/prodcpfs/.../{NAME}_hf/iter_NNNNNNN/
├── _SUCCESS
├── config.json                    # architectures: ["XdgMoEForCausalLM"]
├── configuration_xdgmoe.py        # 复制自 cybertron 仓库
├── generation_config.json
├── model-00001-of-00001.safetensors
├── model.safetensors.index.json
├── modeling_xdgmoe.py
├── tokenizer.json
└── tokenizer_config.json
```

参数命名映射（nano → HF）逆向 `scripts/megatron_to_nano.py`：

| nano 参数 | HF (XdgMoE) 参数 |
|---|---|
| `transformer.wte.weight` | `model.embed_tokens.weight` |
| `lm_head.weight` | `lm_head.weight` |
| `transformer.h.L.attn.q_proj.weight` (+ k_proj, v_proj) | `model.layers.L.self_attn.q_proj.weight` (+ k/v) |
| `transformer.h.L.attn.c_proj.weight` | `model.layers.L.self_attn.o_proj.weight` |
| `transformer.h.L.attn.q_layernorm.weight` | `model.layers.L.self_attn.q_norm.weight` |
| `transformer.h.L.ln_1.weight` | `model.layers.L.input_layernorm.weight` |
| `transformer.h.L.ln_2.weight` | `model.layers.L.post_attention_layernorm.weight` |
| `transformer.h.L.mlp.shared_expert.gate_proj.weight` | `model.layers.L.mlp.shared_experts.gate_proj.weight` |
| `transformer.h.L.mlp.gate_weight[E,h,in]` | `model.layers.L.mlp.experts.E.gate_proj.weight[h,in]` |
| `transformer.h.L.mlp.router.linear.weight` | `model.layers.L.mlp.gate.weight` |
| `transformer.ln_f.weight` | `model.norm.weight` |

**TODO**: 写 `scripts/nano_to_hf.py` 实现这个 mapping，输出 safetensors。可以参考：
- `/prodcpfs/user/ruofeng/ml-research-agents/sessions/gdn/eng/eval_kda/convert_kda_to_hf.py`
- cybertron `scripts/model/save_deepseekv2.py`（在 megatron_dots3.0_swa 仓库）

## 4. 待解决：scheduler 可达性

scheduler 跑在 `localhost:9803`，但当前 DSW 上没有运行。要走 submit 的话需要：
- 选项 A：在 DSW 上启动一份 scheduler + worker（重型，需要 Ray + sglang env）
- 选项 B：去 ml-research-agents 的生产 host 上 ssh，从那边 submit
- 选项 C：直接 curl POST 到生产 scheduler endpoint（需知道 IP）

短期内 nano ckpt 的评测建议用 选项 B：把 nano 转换好的 HF ckpt 放到 CPFS，然后到生产 host 用 `eval_guide.py submit` 走。

## 5. 端到端测试流程（要走通时按这个顺序）

```bash
# 1. 训练 nano（已完成，out_dir 里有 ckpt.pt）

# 2. 转换 HF（待实现）
python3 scripts/nano_to_hf.py \
  --ckpt /prodcpfs/.../nano_run/ckpt.pt \
  --out /prodcpfs/user/data/save/data/scaling/nano_v2_muon_s7_hf/iter_0007485

# 3. pre-check
python3 /prodcpfs/user/ruofeng/ml-research-agents/scripts/eval_guide.py pre-check \
  /prodcpfs/user/data/save/data/scaling/nano_v2_muon_s7_hf/iter_0007485

# 4. submit（在生产 host 上）
python3 /prodcpfs/user/ruofeng/ml-research-agents/scripts/eval_guide.py submit \
  /prodcpfs/user/data/save/data/scaling/nano_v2_muon_s7_hf 7485 --direction muon

# 5. monitor
python3 .../scripts/eval_guide.py status <job_id>

# 6. 收结果
LAST_CSV=/prodcpfs/user/lien/eval_results/pretrain_v4_lite/nano_v2_muon_s7/7485/summary/last.csv
awk -F, 'NR==2 {print "V4L:", $NF}' $LAST_CSV
```

## 6. 反模式（手册 S5 摘录）

- ❌ baseline ckpt 用方向 fork worktree 提交 eval（→ sglang crash）
- ❌ 不跑 pre-check 直接 submit（→ 被 Gate A/G 拒绝）
- ❌ eval 失败后不 troubleshoot 直接重提
- ❌ MLA 变体不传 `MODEL_TYPE=xdg_moe_mla`（→ silent fallback）

nano v2 是 baseline 类型（无 MLA / 无 variant ops），评测时 `--direction muon` 或 `--direction baseline` 即可。
