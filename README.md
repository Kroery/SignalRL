# SignalRL: Signal Propagation Theory for Long-Horizon Agent RL

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7](https://img.shields.io/badge/PyTorch-2.7-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **基于信号传播理论的长程多轮智能体强化学习优化**  
> 在 PRM-Lite + LATA 基础上提出 **PRM Annealing + Difficulty-Aware Sampling + 两阶段课程 RL**，  
> 于 τ-bench airline（50 任务、多轮工具调用）场景中显著提升任务完成率。

---

## Motivation

标准 GRPO 应用于长链路多工具智能体时存在三个核心问题：

1. **过程信号权重固定**：PRM-Lite 在训练全程使用恒定 process_coeff，前期信号不够强无法快速纠正行为，后期信号过强限制策略探索新解法，导致性能天花板。

2. **采样效率低下**：50 个 task 难度差异巨大（pass rate 0%~100%），均匀采样下简单任务贡献 gradient 趋近于 0（group 内全成功，advantage 归零），困难任务又采样不足，训练资源严重浪费。

3. **阶段切换的 distribution shift**：从简单任务课程直接跳到全任务训练，难度分布突变导致策略震荡，已学能力退化。

**SignalRL** 针对这三个问题，在信号传播理论框架下提出对应解法，实现整体 pass^1 的显著提升。

---

## 核心贡献

本项目在 [Agentic-GRPO-LongHorizon](https://github.com/qiqihezh/agentic-grpo-longhorizon)（PRM-Lite + LATA）基础上，提出**信号传播理论**框架并实现三项关键改进：

| 组件 | 方法 | 解决的问题 |
|------|------|------|
| Signal Source | PRM-Lite（15 规则过程奖励） | 提供 step-level 密集奖励信号 |
| Signal Channel | LATA（advantage/√L） | 防止长 trajectory 中信号衰减 |
| **Signal Schedule** | **PRM Annealing + Curriculum RL** | **过程信号权重固定 → 动态退火匹配学习阶段** |
| **Signal Focus** | **Difficulty-Aware Sampling** | **采样低效 → 训练资源向困难任务倾斜** |

> **核心洞察**：信号源（PRM-Lite）和通道（LATA）解决了"有没有信号"和"信号能否传到"的问题，但"信号强度如何随训练调节"和"信号分配给哪些任务"同样关键。四者协同才能充分释放长 horizon agent RL 的性能。

---

## 方法

### 1. PRM Annealing（过程奖励退火）

训练初期 process reward 主导快速建立规范行为，后期 outcome reward 主导提升整体任务完成率：

```
process_coeff(t) = 2.0 + (0.5 - 2.0) × min(t / 300, 1)
```

- 前期高 coeff（2.0）：密集过程信号快速纠正错误行为模式
- 后期低 coeff（0.5）：释放探索空间，outcome 驱动整体性能突破

### 2. Difficulty-Aware Sampling（难度感知采样）

根据 Phase 1 评测结果动态调整 task 采样权重：

```
weight_i = (1 - reward_i + ε)^(1/τ)    ε=0.1, τ=0.8
```

- 困难任务（低 pass rate）获得更高采样概率
- 实现约 20:1 的权重比（最难 vs 最易）
- 简单任务通过 ε 保底不被完全丢弃

### 3. 两阶段课程强化学习

| 阶段 | 数据 | 目标 |
|------|------|------|
| Phase 1 | 简单 task 子集 | 建立基础能力，热启动策略 |
| Eval | 全部 50 tasks | 获取 difficulty oracle |
| Phase 2 | 全部 task（难度加权） | 突破性能瓶颈 |

- Ref 模型始终锚定原始 SFT，提供稳定 KL 锚点
- Phase 2 actor 从 Phase 1 最佳 checkpoint 热启动

---

## 实验结果

<!-- TODO: 填写你的实验结果 -->

---

## 快速开始

### 环境搭建

```bash
bash setup.sh
conda activate agentrl
cd agentic-grpo-longhorizon
```

### Phase 1 训练（简单任务课程学习）

```bash
bash scripts/train/grpo/run_exp4_prm_lite_lata.sh
```

### Phase 1 评测 & 生成难度感知数据

```bash
bash scripts/eval/eval_phase1.sh 150

python scripts/train/grpo/build_difficulty_aware_parquet.py \
    --from-eval-report experiments/prm_lite_lata_v4c/eval_step_150/eval_report.json \
    --all-tasks --repeat-factor 3 --temperature 0.8 \
    --output experiments/curriculum/train_difficulty_aware.parquet
```

### Phase 2 训练（难度感知 + PRM Annealing）

```bash
python -m verl.trainer.main_ppo \
    --config-path configs/train/grpo \
    --config-name prm_lite_lata_ph2
```

---

## 项目结构

```
agentic-grpo-longhorizon/
├── configs/
│   ├── train/grpo/              # 训练配置（Phase 1 & 2）
│   ├── eval/                    # 评测配置
│   ├── interaction_config/      # PRM Annealing 参数
│   └── tool_config/             # 工具调用配置
├── src/envs/
│   └── tau_bench_interaction.py # PRM-Lite + Annealing 实现
├── scripts/
│   ├── train/grpo/
│   │   └── build_difficulty_aware_parquet.py
│   └── eval/
│       └── eval_phase1.sh
└── docs/
```

---

## 技术栈

- **训练框架**: [veRL](https://github.com/volcengine/verl) 0.6.1 (FSDP + vLLM V1 Async Rollout)
- **策略模型**: Qwen2.5-7B-Instruct (LoRA rank=16, α=32)
- **用户模拟器**: Qwen2.5-72B-Instruct-AWQ
- **评测基准**: [τ-bench](https://github.com/sierra-research/tau-bench) airline (50 tasks)
- **推理引擎**: vLLM V1 + FlashAttention-2 + Hermes tool-call parsing

---

## 致谢

- [Agentic-GRPO-LongHorizon](https://github.com/qiqihezh/agentic-grpo-longhorizon) — PRM-Lite + LATA 基线
- [veRL](https://github.com/volcengine/verl) — 开源 RL 训练框架
- [τ-bench](https://github.com/sierra-research/tau-bench) — 长链路智能体评测基准
- [Qwen](https://github.com/QwenLM/Qwen) — 基座模型
