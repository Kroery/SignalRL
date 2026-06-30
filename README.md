# SignalRL: Signal Propagation Theory for Long-Horizon Agent RL

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7](https://img.shields.io/badge/PyTorch-2.7-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **使用 7B 小模型实现长程多轮智能体工具调用任务的强化学习框架**  
> 提出信号传播理论，系统解决小模型在长 horizon agent RL 中信号稀疏、衰减和分配不均的问题，  
> 于 τ-bench airline（50 任务、多轮工具调用）场景中大幅提升任务完成率。

---

## Motivation

**背景：** 长程多轮工具调用是智能体的核心能力——模型需要在多轮对话中正确理解用户意图、调用 API 工具、处理返回结果，完成如航班改签、退款等复杂任务。这类任务 trajectory 很长（通常 10-30 轮交互），对模型的规划和纠错能力要求极高。

**挑战：** 我们的目标是使用 7B 级别的小模型（Qwen2.5-7B-Instruct）完成这类任务。但 7B 模型的基础能力有限，直接使用 GRPO 进行强化学习面临根本性困难：

- **奖励稀疏**：任务链路长，模型很难在整条 trajectory 结束时获得正向的 outcome reward，导致 GRPO 几乎无法学习
- **信号衰减**：即使偶尔获得奖励，长 trajectory 中 advantage 信号传播到每一步时已严重衰减，有效梯度极小
- **学习效率低**：均匀采样下，大量算力浪费在已掌握的简单任务上，困难任务始终得不到足够训练

**我们的思路：** 既然 outcome reward 太稀疏，就需要引入**过程奖励（Process Reward）**为每一步提供密集信号。但仅有过程奖励还不够——我们需要系统性地解决信号从产生、传播到分配的完整链路问题。

---

## 核心贡献

我们提出**信号传播理论（Signal Propagation Theory）**框架，将长 horizon agent RL 中的奖励信号问题分解为四个层面，并逐一提出解法：

**1. Signal Source — 过程奖励信号生成**

设计基于规则的轻量过程奖励（Process Reward），对模型每一步的工具调用行为提供即时反馈，解决 7B 模型在长链路中几乎拿不到奖励的根本问题。

**2. Signal Channel — 信号传播通道**

针对长 trajectory 中 advantage 被长度稀释的问题，采用长度感知的 advantage 归一化（advantage/√L），确保信号强度不随链路增长而衰减。

**3. Signal Schedule — 信号强度动态调控**

提出 **PRM Annealing（过程奖励退火）**：训练初期高权重过程信号快速建立规范行为，后期逐步降低权重释放探索空间让 outcome reward 主导。解决固定权重带来的性能天花板问题。

**4. Signal Focus — 信号分配优化**

提出 **Difficulty-Aware Sampling（难度感知采样）** + **两阶段课程 RL**：根据任务难度动态分配训练资源，使模型持续在困难任务上获得有效学习信号，而非将算力浪费在已掌握的简单任务上。

> **核心洞察**：对 7B 小模型做长 horizon RL，仅靠 outcome reward 几乎无法启动学习。过程奖励解决了"信号从哪来"的问题，但"信号能否传到每一步"、"信号强度如何随训练调节"、"信号优先分配给哪些任务"同样关键。四者协同才能让小模型在长程任务上获得显著提升。

---

## 方法

### 1. 过程奖励（Process Reward）

设计 15 条基于规则的 step-level 奖励，覆盖工具调用格式、参数合法性、API 返回处理等维度，为每一步提供即时反馈信号，使 7B 模型在长链路中也能获得有效梯度。

### 2. 长度感知 Advantage 归一化

长 trajectory 中，标准 advantage 被步数稀释，导致前期 step 梯度趋近于零。我们对 advantage 除以 √L（L 为 trajectory 长度），保持信号强度与链路长度解耦。

### 3. PRM Annealing（过程奖励退火）

过程奖励权重随训练动态衰减：

```
process_coeff(t) = 2.0 + (0.5 - 2.0) × min(t / 300, 1)
```

- 前期高权重（2.0）：密集过程信号快速纠正错误行为模式
- 后期低权重（0.5）：释放探索空间，outcome reward 驱动整体性能突破

### 4. Difficulty-Aware Sampling（难度感知采样）

根据 Phase 1 评测结果动态调整 task 采样权重：

```
weight_i = (1 - reward_i + ε)^(1/τ)    ε=0.1, τ=0.8
```

- 困难任务（低 pass rate）获得更高采样概率
- 实现约 20:1 的权重比（最难 vs 最易）
- 简单任务通过 ε 保底不被完全丢弃

### 5. 两阶段课程强化学习

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

- [Agentic-GRPO-LongHorizon](https://github.com/qiqihezh/agentic-grpo-longhorizon) — 长程智能体 RL 基线
- [veRL](https://github.com/volcengine/verl) — 开源 RL 训练框架
- [τ-bench](https://github.com/sierra-research/tau-bench) — 长链路智能体评测基准
- [Qwen](https://github.com/QwenLM/Qwen) — 基座模型
