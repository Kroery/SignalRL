# SignalRL: Signal Propagation Theory for Long-Horizon Agent RL

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7](https://img.shields.io/badge/PyTorch-2.7-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **使用 7B 模型实现长程多轮智能体工具调用任务的强化学习框架**  
> 提出信号传播理论，系统解决7B模型在长 horizon agent RL 中信号稀疏、衰减和分配不均的问题，  
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

设计基于规则的 step-level 过程奖励，对每一步工具调用行为评分（mean-based，避免长轨迹惩罚被摊平），使 7B 模型在长链路中也能获得有效梯度。具体规则包括：

**惩罚项（Penalties）：**
- **P1 占位符惩罚**：工具参数中包含占位符（如 "xxx", "placeholder"），写入类工具 −0.05，其他 −0.03
- **P2 冗余调用惩罚**：最近 3 步内重复相同工具+参数，−0.03
- **P3 错误重复惩罚**：上一步报错后，用完全相同参数重试 −0.04（若改变参数则奖励 +0.05，鼓励纠错）
- **P4 越级惩罚**：未先查询信息就直接升级（escalation），−0.10（已查询过则 −0.05）
- **P5 无推理惩罚**：整条 trajectory 无任何 think 步骤且长度 ≥ 3，−0.05
- **P8 长度惩罚**：trajectory 超过 8 步后每多一步 −0.01，抑制无效循环
- **P9 浅思考惩罚**：工具调用前 assistant 内容少于 30 字符（"没想清楚就调工具"），−0.02

**奖励项（Bonuses）：**
- **B1 数据链奖励**：参数值来自之前步骤提取的实体（写入类 +0.08，其他 +0.04），鼓励信息串联
- **B2 探索奖励**：首次使用某个 read 类工具 +0.01，鼓励多样信息收集
- **B4/B5 推理奖励**：有效 think 步骤 +0.01（排除连续 think、末尾 think、think 后仍犯错等情况）
- **B7 工具多样性奖励**：使用 ≥ 3 种不同 read 工具 +0.01

最终分数 clip 到 [−0.5, 0.5]，与 outcome reward 加权求和作为总奖励。

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

| Metric | Vanilla GRPO | 过程奖励 | 长度归一化 | 课程学习 | **PRM Annealing + 难度采样** | Δ vs Vanilla |
|--------|-------------|-----------|------------|-----------|-------------------------------|-------------|
| **Overall pass^1** | 0.175 | 0.140 | 0.185 | 0.300 | 0.385 | **+120%** |
| **Generalization** | 0.071 | 0.059 | 0.088 | 0.075 | 0.110 | **+55%** |
| **Error Rate** | 0.200 | 0.365 | 0.290 | 0.255 | 0.015 | **−92%** |

**实验分析：**

- **过程奖励单独使用反而降低整体性能**（0.175→0.140）：密集过程信号虽然提供了梯度，但固定的高权重规则约束过强，限制了策略探索，Error Rate 也随之升高。这说明过程奖励需要配合合适的信号调控机制才能发挥作用。
- **长度归一化恢复并略超 baseline**（0.185）：解决了 advantage 被长 trajectory 稀释的问题，信号有效传播后模型能正确学习，Error Rate 开始下降。
- **课程学习带来跳跃式提升**（0.185→0.300）：仅在简单任务上训练让模型快速建立基础能力，pass^1 提升 62%。但泛化性略有下降（0.088→0.075），说明简单任务子集的分布有限。
- **PRM Annealing + 难度采样实现最终突破**（0.300→0.385）：退火机制释放了后期探索空间，难度感知采样将训练资源集中在困难任务上，Error Rate 从 25.5% 降至 1.5%，泛化性也回升至 0.110。

> 完整流水线相较 Vanilla GRPO：Overall pass^1 提升 120%，Error Rate 下降 92%，验证了信号传播理论四个层面协同优化的有效性。

---

## 快速开始

### 环境搭建

```bash
bash setup.sh
conda activate agentrl
cd agentic-grpo-longhorizon
```

### Step 1: SFT 数据采集与筛选

使用 Qwen2.5-72B-Instruct-AWQ 作为 policy 在 τ-bench airline 50 个任务上采集成功 trajectory，筛选出高质量 SFT 训练数据。

**采集策略：**
- 每个 task 采用 best-of-16 采样，分层温度（0.0×4, 0.5×4, 0.8×4, 1.0×4）
- 仅保留 `success=True` 的 trajectory
- 上下文超过 35000 字符（截断污染）的 trajectory 永久排除

**数据划分：**
- 50 个 task 中，19 个至少有一条成功 trajectory（覆盖率 38%），共采集到 **67 条**成功轨迹
- 手动选取 10 个 task 作为 unseen task（holdout），剩余 40 个为 seen task
- Seen task 上的 **45 条**成功轨迹写入 `train.jsonl`，用于 SFT 训练
- Unseen task 上的 **22 条**成功轨迹写入 `holdout_train.jsonl`，**不参与训练**
- Holdout 的目的：保留一组模型"没见过的 task"，用来评估 GRPO 训练后模型的泛化能力——如果模型在 unseen task 上也能表现好，说明学到了通用的 agent 能力，而不是过拟合到具体 task

```bash
# 启动 72B 模型 vLLM 服务（GPU0 作为 policy，GPU1 作为 user simulator）
bash scripts/vllm_server/72b.sh

# 采集并筛选 SFT 数据
python scripts/train/sft/collect_sft_data.py \
    --config configs/train/sft/sft_collect_airline.yaml
```

输出：
- `experiments/sft_collect_airline/train.jsonl`（45 条 seen task 成功轨迹）
- `experiments/sft_collect_airline/split.json`（seen/unseen task 划分）

### Step 2: SFT 训练

基于 Qwen2.5-7B-Instruct 进行 LoRA 微调（r=16, α=32），使用 45 条成功轨迹作为监督数据：

```bash
python scripts/train/sft/sft_train.py \
    --config configs/train/sft/sft_airline_lora.yaml
```

训练完成后合并 LoRA 权重：

```bash
python scripts/train/sft/merge_lora.py \
    --base <path-to-Qwen2.5-7B-Instruct> \
    --adapter experiments/sft_lora \
    --out experiments/sft_lora_merged
```

### Step 3: Phase 1 GRPO 训练（简单任务课程学习）

从 SFT 数据采集阶段的成功率中筛选**简单任务**用于 Phase 1 课程学习。我们选取 72B 模型成功率 ≥ 25%（≥ 4/16）的 seen task 作为简单任务子集，共 7 个 task：

| Task ID | 72B 成功率 |
|---------|-----------|
| 38 | 9/16 (56%) |
| 21 | 8/16 (50%) |
| 40 | 6/16 (38%) |
| 34 | 6/16 (38%) |
| 47 | 5/16 (31%) |
| 44 | 4/16 (25%) |
| 37 | 4/16 (25%) |

生成简单任务训练数据：

```bash
python scripts/train/grpo/build_curriculum_parquet.py \
    --easy-task-ids 38,21,40,34,47,44,37 \
    --output experiments/curriculum/train_easy.parquet
```

从 SFT checkpoint 启动 Phase 1 训练（仅在简单任务上训练，使用过程奖励 + 长度感知 advantage）：

```bash
python -m verl.trainer.main_ppo \
    --config-path=$(pwd)/configs/train/grpo \
    --config-name=prm_lite_lata.yaml
```

> Phase 1 的目标：在简单任务上快速建立基础的工具调用和多轮对话能力，为 Phase 2 的全量任务训练提供一个好的起点。

### Step 4: Phase 1 评测 & 难度感知数据生成

评测 Phase 1 最佳 checkpoint（step 150），获取每个 task 的 pass rate 作为难度指标：

```bash
bash scripts/eval/eval_phase1.sh 150
```

基于评测结果生成难度加权训练数据（困难任务获得更高采样权重）：

```bash
python scripts/train/grpo/build_difficulty_aware_parquet.py \
    --from-eval-report experiments/prm_lite_lata_v4c/eval_step_150/eval_report.json \
    --all-tasks --repeat-factor 3 --temperature 0.8 \
    --output experiments/curriculum/train_difficulty_aware.parquet
```

### Step 5: Phase 2 GRPO 训练（难度感知 + PRM Annealing）

从 Phase 1 step 150 checkpoint 热启动，使用难度加权数据 + 过程奖励退火继续训练（Ref 模型锚定原始 SFT）：

```bash
python -m verl.trainer.main_ppo \
    --config-path=$(pwd)/configs/train/grpo \
    --config-name=prm_lite_lata_ph2.yaml
```

### Step 6: 最终评测

```bash
bash scripts/eval/eval_exp4_prm_lite_lata.sh
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
