"""
Difficulty-Aware Sampling: 根据历史 reward 动态调整 task 采样权重。

核心思路: 模型越难做对的 task 被越频繁采样，实现自适应课程学习。
相比静态 Phase 1→Phase 2 切换，消除 distribution shift，平滑过渡。

Usage:
    # 从评测日志生成带采样权重的 parquet
    python scripts/train/grpo/build_difficulty_aware_parquet.py \
        --from-eval-log experiments/prm_lite_lata_v4c/eval_step_150/eval_results.jsonl \
        --output experiments/curriculum/train_difficulty_aware.parquet

    # 手动指定每个 task 的历史 avg reward (JSON)
    python scripts/train/grpo/build_difficulty_aware_parquet.py \
        --task-rewards '{"0": 0.8, "1": 0.0, "3": 0.5}' \
        --output experiments/curriculum/train_difficulty_aware.parquet

    # 混合模式: 所有 task 都参与，但按难度加权重复采样
    python scripts/train/grpo/build_difficulty_aware_parquet.py \
        --from-eval-log experiments/eval_results.jsonl \
        --repeat-factor 3 \
        --all-tasks \
        --output experiments/curriculum/train_difficulty_aware.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / "src").is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

SYSTEM_PROMPT = (
    "# Current Date Context\n"
    "The current date is 2024-05-15 (Wednesday). "
    "When users mention dates without specifying the year, "
    "always assume they refer to 2024. "
    "All flight searches and reservations should use 2024 dates unless explicitly stated otherwise."
)

INTERACTION_NAME = "tau_bench_airline"
NUM_TASKS = 50


def compute_sampling_weights(task_rewards: dict[int, float], temperature: float = 1.0) -> dict[int, float]:
    """
    Inverse-reward weighting: tasks with lower avg reward get higher sampling probability.

    weight_i = (1 - reward_i + epsilon) ^ (1/temperature)
    Normalized to sum to 1.

    temperature < 1: more aggressive focus on hard tasks
    temperature > 1: closer to uniform
    """
    epsilon = 0.1
    weights = {}
    for tid, reward in task_rewards.items():
        weights[tid] = (1.0 - reward + epsilon) ** (1.0 / temperature)

    total = sum(weights.values())
    return {tid: w / total for tid, w in weights.items()}


def build_rows(task_ids: list[int], weights: dict[int, float], repeat_factor: int) -> list[dict]:
    """
    Build parquet rows with difficulty-aware repetition.
    Higher-weight tasks get more copies in the dataset.
    """
    if not weights:
        weights = {tid: 1.0 / len(task_ids) for tid in task_ids}

    max_weight = max(weights.values())
    rows = []
    idx = 0
    for tid in task_ids:
        w = weights.get(tid, 1.0 / len(task_ids))
        repeats = max(1, round(repeat_factor * w / max_weight))
        for _ in range(repeats):
            rows.append({
                "prompt": [{"role": "system", "content": SYSTEM_PROMPT}],
                "extra_info": {
                    "index": idx,
                    "task_id": tid,
                    "split": "difficulty_aware",
                    "sampling_weight": w,
                    "interaction_kwargs": {
                        "name": INTERACTION_NAME,
                        "task_id": tid,
                    },
                },
                "data_source": INTERACTION_NAME,
                "reward_model": {"ground_truth": ""},
                "ability": INTERACTION_NAME,
            })
            idx += 1
    return rows


def load_eval_rewards(log_path: Path) -> dict[int, float]:
    """Load per-task average rewards from eval JSONL."""
    task_rewards_sum: dict[int, float] = {}
    task_counts: dict[int, int] = {}

    with open(log_path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            tid = int(entry.get("task_id", -1))
            if tid < 0:
                continue
            reward = float(entry.get("total_reward", entry.get("reward", 0.0)))
            task_rewards_sum[tid] = task_rewards_sum.get(tid, 0.0) + reward
            task_counts[tid] = task_counts.get(tid, 0) + 1

    return {tid: task_rewards_sum[tid] / task_counts[tid] for tid in task_rewards_sum}


def load_eval_report(report_path: Path) -> dict[int, float]:
    """Load per-task pass^1 from eval_report.json (structured JSON with per_task_results)."""
    with open(report_path) as f:
        report = json.load(f)

    task_rewards = {}
    for task in report.get("per_task_results", []):
        tid = int(task["task_id"])
        task_rewards[tid] = float(task.get("pass^1", 0.0))
    return task_rewards


def main():
    parser = argparse.ArgumentParser(description="Build difficulty-aware sampling parquet")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--from-eval-log", type=str, help="Path to eval JSONL")
    group.add_argument("--from-eval-report", type=str, help="Path to eval_report.json (structured JSON with per_task_results)")
    group.add_argument("--task-rewards", type=str, help="JSON dict of {task_id: avg_reward}")

    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (< 1 = more focus on hard tasks)")
    parser.add_argument("--repeat-factor", type=int, default=3,
                        help="Max repetitions for hardest task per epoch")
    parser.add_argument("--all-tasks", action="store_true",
                        help="Include all 50 tasks (fill missing with reward=0)")
    parser.add_argument("--output", default="experiments/curriculum/train_difficulty_aware.parquet")
    args = parser.parse_args()

    if args.from_eval_log:
        task_rewards = load_eval_rewards(Path(args.from_eval_log))
    elif args.from_eval_report:
        task_rewards = load_eval_report(Path(args.from_eval_report))
    else:
        raw = json.loads(args.task_rewards)
        task_rewards = {int(k): float(v) for k, v in raw.items()}

    if args.all_tasks:
        for tid in range(NUM_TASKS):
            if tid not in task_rewards:
                task_rewards[tid] = 0.0

    task_ids = sorted(task_rewards.keys())
    weights = compute_sampling_weights(task_rewards, temperature=args.temperature)

    rows = build_rows(task_ids, weights, args.repeat_factor)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)

    print(f"Difficulty-aware parquet: {len(rows)} rows -> {out_path}")
    print(f"Tasks: {len(task_ids)}, repeat_factor={args.repeat_factor}, temperature={args.temperature}")
    print("\nTop-5 hardest (highest weight):")
    sorted_weights = sorted(weights.items(), key=lambda x: -x[1])[:5]
    for tid, w in sorted_weights:
        print(f"  task_{tid:04d}: weight={w:.4f}, avg_reward={task_rewards[tid]:.3f}")
    print("\nTop-5 easiest (lowest weight):")
    sorted_weights_easy = sorted(weights.items(), key=lambda x: x[1])[:5]
    for tid, w in sorted_weights_easy:
        print(f"  task_{tid:04d}: weight={w:.4f}, avg_reward={task_rewards[tid]:.3f}")


if __name__ == "__main__":
    main()
