"""
Build curriculum parquet: easy tasks for Phase 1 training.

Usage:
    # 方式1: 手动指定简单 task IDs
    python scripts/train/grpo/build_curriculum_parquet.py \
        --easy-task-ids 0,1,3,5,7,12,15,20 \
        --output experiments/curriculum/train_easy.parquet

    # 方式2: 从评测日志自动筛选 (total_reward > 0 的 task)
    python scripts/train/grpo/build_curriculum_parquet.py \
        --from-eval-log experiments/eval_results.jsonl \
        --threshold 0.0 \
        --output experiments/curriculum/train_easy.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


def build_rows(task_ids: list[int], split: str) -> list[dict]:
    rows = []
    for idx, tid in enumerate(task_ids):
        rows.append({
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT}],
            "extra_info": {
                "index": idx,
                "task_id": tid,
                "split": split,
                "interaction_kwargs": {
                    "name": INTERACTION_NAME,
                    "task_id": tid,
                },
            },
            "data_source": INTERACTION_NAME,
            "reward_model": {"ground_truth": ""},
            "ability": INTERACTION_NAME,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build curriculum (easy) parquet")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--easy-task-ids", type=str, help="Comma-separated easy task IDs")
    group.add_argument("--from-eval-log", type=str, help="Path to eval JSONL with task_id + total_reward")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Min total_reward to be considered 'easy' (default: >0)")
    parser.add_argument("--output", default="experiments/curriculum/train_easy.parquet")
    args = parser.parse_args()

    if args.easy_task_ids:
        easy_ids = [int(x.strip()) for x in args.easy_task_ids.split(",")]
    else:
        log_path = Path(args.from_eval_log)
        task_rewards = {}
        with open(log_path) as f:
            for line in f:
                entry = json.loads(line)
                tid = entry.get("task_id")
                reward = entry.get("total_reward", 0.0)
                if tid not in task_rewards or reward > task_rewards[tid]:
                    task_rewards[tid] = reward
        easy_ids = [tid for tid, r in sorted(task_rewards.items()) if r > args.threshold]
        print(f"Found {len(easy_ids)} easy tasks (reward > {args.threshold})")

    rows = build_rows(easy_ids, split="easy")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"Curriculum parquet: {len(rows)} rows -> {out_path}")
    print(f"Easy task IDs: {easy_ids}")


if __name__ == "__main__":
    main()
