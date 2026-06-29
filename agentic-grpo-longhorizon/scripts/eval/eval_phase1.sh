#!/bin/bash
# 评估 Phase 1 (简单任务课程学习) 的 checkpoint
# 用于生成 eval_report.json，作为 Phase 2 难度感知采样的输入
#
# 前置条件：
#   1. 72B user sim 已在 port 8001 运行：
#      bash scripts/vllm_server/72b.sh
#   2. 当前目录为 agentic-grpo-longhorizon/
#   3. Phase 1 模型已 merge：experiments/prm_lite_lata_v4c/checkpoints/global_step_XXX/actor/merged
#
# 用法：
#      bash scripts/eval/eval_phase1.sh
#      bash scripts/eval/eval_phase1.sh 150        # 指定 step

set -e

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$(dirname "$0")/../.."

# 默认评测最后一个 step，也可通过参数指定
STEP=${1:-150}
SPLIT_FILE="experiments/sft_collect_airline/split.json"

MODEL_PATH="experiments/prm_lite_lata_v4c/checkpoints/global_step_${STEP}/actor/merged"
OUTPUT_DIR="experiments/prm_lite_lata_v4c/eval_step_${STEP}"

echo "========================================"
echo "  Phase 1 课程学习评测"
echo "  Model: ${MODEL_PATH}"
echo "  Step: ${STEP}"
echo "  Split file: ${SPLIT_FILE}"
echo "========================================"

# 如果 merged 目录不存在，尝试用 merge_lora.py 生成
if [ ! -f "${MODEL_PATH}/model.safetensors" ] && ! ls "${MODEL_PATH}"/model-*.safetensors >/dev/null 2>&1; then
    ADAPTER_PATH="experiments/prm_lite_lata_v4c/checkpoints/global_step_${STEP}/actor/lora_adapter"
    if [ -d "${ADAPTER_PATH}" ]; then
        echo "Merged model not found. Running merge_lora.py..."
        python scripts/train/sft/merge_lora.py \
            --base experiments/sft_lora_merged \
            --adapter "${ADAPTER_PATH}" \
            --out "${MODEL_PATH}"
    else
        echo "ERROR: Neither merged model nor LoRA adapter found at step ${STEP}"
        echo "  Expected: ${MODEL_PATH}/model.safetensors"
        echo "  Or: ${ADAPTER_PATH}/"
        exit 1
    fi
fi

# 清理可能残留的 8000 端口进程
echo "Cleaning up port 8000..."
pkill -f "api_server.*port 8000" || true
sleep 3

mkdir -p "${OUTPUT_DIR}"

# 启动 policy server (后台)
echo "Starting policy server on port 8000 (GPU 0)..."
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "Qwen/Qwen2.5-7B-Instruct" \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.82 \
    --max-model-len 16384 \
    --max-num-seqs 8 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --trust-remote-code \
    > "${OUTPUT_DIR}/vllm_server.log" 2>&1 &

SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"

# 等待 vLLM 启动完成
echo "Waiting for vLLM to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "vLLM is ready!"
        break
    fi
    sleep 5
    if [ $i -eq 60 ]; then
        echo "ERROR: vLLM failed to start within 5 minutes. Check ${OUTPUT_DIR}/vllm_server.log"
        kill $SERVER_PID || true
        exit 1
    fi
done
sleep 5

# 运行评估（全部50个task，用于生成难度感知采样权重）
echo "Running eval on all 50 tasks..."
python scripts/eval/eval_sft.py \
    --config configs/eval/prm_lite_lata/eval_prm_lite_lata_step${STEP}.yaml \
    --split-file "${SPLIT_FILE}"

# 关闭 policy server
echo "Stopping policy server..."
kill $SERVER_PID || true
sleep 3
pkill -f "api_server.*port 8000" || true

echo ""
echo "========================================"
echo "  Phase 1 评测完成！"
echo "  结果: ${OUTPUT_DIR}/eval_report.json"
echo ""
echo "  下一步: 生成难度感知 parquet"
echo "  python scripts/train/grpo/build_difficulty_aware_parquet.py \\"
echo "      --from-eval-report ${OUTPUT_DIR}/eval_report.json \\"
echo "      --all-tasks --repeat-factor 3 --temperature 0.8 \\"
echo "      --output experiments/curriculum/train_difficulty_aware.parquet"
echo "========================================"
