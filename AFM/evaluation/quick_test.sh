#!/bin/bash

# 快速测试脚本 - 每个数据集只评估少量样本
# 用于验证评估流程是否正常运行

set -e

echo "==================================================================================================="
echo "快速测试模式 - 每个数据集仅评估 50 个样本"
echo "==================================================================================================="

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$REPO_ROOT"

# 配置参数
MODEL_PATH="/mnt/tongyan.zjy/openlm/model/Qwen/Qwen3-4B-Thinking-2507"
OUTPUT_DIR="./eval_results/quick_test_$(date +%Y%m%d_%H%M%S)"
NQ_FILE="/home/tongyan.zjy/workspace/git/Agent_Foundation_Models/AFM/data/mhqa_agent/test_benchmarks/nq_full.jsonl"
STORY_FILE="/mnt/tongyan.zjy/data/story_room/sft/training_samples_test.jsonl"

echo "模型: $MODEL_PATH"
echo "输出: $OUTPUT_DIR"
echo ""

# CUDA_VISIBLE_DEVICES=0,1,2,3
CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES

# 运行快速测试
python3 AFM/evaluation/comprehensive_eval.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --datasets story nq \
    --n_shot 3 \
    --batch_size 2 \
    --max_length 16384 \
    --nq_file "$NQ_FILE" \
    --story_file "$STORY_FILE" \
    --max_samples 10

echo ""
echo "==================================================================================================="
echo "快速测试完成！"
echo "==================================================================================================="
echo "查看结果:"
echo "  ls -lh $OUTPUT_DIR"
echo ""
echo "如果一切正常，运行完整评估:"
echo "  bash AFM/evaluation/run_comprehensive_eval.sh"
echo "==================================================================================================="

