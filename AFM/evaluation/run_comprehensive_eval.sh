#!/bin/bash

# AFM 模型综合测评脚本
# 支持 MMLU, C-Eval, CMMLU, NQ, Story Agent Thinking 等多个数据集

set -e

# =====================================================================================================================
#                                      配置参数
# =====================================================================================================================

# 模型路径
MODEL_PATH="/mnt/tongyan.zjy/openlm/model/Qwen/Qwen3-4B-Thinking-2507"

# 输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./eval_results/${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# 数据集配置
NQ_FILE="/mnt/tongyan.zjy/data/mhqa/nq_full.jsonl"
STORY_FILE="/mnt/tongyan.zjy/data/story_room/sft/training_samples_test.jsonl"

# 评估参数
N_SHOT=5
BATCH_SIZE=4
MAX_LENGTH=2048

# 要评估的数据集（可选：mmlu ceval cmmlu nq story）
DATASETS="mmlu ceval cmmlu nq story"

# 快速测试模式（设置最大样本数，None 表示全量评估）
MAX_SAMPLES=None  # 设置为数字如 100 可快速测试

# GPU 配置
export CUDA_VISIBLE_DEVICES=0,1,2,3

# =====================================================================================================================
#                                      日志配置
# =====================================================================================================================

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/comprehensive_eval_${TIMESTAMP}.log"

# 重定向输出到日志文件
exec > >(tee -a "$LOG_FILE") 2>&1

echo "==================================================================================================="
echo "AFM 模型综合测评"
echo "==================================================================================================="
echo "开始时间: $(date)"
echo "模型路径: $MODEL_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "日志文件: $LOG_FILE"
echo "评估数据集: $DATASETS"
echo "Few-shot 数量: $N_SHOT"
echo "批处理大小: $BATCH_SIZE"
echo "==================================================================================================="

# =====================================================================================================================
#                                      环境检查
# =====================================================================================================================

echo ""
echo ">>> 检查环境..."

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3"
    exit 1
fi

echo "Python 版本: $(python3 --version)"

# 检查 GPU
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU 信息:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
else
    echo "警告: 未找到 nvidia-smi，可能未安装 CUDA"
fi

# 检查必要的 Python 包
echo ""
echo ">>> 检查 Python 依赖..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" || { echo "错误: 未安装 torch"; exit 1; }
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')" || { echo "错误: 未安装 transformers"; exit 1; }
python3 -c "import datasets; print(f'Datasets: {datasets.__version__}')" || { echo "错误: 未安装 datasets"; exit 1; }

# =====================================================================================================================
#                                      数据集检查
# =====================================================================================================================

echo ""
echo ">>> 检查数据集..."

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 检查评估数据集目录
EVAL_DATA_DIR="${REPO_ROOT}/LLaMA-Factory/evaluation"

if [[ "$DATASETS" == *"mmlu"* ]]; then
    if [ -f "${EVAL_DATA_DIR}/mmlu/mmlu.zip" ] || [ -d "${EVAL_DATA_DIR}/mmlu/data" ]; then
        echo "✓ MMLU 数据集已准备"
    else
        echo "✗ MMLU 数据集未找到，请确保已下载到 ${EVAL_DATA_DIR}/mmlu/"
    fi
fi

if [[ "$DATASETS" == *"ceval"* ]]; then
    if [ -f "${EVAL_DATA_DIR}/ceval/ceval.zip" ] || [ -d "${EVAL_DATA_DIR}/ceval/data" ]; then
        echo "✓ C-Eval 数据集已准备"
    else
        echo "✗ C-Eval 数据集未找到，请确保已下载到 ${EVAL_DATA_DIR}/ceval/"
    fi
fi

if [[ "$DATASETS" == *"cmmlu"* ]]; then
    if [ -f "${EVAL_DATA_DIR}/cmmlu/cmmlu.zip" ] || [ -d "${EVAL_DATA_DIR}/cmmlu/data" ]; then
        echo "✓ CMMLU 数据集已准备"
    else
        echo "✗ CMMLU 数据集未找到，请确保已下载到 ${EVAL_DATA_DIR}/cmmlu/"
    fi
fi

if [[ "$DATASETS" == *"nq"* ]]; then
    if [ -f "$NQ_FILE" ]; then
        NQ_COUNT=$(wc -l < "$NQ_FILE")
        echo "✓ NQ 数据集已准备 (${NQ_COUNT} 条样本)"
    else
        echo "✗ NQ 数据集未找到: $NQ_FILE"
        echo "  请下载或指定正确的路径"
    fi
fi

if [[ "$DATASETS" == *"story"* ]]; then
    if [ -f "$STORY_FILE" ]; then
        STORY_COUNT=$(wc -l < "$STORY_FILE")
        echo "✓ Story Agent Thinking 数据集已准备 (${STORY_COUNT} 条样本)"
    else
        echo "✗ Story Agent Thinking 数据集未找到: $STORY_FILE"
        echo "  请确保已运行训练脚本生成测试集"
    fi
fi

# =====================================================================================================================
#                                      运行评估
# =====================================================================================================================

echo ""
echo "==================================================================================================="
echo "开始评估..."
echo "==================================================================================================="
echo ""

cd "$REPO_ROOT"

# 构建命令
CMD="python3 AFM/evaluation/comprehensive_eval.py \
    --model_path \"$MODEL_PATH\" \
    --output_dir \"$OUTPUT_DIR\" \
    --datasets $DATASETS \
    --n_shot $N_SHOT \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LENGTH \
    --nq_file \"$NQ_FILE\" \
    --story_file \"$STORY_FILE\""

# 添加 max_samples（如果不是 None）
if [ "$MAX_SAMPLES" != "None" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

echo "执行命令:"
echo "$CMD"
echo ""

# 执行评估
eval $CMD

EXIT_CODE=$?

# =====================================================================================================================
#                                      结果汇总
# =====================================================================================================================

echo ""
echo "==================================================================================================="
echo "评估完成"
echo "==================================================================================================="
echo "结束时间: $(date)"
echo "退出码: $EXIT_CODE"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ 评估成功完成"
    echo ""
    echo "结果文件:"
    ls -lh "$OUTPUT_DIR"/*.json "$OUTPUT_DIR"/*.txt 2>/dev/null || echo "  未找到结果文件"
    echo ""
    echo "查看详细结果:"
    echo "  JSON: cat ${OUTPUT_DIR}/comprehensive_eval_*.json"
    echo "  摘要: cat ${OUTPUT_DIR}/summary_*.txt"
    echo ""
    
    # 显示摘要（如果存在）
    SUMMARY_FILE=$(ls -t "${OUTPUT_DIR}"/summary_*.txt 2>/dev/null | head -1)
    if [ -f "$SUMMARY_FILE" ]; then
        echo "==================================================================================================="
        echo "评估摘要"
        echo "==================================================================================================="
        cat "$SUMMARY_FILE"
    fi
else
    echo "✗ 评估失败，请检查日志: $LOG_FILE"
fi

echo ""
echo "完整日志: $LOG_FILE"
echo "==================================================================================================="

exit $EXIT_CODE

