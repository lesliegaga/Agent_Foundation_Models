#!/bin/bash
set -e

# 设置日志文件路径
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/story_agent_sft_qwen3_7b_$(date +%Y%m%d_%H%M%S).log"

# 重定向所有输出到日志文件
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== 训练开始时间: $(date) ==="
echo "=== 日志文件: $LOG_FILE ==="

# 解析脚本所在目录，构造 DeepSpeed 配置的绝对路径，避免多进程相对路径失效
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# 项目根目录：从当前脚本向上四级到仓库根
REPO_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)
DS_CONFIG="$REPO_ROOT/LLaMA-Factory/examples/deepspeed/ds_z3_config.json"

# Data paths
DATA_DIR="/mnt/tongyan.zjy/data/story_room/sft"
FULL_DATA="${DATA_DIR}/training_samples.jsonl"
TRAIN_DATA="${DATA_DIR}/training_samples_train.jsonl"
TEST_DATA="${DATA_DIR}/training_samples_test.jsonl"
TEST_SIZE=200

# Split data into train and test if not already done
if [ ! -f "$TRAIN_DATA" ] || [ ! -f "$TEST_DATA" ]; then
    echo "=== Splitting dataset into train and test sets ==="
    python3 "$REPO_ROOT/AFM/data/split_train_test.py" \
        "$FULL_DATA" \
        "$TRAIN_DATA" \
        "$TEST_DATA" \
        --test_size $TEST_SIZE \
        --seed 42
    echo "=== Data split completed ==="
else
    echo "=== Using existing train/test split ==="
    echo "Train data: $TRAIN_DATA"
    echo "Test data: $TEST_DATA"
fi

MODEL_PATH="/mnt/tongyan.zjy/openlm/model/Qwen/Qwen3-4B-Thinking-2507"

export NNODES=1 # Nodes number for training
NODE_RANK=${RANK:-0}
export NODE_RANK
# CUDA_VISIBLE_DEVICES=0,1,2,3
CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES

STAGE=sft
finetuning_type=full
# 基础输出目录，具体实验目录根据训练参数拼接
OUTPUT_DIR_BASE="/mnt/tongyan.zjy/model_output/AFM/AFM-StoryAgent-7B-sft"
LEARNING_RATE="3e-5"
BATCH_SIZE=
GRADIENT_ACCUMULATION=4
EPOCHS=2.0
PRECISION="bf16"
CUTOFF_LEN=32768
ignore_observation=true
ignore_observation_token=observation

# datasets key of the `LLaMA-Factory/data/dataset_info.json`
TRAIN_DATASET=story_agent_thinking_sft_train
EVAL_DATASET=story_agent_thinking_sft_test
TEMPALTE=qwen

# Evaluation settings
EVAL_STEPS=100  # Evaluate every 100 steps
EVAL_STRATEGY="steps"  # Can be "steps" or "epoch"
EVAL_BATCH_SIZE=2

# Logging settings (log train metrics to Swanlab every N steps)
LOGGING_STEPS=10
LOGGING_STRATEGY="steps"

# Swanlab
SWANLAB_API_KEY=ZjDMPe0DCAnwiVUndD5sB
SWANLAB_PROJECT=story_agent_sft

# 根据训练参数构造实验目录名，参考 web_agent 脚本风格
EXPERIMENT_ID="exp_lr${LEARNING_RATE}_bs${BATCH_SIZE}_ga${GRADIENT_ACCUMULATION}_ep${EPOCHS}_cl${CUTOFF_LEN}_${PRECISION}"
OUTPUT_DIR="${OUTPUT_DIR_BASE}/${EXPERIMENT_ID}"

# train
echo "=== Training Configuration ==="
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Train Dataset: $TRAIN_DATASET"
echo "Eval Dataset: $EVAL_DATASET"
echo "Eval Strategy: $EVAL_STRATEGY every $EVAL_STEPS steps"
echo "Eval on Start: ENABLED - will evaluate before training starts"
echo "=============================="
echo ""
echo "Starting training..."

llamafactory-cli train \
  --dataset_dir "$REPO_ROOT/LLaMA-Factory/data" \
  --deepspeed "$DS_CONFIG" \
  --model_name_or_path "$MODEL_PATH" \
  --trust_remote_code \
  --stage $STAGE \
  --do_train \
  --do_eval \
  --finetuning_type $finetuning_type \
  --dataset $TRAIN_DATASET \
  --eval_dataset $EVAL_DATASET \
  --template $TEMPALTE \
  --cutoff_len $CUTOFF_LEN \
  --output_dir "$OUTPUT_DIR" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION" \
  --learning_rate "$LEARNING_RATE" \
  --num_train_epochs "$EPOCHS" \
  --eval_strategy "$EVAL_STRATEGY" \
  --eval_steps "$LOGGING_STEPS" \
  --logging_steps "$LOGGING_STEPS" \
  --${PRECISION} \
  --save_only_model true \
  --report_to swanlab \
  --use_swanlab \
  --swanlab_api_key $SWANLAB_API_KEY \
  --swanlab_project $SWANLAB_PROJECT \
  --ignore_observation_token $ignore_observation_token \
  --ignore_observation $ignore_observation \
  --enable_thinking_mode \
  --thinking_separator "</think>\n" \
  --eval_on_start

echo ""
echo "=== 训练完成时间: $(date) ==="
