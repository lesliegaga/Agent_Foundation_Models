#!/bin/bash
set -e

# 设置日志文件路径
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/code_agent_sft_qwen2.5_32b_$(date +%Y%m%d_%H%M%S).log"

# 重定向所有输出到日志文件
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== 训练开始时间: $(date) ==="
echo "=== 日志文件: $LOG_FILE ==="

MODEL_PATH="/mnt/tongyan.zjy/openlm/model/Qwen/Qwen2.5-Coder-32B-Instruct"

export NNODES=2 # Nodes number for training
NODE_RANK=${RANK:-0}
export NODE_RANK
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
export CUDA_VISIBLE_DEVICES

# 解析脚本所在目录，构造 DeepSpeed 配置的绝对路径，避免多进程相对路径失效
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# 项目根目录：从当前脚本向上四级到仓库根
REPO_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)
DS_CONFIG="$REPO_ROOT/LLaMA-Factory/examples/deepspeed/ds_z3_config.json"

STAGE=sft
finetuning_type=full
OUTPUT_DIR_BASE="/mnt/tongyan.zjy/model_output/AFM/AFM-CodeAgent-32B-sft"
LEARNING_RATE="1.4e-5"
BATCH_SIZE=1
GRADIENT_ACCUMULATION=4
EPOCHS=2.0
PRECISION="bf16"
CUTOFF_LEN=32768
ignore_observation=true
ignore_observation_token=observation

# datasets key of the `LLaMA-Factory/data/dataset_info.json`
DATA_DATA=code_agent_sft
TEMPALTE=qwen

# Swanlab
SWANLAB_API_KEY=ZjDMPe0DCAnwiVUndD5sB
SWANLAB_PROJECT=code_agent_sft

# 根据训练参数构造实验目录名，参考 web_agent 脚本风格
EXPERIMENT_ID="exp_lr${LEARNING_RATE}_bs${BATCH_SIZE}_ga${GRADIENT_ACCUMULATION}_ep${EPOCHS}_cl${CUTOFF_LEN}_${PRECISION}"
OUTPUT_DIR="${OUTPUT_DIR_BASE}/${EXPERIMENT_ID}"

# train
echo "Training start: $MODEL_PATH -> $OUTPUT_DIR"
llamafactory-cli train \
  --dataset_dir "$REPO_ROOT/LLaMA-Factory/data" \
  --deepspeed "$DS_CONFIG" \
  --model_name_or_path "$MODEL_PATH" \
  --trust_remote_code \
  --stage $STAGE \
  --do_train \
  --finetuning_type $finetuning_type \
  --dataset $DATA_DATA \
  --template $TEMPALTE \
  --cutoff_len $CUTOFF_LEN \
  --output_dir "$OUTPUT_DIR" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION" \
  --learning_rate "$LEARNING_RATE" \
  --num_train_epochs "$EPOCHS" \
  --${PRECISION} \
  --save_only_model true \
  --report_to swanlab \
  --use_swanlab \
  --swanlab_api_key $SWANLAB_API_KEY \
  --swanlab_project $SWANLAB_PROJECT \
  --ignore_observation_token $ignore_observation_token \
  --ignore_observation $ignore_observation
