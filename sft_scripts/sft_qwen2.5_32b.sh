#!/bin/bash
# 极简版llama训练脚本
set -e

# 模型参数
MODEL_PATH="xxx"

# 训练参数
export NNODES=2 # 设置节点数。同步设定starfire 的task上节点数
NODE_RANK=${RANK:-0}
export NODE_RANK
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # 设置可见的CUDA编号。同步设定starfire 的task上的CUDA数量

STAGE=dpo
finetuning_type=full
OUTPUT_DIR="saves/qwen2.5_32b"
LEARNING_RATE="1e-6"
BATCH_SIZE=1
GRADIENT_ACCUMULATION=1
EPOCHS=4.0
PRECISION="bf16"
CUTOFF_LEN=30000
ignore_observation=true
ignore_observation_token=observation

# 数据集参数
DATA_DATA=merged_other_tools_dataset
TEMPALTE=qwen

# Swanlab参数
SWANLAB_API_KEY=xxx
SWANLAB_PROJECT=xxx


# 执行训练命令
echo "开始训练: $MODEL_PATH -> $OUTPUT_DIR"
llamafactory-cli train \
  --deepspeed "./examples/deepspeed/ds_z3_config.json" \
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
