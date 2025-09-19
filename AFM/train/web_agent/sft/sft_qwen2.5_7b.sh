#!/bin/bash
set -e

# 设置日志文件路径
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/web_agent_sft_qwen2.5_7b_$(date +%Y%m%d_%H%M%S).log"

# 重定向所有输出到日志文件
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== 训练开始时间: $(date) ==="
echo "=== 日志文件: $LOG_FILE ==="

MODEL_PATH="/mnt/tongyan.zjy/openlm/model/Qwen/Qwen2.5-7B-Instruct"

export NNODES=1
NODE_RANK=${RANK:-0}
export NODE_RANK
CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES

# 解析脚本所在目录，构造 DeepSpeed 配置的绝对路径，避免多进程相对路径失效
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# 项目根目录：从当前脚本向上四级到仓库根
REPO_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)
DS_CONFIG="$REPO_ROOT/LLaMA-Factory/examples/deepspeed/ds_z3_config.json"

STAGE=sft
finetuning_type=full
OUTPUT_DIR_BASE="/mnt/tongyan.zjy/model_output/AFM/web_agent_sft"
EPOCHS=4.0
PRECISION="bf16"
# CUTOFF_LEN=32768
CUTOFF_LEN=16384
ignore_observation=true
ignore_observation_token=observation

# prepare the dataset file name on your dataset_info.json about the "./AFM-WebAgent-SFT-Dataset".
DATA_DATA="web_agent_sft"
TEMPALTE=qwen

SWANLAB_API_KEY=ZjDMPe0DCAnwiVUndD5sB
SWANLAB_PROJECT=web_agent_sft

LEARNING_RATES=("1.4e-5")
WARMUP_RATIOS=("0.1")
BATCH_SIZES=(1)  # with 2 nodes
GRADIENT_ACCUMULATIONS=(16)

mkdir -p "grid_search_results"
RESULT_FILE="grid_search_results/results_$(date +%Y%m%d_%H%M%S).txt"

START_TIME=$(date +%s)

TOTAL_EXPERIMENTS=$((${#LEARNING_RATES[@]} * ${#WARMUP_RATIOS[@]} * ${#BATCH_SIZES[@]} * ${#GRADIENT_ACCUMULATIONS[@]}))
EXPERIMENT_COUNTER=0

# 网格搜索主循环
for LR in "${LEARNING_RATES[@]}"; do
    for WR in "${WARMUP_RATIOS[@]}"; do
        for BS in "${BATCH_SIZES[@]}"; do
            for GA in "${GRADIENT_ACCUMULATIONS[@]}"; do
                EXPERIMENT_COUNTER=$((EXPERIMENT_COUNTER + 1))
                EXPERIMENT_ID="exp_${EXPERIMENT_COUNTER}_lr${LR}_wr${WR}_bs${BS}_ga${GA}"
                CURRENT_OUTPUT_DIR="${OUTPUT_DIR_BASE}/${EXPERIMENT_ID}"
                SWANLAB_EXPERIMENT_NAME="qwen2.5_7b_${EXPERIMENT_ID}"
                
                EXPERIMENT_START_TIME=$(date +%s)
                
                echo "=== 开始实验 $EXPERIMENT_ID ==="
                echo "=== 参数: LR=$LR, WR=$WR, BS=$BS, GA=$GA ==="
                
                llama_factory_status=0
                # 直接执行训练命令，输出会自动重定向到日志文件
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
                  --output_dir "$CURRENT_OUTPUT_DIR" \
                  --per_device_train_batch_size "$BS" \
                  --gradient_accumulation_steps "$GA" \
                  --learning_rate "$LR" \
                  --warmup_ratio "$WR" \
                  --num_train_epochs "$EPOCHS" \
                  --${PRECISION} \
                  --save_strategy epoch \
                  --save_only_model true \
                  --report_to swanlab \
                  --logging_steps 10 \
                  --use_swanlab \
                  --swanlab_api_key $SWANLAB_API_KEY \
                  --swanlab_project $SWANLAB_PROJECT \
                  --ignore_observation_token $ignore_observation_token \
                  --ignore_observation $ignore_observation || llama_factory_status=$?
                
                EXPERIMENT_END_TIME=$(date +%s)
                EXPERIMENT_DURATION=$((EXPERIMENT_END_TIME - EXPERIMENT_START_TIME))
                
                if [ $llama_factory_status -eq 0 ]; then
                    echo "=== 实验 $EXPERIMENT_ID 成功完成 ==="
                    echo "=== 训练时长: $EXPERIMENT_DURATION 秒 ==="
                    
                    # 从日志文件中提取loss和perplexity（如果存在）
                    LOSS=$(grep -oP 'loss: \K[\d.]+' "$LOG_FILE" | tail -1 || echo "")
                    PERPLEXITY=$(grep -oP 'perplexity: \K[\d.]+' "$LOG_FILE" | tail -1 || echo "")
                    
                    echo "$LR,$WR,$BS,$GA,$LOSS,$PERPLEXITY,$EXPERIMENT_DURATION" >> grid_search_results/results_summary.csv
                else
                    echo "=== 实验 $EXPERIMENT_ID 失败 ==="
                    echo "=== 错误代码: $llama_factory_status ==="
                fi
                
                ELAPSED_TIME=$((EXPERIMENT_END_TIME - START_TIME))
                AVG_TIME_PER_EXP=$((ELAPSED_TIME / EXPERIMENT_COUNTER))
                REMAINING_EXPS=$((TOTAL_EXPERIMENTS - EXPERIMENT_COUNTER))
                ESTIMATED_REMAINING=$((AVG_TIME_PER_EXP * REMAINING_EXPS))
                
            done
        done
    done
done

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo "=== 所有实验完成 ==="
echo "=== 总耗时: $TOTAL_DURATION 秒 ==="
echo "=== 日志文件: $LOG_FILE ==="