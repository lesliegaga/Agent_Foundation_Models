#!/bin/bash
# 这是禁用CUDA图的fallback版本，用于解决CUDA图初始化失败问题

set -x

ulimit -n 65535
# =====================================================================================================================
#                                      Param
# =====================================================================================================================
ACTOR_LR=1e-6
TRAIN_BS=64   # 进一步减少批量大小
PPO_MINI_BS=16
GEN_BS=64     # 进一步减少批量大小
EPOCHS=100
STEPS=2000
N=8
PPO_MICRO_BSZ_PER_GPU=1  # 减少到1
LOG_PROB_MICRO_BSZ_PER_GPU=8  # 减少到8
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
# context window
max_prompt_length=$((1024 * 4))
max_response_length=$((1024 * 16))
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
# performance related param
SP_SIZE=4
GEN_TP=4
use_dynamic_bsz=False
# =====================================================================================================================
#                                      Env
# =====================================================================================================================
export WANDB_MODE="offline"
CURRENT_DIR=$(pwd)
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NNODES=1
export PROJECT_NAME="agent_foundation_models"
SAVE_MODEL_FOLDER="${CURRENT_DIR}/experiments"
export EXPERIMENT_NAME="DAPO-QWEN7B-CodeAgent-NoCudaGraph"
export BASE_MODEL="/mnt/tongyan.zjy/model_output/AFM/AFM-CodeAgent-7B-sft/exp_lr3e-5_bs1_ga4_ep2.0_cl32768_bf16"
export VLLM_ATTENTION_BACKEND=XFORMERS

# Ray环境配置
export RAY_TMPDIR="/mnt/tongyan.zjy/tmp/ray"
export RAY_DEDUP_LOGS=0
export RAY_NUM_PRESTART_WORKERS=0
export RAY_MAXIMUM_STARTUP_CONCURRENCY=1  # 减少并发
export RAY_WORKER_REGISTER_TIMEOUT_SECONDS=600
export RAY_memory_usage_threshold=0.95
export RAY_memory_monitor_refresh_ms=2000
export RAY_object_spilling_threshold=0.7
export RAY_NODE_IP_ADDRESS="127.0.0.1"
export RAY_DASHBOARD_HOST="127.0.0.1"

# PyTorch distributed配置 - 更保守的设置
export TORCH_DISTRIBUTED_INIT_TIMEOUT=1200  # 增加到20分钟
export NCCL_TIMEOUT=1200
export NCCL_SOCKET_TIMEOUT=1200
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=WARN  # 减少日志量
export NCCL_DEBUG_SUBSYS=INIT,COLL

TRAIN_DATASETS="${CURRENT_DIR}/amap_search_rag_AFM-CodeAgent-RL-Dataset_20250924165348/CodeAgentRLDataset.parquet"
VAL_DATASETS="${CURRENT_DIR}/amap_search_rag_AFM-CodeAgent-RL-Dataset_20250924165348/CodeAgentRLDataset.parquet"

# =====================================================================================================================
#                                      Tool
# =====================================================================================================================
CODE_CONFIG="${CURRENT_DIR}/verl/verl/tools/config/code_tool_config/code_executor.yaml"
SEARCH_CONFIG="${CURRENT_DIR}/verl/verl/tools/config/search_tool_config/training_servers_config.yaml"
AFM_CONFIG="${CURRENT_DIR}/verl/verl/tools/config/afm_tool_config/afm_tool_config.yaml" 

# =====================================================================================================================
#                                      Train
# =====================================================================================================================
cd verl

# 创建日志目录
mkdir -p logs

echo "[no_cuda_graph] ====== 禁用CUDA图版本启动 ======"
echo "[no_cuda_graph] 这个版本专门解决CUDA图初始化同步问题"
echo "[no_cuda_graph] 配置摘要:"
echo "[no_cuda_graph]   GPU并行度: $GEN_TP"
echo "[no_cuda_graph]   训练批量大小: $TRAIN_BS (减少)"
echo "[no_cuda_graph]   生成批量大小: $GEN_BS (减少)"
echo "[no_cuda_graph]   CUDA图: 禁用"
echo "[no_cuda_graph]   GPU内存利用率: 0.3 (保守)"

# 强制清理所有Ray进程
echo "[no_cuda_graph] Cleaning up Ray resources..."
ray stop --force >/dev/null 2>&1 || true
pkill -f ray:: >/dev/null 2>&1 || true
pkill -f raylet >/dev/null 2>&1 || true
sleep 3

# 清理Ray临时目录
if [ -d "$RAY_TMPDIR" ]; then
    echo "[no_cuda_graph] Cleaning Ray temp directory: $RAY_TMPDIR"
    rm -rf "$RAY_TMPDIR"/* 2>/dev/null || true
fi

# 启动Ray
echo "[no_cuda_graph] Starting Ray head node..."
ray start --head --num-cpus=8 --temp-dir="$RAY_TMPDIR" --include-dashboard=true --dashboard-host="$RAY_DASHBOARD_HOST" ${RAY_NODE_IP_ADDRESS:+--node-ip-address="$RAY_NODE_IP_ADDRESS"} | cat

sleep 3

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.filter_groups.enable=true \
    data.train_files=[\"${TRAIN_DATASETS}\"] \
    data.val_files=[\"${VAL_DATASETS}\"] \
    data.train_batch_size="${TRAIN_BS}" \
    data.gen_batch_size="${GEN_BS}" \
    data.val_batch_size=2048 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.shuffle=true \
    data.return_raw_chat=true \
    data.filter_overlong_prompts=False \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.hybrid_engine=true \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BS}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BSZ_PER_GPU \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.fsdp_config.offload_policy=true \
    actor_rollout_ref.actor.fsdp_config.timeout=20 \
    actor_rollout_ref.actor.checkpoint.save_contents="['model', 'optimizer', 'extra']" \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW \
    actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
    actor_rollout_ref.rollout.max_model_len=${actor_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BSZ_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$GEN_TP \
    actor_rollout_ref.rollout.name=sglang_async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.disable_cuda_graph=true \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.n=$N \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BSZ_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    trainer.logger=['wandb','tensorboard'] \
    trainer.val_only=false \
    trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=$NNODES \
    trainer.save_freq=5 \
    trainer.test_freq=10 \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs="${EPOCHS}" \
    trainer.total_training_steps="${STEPS}" \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="${SAVE_MODEL_FOLDER}/${EXPERIMENT_NAME}" \
    trainer.ray_wait_register_center_timeout=1800 \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_turns=8 \
    +actor_rollout_ref.rollout.multi_turn.format=qwen \
    actor_rollout_ref.rollout.multi_turn.use_xml_tool_parser=true \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$CODE_CONFIG" \
    reward_model.reward_manager="afm" \
    2>&1 | tee logs/$EXPERIMENT_NAME.log

echo "[no_cuda_graph] ====== 训练完成或退出 ======"
