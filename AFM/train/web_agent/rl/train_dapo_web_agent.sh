set -x

ulimit -n 65535
# =====================================================================================================================
#                                      Param
# =====================================================================================================================
ACTOR_LR=1e-6
TRAIN_BS=64
PPO_MINI_BS=64
GEN_BS=128
EPOCHS=2
STEPS=60
N=8
PPO_MICRO_BSZ_PER_GPU=2
LOG_PROB_MICRO_BSZ_PER_GPU=8
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
# context window
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 30))
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
# performance related param
SP_SIZE=8
GEN_TP=4 # 和CUDA_VISIBLE_DEVICES的GPU数量一致
use_dynamic_bsz=True
offload=True
# =====================================================================================================================
#                                      Env
# =====================================================================================================================
CURRENT_DIR=$(pwd)
# NOTE: We recommend to use wandb as log backend. Export your own wandb project and key to use it.
export WANDB_MODE="offline"
export NNODES=1 # 单机为1，多机为实际GPU数量
export PROJECT_NAME="agent_foundation_models"
EXPERIMENT_DIR=$(dirname "$(readlink -f "$0")")
# export EXPERIMENT_NAME="DAPO-QWEN32B-WebAgent"
export EXPERIMENT_NAME="DAPO-QWEN7B-WebAgent"
# export BASE_MODEL="${CURRENT_DIR}/AFM/models/web_agent/AFM-WebAgent-32B-sft"   # your train model path
export BASE_MODEL="/mnt/tongyan.zjy/model_output/AFM/web_agent_sft/exp_1_lr1.4e-5_wr0.1_bs1_ga16"   # your train model path
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export RAY_NAMESPACE="${EXPERIMENT_NAME}"
unset RAY_MEMORY
unset RAY_OBJECT_STORE_MEMORY
export RAY_DISABLE_DASHBOARD=0
export RAY_TMPDIR="/tmp/ray"
mkdir -p "$RAY_TMPDIR" && chmod 777 "$RAY_TMPDIR"
export RAY_DEDUP_LOGS=0
export RAY_BACKEND_LOG_LEVEL=debug
# 绑定到真实主机 IP，dashboard 监听 0.0.0.0，避免 agent 绑定不可达地址
# 单机绑定回环地址，确保 raylet 与 agents 在相同地址通信，避免本机外网地址导致的拒连
export RAY_NODE_IP_ADDRESS="127.0.0.1"
export RAY_DASHBOARD_HOST="127.0.0.1"
unset RAY_AGENT_PORT
unset RAY_RUNTIME_ENV_AGENT_STARTUP_TIMEOUT_MS
TRAIN_DATASETS="${CURRENT_DIR}/amap_search_rag_AFM-WebAgent-RL-Dataset_20250917181100/combined_data_0724.parquet"   # your train dataset
VAL_DATASETS="" # "your val datasets"
# =====================================================================================================================
#                                      Tool
# =====================================================================================================================
# code tool
CODE_CONFIG="${CURRENT_DIR}/verl/verl/tools/config/code_tool_config/code_executor.yaml"
# search tools
SEARCH_CONFIG="${CURRENT_DIR}/verl/verl/tools/config/search_tool_config/training_servers_config.yaml"
# afm tools
AFM_CONFIG="${CURRENT_DIR}/verl/verl/tools/config/afm_tool_config/afm_tool_config.yaml" 

# =====================================================================================================================
#                                      Train
# =====================================================================================================================
cd verl
ray stop --force >/dev/null 2>&1 || true
# 预启动本地 Ray head，以提升 runtime env agent 稳定性
ray start --head --temp-dir="$RAY_TMPDIR" --include-dashboard=true --dashboard-host="$RAY_DASHBOARD_HOST" ${RAY_NODE_IP_ADDRESS:+--node-ip-address="$RAY_NODE_IP_ADDRESS"} | cat
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.filter_groups.enable=true \
    data.train_files=[\"${TRAIN_DATASETS}\"] \
    data.val_files=[\"${VAL_DATASETS}\"] \
    data.train_batch_size="${TRAIN_BS}" \
    data.gen_batch_size="${GEN_BS}" \
    data.val_batch_size=4096 \
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
    actor_rollout_ref.actor.optim.lr_warmup_steps=3 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BS}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BSZ_PER_GPU \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.checkpoint.save_contents="['model', 'optimizer', 'extra']" \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW \
    actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.rollout.max_model_len=${actor_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BSZ_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$GEN_TP \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.n=$N \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BSZ_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    trainer.logger=['wandb','tensorboard'] \
    trainer.val_only=false \
    trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=$NNODES \
    trainer.save_freq=5 \
    trainer.test_freq=1000 \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs="${EPOCHS}" \
    trainer.total_training_steps="${STEPS}" \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="${EXPERIMENT_DIR}/${EXPERIMENT_NAME}" \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_turns=25 \
    +actor_rollout_ref.rollout.multi_turn.format=qwen \
    actor_rollout_ref.rollout.multi_turn.use_xml_tool_parser=true \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$SEARCH_CONFIG" \
    reward_model.reward_manager="batch" \
    custom_reward_function.train_path="${CURRENT_DIR}/verl/verl/utils/reward_score/grm_simple.py" \
    custom_reward_function.train_name="compute_score_grm_batch" \
    custom_reward_function.val_path="${CURRENT_DIR}/verl/verl/utils/reward_score/grm_simple.py" \
    custom_reward_function.val_name="compute_score_grm_batch" \
    hydra.run.dir=. \
    hydra.output_subdir=null \
    hydra/job_logging=disabled \
    hydra/hydra_logging=disabled \
    2>&1 | tee $EXPERIMENT_DIR/$EXPERIMENT_NAME.log

# n_gpus_per_node 和 CUDA_VISIBLE_DEVICES的GPU数量一致