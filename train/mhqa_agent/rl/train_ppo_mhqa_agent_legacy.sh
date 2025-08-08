set -x

source /home/notebook/code/group/linjianbo/searchr1-sglang/.envrc


ulimit -n 65535

NNODES=2
N_GPUS_PER_NODE=8

CONFIG_PATH="/home/notebook/code/group/linjianbo/searchr1-sglang/experiments/tool_config"
MODEL_PATH=/home/notebook/code/group/leewz/Mask-SFT-for-qwen3/saves/qwen2.5_7b_base_grid_prediction_list_0722/exp_1_lr1e-5_wr0.03_bs1_ga4/checkpoint-546

TRAIN_DATA=/home/notebook/code/group/linjianbo/searchr1-sglang/data/wiki_search_v3/original_16w_column_filter.parquet
# VAL_DATA=/home/notebook/code/group/linjianbo/searchr1-sglang/data/A_qa_data/wiki_search_v3/test_seven_dataset_all_column_filter.parquet
VAL_DATA=/home/notebook/code/group/linjianbo/searchr1-sglang/data/wiki_search_v3/nq_full.parquet

TOOL_CONFIG="$CONFIG_PATH/wiki_rag_config.yaml"

EXPERIMENT_DIR=$(dirname "$(readlink -f "$0")")
EXPERIMENT_NAME="base-2qasftv4-PPO+FormatEM"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='search_multiturn_ppo' \
    algorithm.adv_estimator=gae \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.05 \
    critic.model.path=$MODEL_PATH \
    critic.model.enable_gradient_checkpointing=True \
    critic.use_dynamic_bsz=True \
    critic.ppo_max_token_len_per_gpu=100000 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA"  \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=30000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.max_model_len=15000 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang_async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.multi_turn.max_turns=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.val_before_train=False \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='search_r1_like_async_rl' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.log_val_generations=50000 \
    +trainer.val_generations_jsonl_path=$EXPERIMENT_DIR/$EXPERIMENT_NAME/val_generations.jsonl \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1005 \
    trainer.default_local_dir=$EXPERIMENT_DIR/$EXPERIMENT_NAME/checkpoints \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    custom_reward_function.path=/home/notebook/code/group/linjianbo/searchr1-sglang/verl/utils/reward_score/qa_em_with_answer_list.py \
    custom_reward_function.name="compute_score_em_batch" \
    reward_model.reward_manager="batch" \
    2>&1 | tee $EXPERIMENT_DIR/$EXPERIMENT_NAME.log