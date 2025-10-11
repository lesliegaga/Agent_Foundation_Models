set -x

ulimit -n 65535
# =====================================================================================================================
#                                      Param
# =====================================================================================================================
ACTOR_LR=1e-6
TRAIN_BS=128  # 减少从256到128
PPO_MINI_BS=32
GEN_BS=128    # 减少从256到128
EPOCHS=100
STEPS=2000
N=8
PPO_MICRO_BSZ_PER_GPU=2
LOG_PROB_MICRO_BSZ_PER_GPU=16
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
# CUDA graph optimization params
CUDA_GRAPH_MAX_BS=16
MEM_FRACTION_STATIC=0.7
# =====================================================================================================================
#                                      Env
# =====================================================================================================================
# NOTE: We recommend to use wandb as log backend. Export your own wandb project and key to use it. Remember to turn on wandb_mode if you sync online.
export WANDB_MODE="offline"
CURRENT_DIR=$(pwd)
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NNODES=1 # "your GPU group number"
export PROJECT_NAME="agent_foundation_models"
SAVE_MODEL_FOLDER="${CURRENT_DIR}/experiments"  # your save model folder
export EXPERIMENT_NAME="DAPO-QWEN7B-CodeAgent"
export BASE_MODEL="/mnt/tongyan.zjy/model_output/AFM/AFM-CodeAgent-7B-sft/exp_lr3e-5_bs1_ga4_ep2.0_cl32768_bf16"   # your train model path
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues
# export RAY_NAMESPACE="${EXPERIMENT_NAME}"
# unset RAY_MEMORY
# unset RAY_OBJECT_STORE_MEMORY
# export RAY_DISABLE_DASHBOARD=0
export RAY_TMPDIR="/mnt/tongyan.zjy/tmp/ray"
export RAY_DEDUP_LOGS=0
export RAY_NUM_PRESTART_WORKERS=0
export RAY_MAXIMUM_STARTUP_CONCURRENCY=2  # 减少并发worker数量，避免内存峰值
export RAY_WORKER_REGISTER_TIMEOUT_SECONDS=300
# Ray内存管理配置
export RAY_memory_usage_threshold=0.98  # 提高内存阈值到98%
export RAY_memory_monitor_refresh_ms=1000  # 1秒检查一次内存
export RAY_object_spilling_threshold=0.8  # 80%时开始spill对象到磁盘
# 绑定到真实主机 IP，dashboard 监听 0.0.0.0，避免 agent 绑定不可达地址
# 单机绑定回环地址，确保 raylet 与 agents 在相同地址通信，避免本机外网地址导致的拒连
export RAY_NODE_IP_ADDRESS="127.0.0.1"
export RAY_DASHBOARD_HOST="127.0.0.1"
# 启用详细的Ray日志
# export RAY_LOG_TO_STDERR=1
# export RAY_BACKEND_LOG_LEVEL=debug
# PyTorch distributed timeout and debug settings
export TORCH_DISTRIBUTED_INIT_TIMEOUT=600
export NCCL_TIMEOUT=600
export NCCL_SOCKET_TIMEOUT=600
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
# SGLang specific debugging
# export SGLANG_DEBUG=1
# export SGLANG_CUDA_GRAPH_DEBUG=1
TRAIN_DATASETS="${CURRENT_DIR}/amap_search_rag_AFM-CodeAgent-RL-Dataset_20250924165348/CodeAgentRLDataset.parquet"   # your train dataset
VAL_DATASETS="${CURRENT_DIR}/amap_search_rag_AFM-CodeAgent-RL-Dataset_20250924165348/CodeAgentRLDataset.parquet"
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
# 创建日志目录
mkdir -p logs
# 强制清理所有Ray进程和资源
echo "[train_sh] Cleaning up Ray resources..."
ray stop --force >/dev/null 2>&1 || true
pkill -f ray:: >/dev/null 2>&1 || true
pkill -f raylet >/dev/null 2>&1 || true
sleep 2

# 清理Ray临时目录
if [ -d "$RAY_TMPDIR" ]; then
    echo "[train_sh] Cleaning Ray temp directory: $RAY_TMPDIR"
    rm -rf "$RAY_TMPDIR"/* 2>/dev/null || true
fi

# 预启动本地 Ray head，以提升 runtime env agent 稳定性
echo "[train_sh] Starting Ray head node..."
ray start --head --num-cpus=16 --temp-dir="$RAY_TMPDIR" --include-dashboard=true --dashboard-host="$RAY_DASHBOARD_HOST" ${RAY_NODE_IP_ADDRESS:+--node-ip-address="$RAY_NODE_IP_ADDRESS"} | cat

# 等待Ray集群完全启动
echo "[train_sh] Waiting for Ray cluster to be ready..."
sleep 5

# 检查Ray集群状态
echo "[train_sh] Checking Ray cluster status..."
ray status || echo "[train_sh] Warning: Ray status check failed, but continuing..."

export RAY_GCS_ADDRESS="${SERVER_HOST}:6379"
export RAY_ADDRESS="$RAY_GCS_ADDRESS"
echo "[train_sh] Ray cluster configured with GCS address: $RAY_GCS_ADDRESS"

# 验证Ray连接
echo "[train_sh] Verifying Ray connection..."
python3 -c "
import ray
try:
    ray.init(address='$RAY_ADDRESS', ignore_reinit_error=True)
    print('[train_sh] Ray connection successful')
    print(f'[train_sh] Ray cluster info: {ray.cluster_resources()}')
    
    # 检查现有的named actors
    from ray.util.state import list_actors
    actors = list_actors()
    print(f'[train_sh] Current Ray actors: {len(actors)} total')
    for actor in actors[:5]:  # 只显示前5个
        print(f'[train_sh] Actor: {actor}')
    
    # 检查命名空间
    import ray.util.state as state
    namespaces = state.list_nodes()
    print(f'[train_sh] Available namespaces: {len(namespaces)}')
    
    # 检查资源可用性
    resources = ray.available_resources()
    print(f'[train_sh] Available resources: {resources}')
    
    ray.shutdown()
except Exception as e:
    print(f'[train_sh] Ray connection failed: {e}')
    import traceback
    traceback.print_exc()
" || {
    echo "[train_sh] Ray connection verification failed, stopping Ray and letting training code manage it"
    ray stop --force >/dev/null 2>&1 || true
    unset RAY_ADDRESS
    unset RAY_GCS_ADDRESS
    echo "[train_sh] Ray environment variables cleared, training will start its own Ray cluster"
}

echo "[train_sh] ====== 开始SGLang兼容性检查 ======"
# 检查SGLang和CUDA兼容性
python3 -c "
import torch
import sglang
print(f'[train_sh] PyTorch version: {torch.__version__}')
print(f'[train_sh] CUDA available: {torch.cuda.is_available()}')
print(f'[train_sh] CUDA device count: {torch.cuda.device_count()}')
print(f'[train_sh] SGLang version: {sglang.__version__}')

# 检查NCCL
try:
    import torch.distributed as dist
    print(f'[train_sh] NCCL available: {dist.is_nccl_available()}')
except Exception as e:
    print(f'[train_sh] NCCL check failed: {e}')

# 测试基本GPU操作
try:
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            x = torch.randn(10, 10).cuda()
            y = x @ x.T
            print(f'[train_sh] GPU {i} basic operations: OK')
except Exception as e:
    print(f'[train_sh] GPU operations test failed: {e}')
    import traceback
    traceback.print_exc()
"
echo "[train_sh] ====== SGLang兼容性检查完成 ======"

# 检查模型路径和权限
echo "[train_sh] Checking model path and permissions..."
if [ -d "$BASE_MODEL" ]; then
    echo "[train_sh] ✓ Base model directory exists: $BASE_MODEL"
    echo "[train_sh] Model directory size: $(du -sh "$BASE_MODEL" 2>/dev/null || echo 'Unknown')"
    ls -la "$BASE_MODEL" | head -10
else
    echo "[train_sh] ✗ ERROR: Base model directory not found: $BASE_MODEL"
    exit 1
fi

# 检查可用内存和GPU状态
echo "[train_sh] Checking system resources..."
TOTAL_MEM_GB=$(free -g | grep '^Mem:' | awk '{print $2}')
USED_MEM_GB=$(free -g | grep '^Mem:' | awk '{print $3}')
AVAILABLE_MEM_GB=$(free -g | grep '^Mem:' | awk '{print $7}')
MEM_USAGE_PERCENT=$((USED_MEM_GB * 100 / TOTAL_MEM_GB))

echo "[train_sh] Memory Status:"
echo "[train_sh]   Total: ${TOTAL_MEM_GB}GB"
echo "[train_sh]   Used: ${USED_MEM_GB}GB (${MEM_USAGE_PERCENT}%)"
echo "[train_sh]   Available: ${AVAILABLE_MEM_GB}GB"

# 内存预检查：确保至少有120GB可用内存
REQUIRED_MEM_GB=120
if [ $AVAILABLE_MEM_GB -lt $REQUIRED_MEM_GB ]; then
    echo "[train_sh] ⚠️  WARNING: Available memory (${AVAILABLE_MEM_GB}GB) < Required (${REQUIRED_MEM_GB}GB)"
    echo "[train_sh] This may cause OOM during model initialization."
fi

if [ $MEM_USAGE_PERCENT -gt 85 ]; then
    echo "[train_sh] 🚨 WARNING: Memory usage ${MEM_USAGE_PERCENT}% > 85%, high risk of OOM"
fi

echo "[train_sh] GPU status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv

# 检查训练数据集
echo "[train_sh] Checking training dataset..."
if [ -f "$TRAIN_DATASETS" ]; then
    echo "[train_sh] ✓ Training dataset exists: $TRAIN_DATASETS"
    echo "[train_sh] Dataset size: $(du -sh "$TRAIN_DATASETS" 2>/dev/null || echo 'Unknown')"
else
    echo "[train_sh] ✗ ERROR: Training dataset not found: $TRAIN_DATASETS"
    exit 1
fi

# 创建worker初始化监控脚本
echo "[train_sh] Creating worker initialization monitor..."
cat > /tmp/monitor_workers.py << 'EOF'
import time
import subprocess
import threading
import os
import signal

def monitor_workers():
    """监控worker进程的创建和状态"""
    print(f"[{time.strftime('%H:%M:%S')}] Starting worker monitor...")
    
    while True:
        try:
            # 检查Python worker进程
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=5)
            python_procs = []
            for line in result.stdout.split('\n'):
                if 'python' in line and ('verl' in line or 'fsdp' in line or 'ActorRollout' in line):
                    parts = line.split()
                    if len(parts) >= 11:
                        pid = parts[1]
                        cmd = ' '.join(parts[10:])[:100]
                        python_procs.append(f"PID:{pid} - {cmd}")
            
            print(f"[{time.strftime('%H:%M:%S')}] Worker processes: {len(python_procs)}")
            for proc in python_procs:
                print(f"  {proc}")
            
            # 检查Ray actors
            try:
                actors_result = subprocess.run(['python3', '-c', '''
import ray
try:
    ray.init(address="33.93.148.4:6379", ignore_reinit_error=True)
    from ray.util.state import list_actors
    actors = list_actors()
    print(f"Ray actors: {len(actors)}")
    for actor in actors[:3]:
        print(f"  {actor}")
    from ray.util import list_named_actors
    named = list_named_actors()
    print(f"Named actors: {named}")
    ray.shutdown()
except Exception as e:
    print(f"Ray check failed: {e}")
'''], capture_output=True, text=True, timeout=10)
                print(actors_result.stdout.strip())
                if actors_result.stderr:
                    print(f"Ray errors: {actors_result.stderr.strip()}")
            except Exception as e:
                print(f"Failed to check Ray actors: {e}")
            
            print("-" * 60)
            time.sleep(15)
            
        except Exception as e:
            print(f"Monitor error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    try:
        monitor_workers()
    except KeyboardInterrupt:
        print("Worker monitor stopped")
EOF

# 启动worker监控（后台运行）
echo "[train_sh] Starting worker monitor in background..."
python3 /tmp/monitor_workers.py > logs/worker_monitor.log 2>&1 &
MONITOR_PID=$!
echo "[train_sh] Worker monitor started with PID: $MONITOR_PID"

# 设置清理函数
cleanup_monitor() {
    if [ ! -z "$MONITOR_PID" ]; then
        echo "[train_sh] Stopping worker monitor (PID: $MONITOR_PID)..."
        kill $MONITOR_PID 2>/dev/null || true
    fi
}
trap cleanup_monitor EXIT
# # 解析当前 Ray 会话的 GCS 地址（固定 6379）
# SESSION_DIR=$(readlink -f "$RAY_TMPDIR/session_latest" 2>/dev/null || echo "")
# if [ -n "$SESSION_DIR" ] && [ -f "$SESSION_DIR/node_ip_address.json" ]; then
#     export SESSION_DIR
#     PY_IP=$(python3 - <<'PY'
# import json, os
# sd = os.environ.get('SESSION_DIR','')
# ip = ''
# try:
#     with open(os.path.join(sd, 'node_ip_address.json')) as f:
#         data = json.load(f)
#         if isinstance(data, dict):
#             ip = data.get('node_ip_address') or data.get('ip') or ''
#         else:
#             ip = str(data).strip('"')
# except Exception:
#     pass
# print(ip)
# PY
# )
#     if [ -n "$PY_IP" ]; then
#         export RAY_GCS_ADDRESS="$PY_IP:6379"
#         export RAY_ADDRESS="$RAY_GCS_ADDRESS"
#         echo "[train_sh] RAY_GCS_ADDRESS=$RAY_GCS_ADDRESS"
#     fi
# fi

echo "[train_sh] ====== 开始训练主程序 ======"
echo "[train_sh] 配置摘要:"
echo "[train_sh]   GPU并行度: $GEN_TP"
echo "[train_sh]   训练批量大小: $TRAIN_BS"
echo "[train_sh]   生成批量大小: $GEN_BS"
echo "[train_sh]   GPU内存利用率: 0.4"
echo "[train_sh]   CUDA图最大批量: $CUDA_GRAPH_MAX_BS"
echo "[train_sh]   内存分配比例: $MEM_FRACTION_STATIC"

# # 设置fallback处理
# trap 'handle_training_failure' ERR

# handle_training_failure() {
#     echo "[train_sh] ❌ 训练失败，检查是否为CUDA图问题..."
    
#     # 检查日志中是否包含CUDA图相关错误
#     if grep -q "Capture CUDA graph failed" logs/$EXPERIMENT_NAME.log 2>/dev/null; then
#         echo "[train_sh] 🔄 检测到CUDA图失败，尝试禁用CUDA图重新运行..."
        
#         # 添加禁用CUDA图的参数并重新运行
#         echo "[train_sh] 正在添加 --disable-cuda-graph 参数..."
#         export DISABLE_CUDA_GRAPH=true
        
#         # 这里可以添加重试逻辑，但为了避免无限循环，暂时只记录
#         echo "[train_sh] 请手动添加以下参数重新运行："
#         echo "    actor_rollout_ref.rollout.disable_cuda_graph=true \\"
#     fi
    
#     echo "[train_sh] 训练失败，请检查日志: logs/$EXPERIMENT_NAME.log"
#     exit 1
# }
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
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BS}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BSZ_PER_GPU \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.fsdp_config.offload_policy=true \
    actor_rollout_ref.actor.fsdp_config.timeout=10 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
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
    trainer.ray_wait_register_center_timeout=900 \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_turns=8 \
    +actor_rollout_ref.rollout.multi_turn.format=qwen \
    actor_rollout_ref.rollout.multi_turn.use_xml_tool_parser=true \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$CODE_CONFIG" \
    reward_model.reward_manager="afm" \
    2>&1 | tee logs/$EXPERIMENT_NAME.log