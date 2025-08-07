#!/bin/bash
export TORCHDYNAMO_VERBOSE=1

### RL ###
model_path="your model path"
base_modelname=AFM-WebAgent-32B-RL
base_port=10000


INSTANCES=1
GPUS_PER_INSTANCE=4
max_model_len=32768
LOG_DIR="logs"
WAIT_TIMEOUT=300

mkdir -p "$LOG_DIR"

net0_ip=$(ifconfig net0 | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -n 1)

if [ -z "$net0_ip" ]; then
    net0_ip=$(hostname -I | awk '{print $1}')
fi

ip_sanitized=$(echo "$net0_ip" | tr '.' '_')

log_prefix="${base_modelname}_${ip_sanitized}"

echo "Start deploy ${INSTANCES} instances with model ${base_modelname}"

for ((i=0; i<INSTANCES; i++)); do
    start_gpu=$((i * GPUS_PER_INSTANCE))
    end_gpu=$((start_gpu + GPUS_PER_INSTANCE - 1))
    gpu_list=$(seq $start_gpu $end_gpu | tr '\n' ',')
    gpu_list=${gpu_list%,}

    port=$((base_port + i))
    
    instance_name="${base_modelname}_inst${i}"
    log_file="${LOG_DIR}/${log_prefix}_inst${i}.log"
    
    echo "Start server ${instance_name}: Port ${port}, use GPU ${gpu_list}"
    
    # --rope_scaling '{\"rope_type\":\"yarn\",\"factor\":4.0,\"original_max_position_embeddings\":32768}' \
    nohup bash -c "
        export CUDA_VISIBLE_DEVICES=${gpu_list}
        vllm serve ${model_path} \
            --served_model_name ${base_modelname} \
            --max_model_len ${max_model_len} \
            --max_seq_len ${max_model_len} \
            --tensor_parallel_size ${GPUS_PER_INSTANCE} \
            --gpu_memory_utilization 0.7 \
            --trust_remote_code \
            --uvicorn_log_level debug \
            --host 0.0.0.0 \
            --port ${port}
    " > "$log_file" 2>&1 &
    
    echo "Wait for instance ${instance_name} start..."
    start_time=$(date +%s)
    server_started=0
    
    while [ $(( $(date +%s) - start_time )) -lt $WAIT_TIMEOUT ]; do
        if grep -q "INFO:     Started server process" "$log_file"; then
            server_started=1
            break
        fi
        sleep 5
    done
    
    if [ $server_started -eq 1 ]; then
        echo "Instance ${instance_name} start successfully."
    else
        echo "Warn: instance ${instance_name} fail to start in ${WAIT_TIMEOUT}sec."
    fi
done

echo -e "\n===== finish deployment ====="
echo "IP(net0)：$net0_ip"
echo "URL Endpoint:"
for ((i=0; i<INSTANCES; i++)); do
    port=$((base_port + i))
    echo "  - http://$net0_ip:$port/v1"
done
echo "===================="
echo "Log path: $LOG_DIR"
echo "Log name: ${base_modelname}_${ip_sanitized}_inst<id>.log"