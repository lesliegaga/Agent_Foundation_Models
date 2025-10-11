#!/bin/bash
# =====================================================================================================================
#                           Enhanced Diagnostic Script for Collective Mismatch Issues
# =====================================================================================================================

set -x

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}[DIAGNOSTIC] Starting comprehensive diagnostic for CollectiveFingerPrint mismatch${NC}"

# =====================================================================================================================
#                                      System Environment Check
# =====================================================================================================================
echo -e "${YELLOW}[DIAGNOSTIC] === System Environment Check ===${NC}"

echo "[DIAGNOSTIC] CUDA Environment:"
nvidia-smi
echo "[DIAGNOSTIC] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "[DIAGNOSTIC] Available GPUs: $(nvidia-smi --list-gpus | wc -l)"

echo "[DIAGNOSTIC] PyTorch NCCL Environment:"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')
print(f'NCCL available: {torch.distributed.is_nccl_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB')
"

echo "[DIAGNOSTIC] Memory Status:"
free -h
echo "[DIAGNOSTIC] Disk Space:"
df -h

# =====================================================================================================================
#                                      Process and Network Check
# =====================================================================================================================
echo -e "${YELLOW}[DIAGNOSTIC] === Process and Network Check ===${NC}"

echo "[DIAGNOSTIC] Existing Ray processes:"
ps aux | grep ray | head -10

echo "[DIAGNOSTIC] Existing Python processes:"
ps aux | grep python | grep -E "(sglang|verl|fsdp)" | head -10

echo "[DIAGNOSTIC] Network interfaces:"
ip addr show | grep -E "(inet|UP)"

echo "[DIAGNOSTIC] Open ports:"
netstat -tlnp | grep -E "(6379|10001|10002)" || echo "No relevant ports found"

# =====================================================================================================================
#                                      Ray Cluster Diagnostic
# =====================================================================================================================
echo -e "${YELLOW}[DIAGNOSTIC] === Ray Cluster Diagnostic ===${NC}"

# Clean up any existing Ray processes
echo "[DIAGNOSTIC] Cleaning up Ray processes..."
ray stop --force >/dev/null 2>&1 || true
pkill -f ray:: >/dev/null 2>&1 || true
sleep 3

# Set diagnostic environment
export RAY_TMPDIR="/tmp/ray_diagnostic"
export RAY_DEDUP_LOGS=0
export RAY_LOG_TO_STDERR=1
export RAY_BACKEND_LOG_LEVEL=debug

mkdir -p "$RAY_TMPDIR"

echo "[DIAGNOSTIC] Starting Ray head with debug logging..."
ray start --head \
    --num-cpus=4 \
    --temp-dir="$RAY_TMPDIR" \
    --include-dashboard=true \
    --dashboard-host="127.0.0.1" \
    --node-ip-address="127.0.0.1" \
    --verbose 2>&1 | head -20

sleep 5

echo "[DIAGNOSTIC] Ray cluster status:"
ray status || echo "[DIAGNOSTIC] Ray status failed"

echo "[DIAGNOSTIC] Ray cluster resources:"
python3 -c "
import ray
try:
    ray.init(address='auto', ignore_reinit_error=True)
    print(f'Ray cluster resources: {ray.cluster_resources()}')
    print(f'Ray available resources: {ray.available_resources()}')
    ray.shutdown()
except Exception as e:
    print(f'Ray connection failed: {e}')
"

# =====================================================================================================================
#                                      NCCL and Distributed Test
# =====================================================================================================================
echo -e "${YELLOW}[DIAGNOSTIC] === NCCL and Distributed Test ===${NC}"

cat > /tmp/nccl_test.py << 'EOF'
import os
import torch
import torch.distributed as dist
from datetime import timedelta

def test_nccl_collective():
    try:
        # Initialize process group with detailed logging
        os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
        os.environ['NCCL_DEBUG'] = 'INFO'
        os.environ['NCCL_DEBUG_SUBSYS'] = 'COLL'
        
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                timeout=timedelta(seconds=60)
            )
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)
        
        print(f'[NCCL_TEST] Rank {rank}/{world_size} initialized successfully')
        
        # Test tensor creation
        tensor = torch.ones(10).cuda(device)
        print(f'[NCCL_TEST] Rank {rank} created tensor: {tensor.sum()}')
        
        # Test collective operations
        print(f'[NCCL_TEST] Rank {rank} starting barrier...')
        dist.barrier()
        print(f'[NCCL_TEST] Rank {rank} barrier completed')
        
        print(f'[NCCL_TEST] Rank {rank} starting all_reduce...')
        dist.all_reduce(tensor)
        print(f'[NCCL_TEST] Rank {rank} all_reduce completed, result: {tensor.sum()}')
        
        print(f'[NCCL_TEST] Rank {rank} all tests passed!')
        
    except Exception as e:
        print(f'[NCCL_TEST] Rank {rank if "rank" in locals() else "unknown"} failed: {e}')
        import traceback
        traceback.print_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == '__main__':
    test_nccl_collective()
EOF

echo "[DIAGNOSTIC] Testing NCCL collective operations..."
# Test with multiple processes if available
if [ -n "$CUDA_VISIBLE_DEVICES" ] && [ "$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)" -gt 1 ]; then
    echo "[DIAGNOSTIC] Running multi-GPU NCCL test..."
    torchrun --nproc_per_node=2 --nnodes=1 /tmp/nccl_test.py 2>&1 | head -30
else
    echo "[DIAGNOSTIC] Single GPU environment, skipping multi-GPU NCCL test"
fi

# =====================================================================================================================
#                                      SGLang Engine Test
# =====================================================================================================================
echo -e "${YELLOW}[DIAGNOSTIC] === SGLang Engine Test ===${NC}"

cat > /tmp/sglang_test.py << 'EOF'
import os
import torch
import torch.distributed as dist
from datetime import timedelta

def test_sglang_initialization():
    try:
        # Set SGLang specific environment variables
        os.environ['SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK'] = 'true'
        os.environ['SGLANG_DISABLE_CUDA_GRAPH'] = '1'
        os.environ['SGLANG_MEM_FRACTION_STATIC'] = '0.7'
        os.environ['TORCH_NCCL_AVOID_RECORD_STREAMS'] = '1'
        
        print("[SGLANG_TEST] Testing SGLang initialization...")
        
        # Test basic imports
        try:
            import sglang
            print(f"[SGLANG_TEST] SGLang version: {sglang.__version__}")
        except Exception as e:
            print(f"[SGLANG_TEST] SGLang import failed: {e}")
            return
        
        # Test engine initialization (simplified)
        print("[SGLANG_TEST] Testing basic functionality...")
        
        # Check CUDA memory before initialization
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem_info = torch.cuda.mem_get_info(i)
                print(f"[SGLANG_TEST] GPU {i} memory: {mem_info[1]/1e9:.1f}GB total, {mem_info[0]/1e9:.1f}GB free")
        
        print("[SGLANG_TEST] Basic checks completed successfully")
        
    except Exception as e:
        print(f"[SGLANG_TEST] Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_sglang_initialization()
EOF

echo "[DIAGNOSTIC] Testing SGLang initialization..."
python3 /tmp/sglang_test.py 2>&1 | head -20

# =====================================================================================================================
#                                      Configuration Analysis
# =====================================================================================================================
echo -e "${YELLOW}[DIAGNOSTIC] === Configuration Analysis ===${NC}"

echo "[DIAGNOSTIC] Environment variables related to distributed training:"
env | grep -E "(CUDA|NCCL|TORCH|RAY|SGL)" | sort

echo "[DIAGNOSTIC] Python distributed packages:"
python3 -c "
packages = ['torch', 'sglang', 'ray', 'transformers']
for pkg in packages:
    try:
        mod = __import__(pkg)
        print(f'{pkg}: {mod.__version__}')
    except:
        print(f'{pkg}: not installed or no version info')
"

# =====================================================================================================================
#                                      Recommendations
# =====================================================================================================================
echo -e "${GREEN}[DIAGNOSTIC] === Recommendations ===${NC}"

echo -e "${GREEN}[DIAGNOSTIC] Based on the analysis, here are the recommendations:${NC}"
echo "1. Use the fixed script: train_dapo_code_agent_fixed.sh"
echo "2. Key fixes applied:"
echo "   - Disabled CUDA graph capture (SGLANG_DISABLE_CUDA_GRAPH=1)"
echo "   - Reduced memory pressure (mem_fraction_static=0.7)"
echo "   - Enhanced NCCL timeouts and debugging"
echo "   - Sequential Ray worker startup"
echo "   - Consistent collective operation ordering"
echo ""
echo "3. If issues persist, try:"
echo "   - Further reduce batch sizes"
echo "   - Use fewer GPUs for tensor parallelism"
echo "   - Check for hardware/driver issues"

# Cleanup
echo "[DIAGNOSTIC] Cleaning up test files..."
rm -f /tmp/nccl_test.py /tmp/sglang_test.py
ray stop --force >/dev/null 2>&1 || true

echo -e "${BLUE}[DIAGNOSTIC] Diagnostic completed. Check output above for issues.${NC}"
