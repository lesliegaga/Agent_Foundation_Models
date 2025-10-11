#!/bin/bash

# ============================================================================
#                    CUDA图问题诊断脚本
# ============================================================================

set -x

echo "========================================="
echo "开始诊断CUDA图相关问题"
echo "========================================="

# 检查当前环境变量
echo "1. 检查相关环境变量:"
echo "SGLANG_DISABLE_CUDA_GRAPH: $SGLANG_DISABLE_CUDA_GRAPH"
echo "DISABLE_CUDA_GRAPH: $DISABLE_CUDA_GRAPH"
echo "CUDA_GRAPHS_ENABLED: $CUDA_GRAPHS_ENABLED"
echo "SGLANG_FORCE_EAGER: $SGLANG_FORCE_EAGER"
echo "VERL_DISABLE_CUDA_GRAPH: $VERL_DISABLE_CUDA_GRAPH"

# 检查SGLang版本和配置
echo "2. 检查SGLang安装:"
python3 -c "
try:
    import sglang
    print(f'SGLang version: {sglang.__version__}')
    
    # 检查SGLang的配置选项
    try:
        from sglang.srt.server_args import ServerArgs
        args = ServerArgs()
        print('ServerArgs available attributes:')
        cuda_graph_attrs = [attr for attr in dir(args) if 'cuda' in attr.lower() or 'graph' in attr.lower()]
        for attr in cuda_graph_attrs:
            print(f'  {attr}: {getattr(args, attr, \"N/A\")}')
    except Exception as e:
        print(f'Failed to inspect ServerArgs: {e}')
        
except ImportError as e:
    print(f'SGLang not installed: {e}')
"

# 检查PyTorch分布式和NCCL
echo "3. 检查PyTorch分布式支持:"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'NCCL available: {torch.distributed.is_nccl_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
"

# 测试简单的NCCL集体通信
echo "4. 测试基础NCCL通信:"
cat > /tmp/test_nccl_simple.py << 'EOF'
import os
import torch
import torch.distributed as dist

def test_nccl():
    try:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['RANK'] = '0'
        
        dist.init_process_group(backend='nccl', timeout=torch.distributed.timedelta(seconds=30))
        
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            tensor = torch.ones(10).cuda(device)
            print(f"NCCL init successful, tensor: {tensor.sum()}")
            
        dist.destroy_process_group()
        print("NCCL test passed")
        
    except Exception as e:
        print(f"NCCL test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_nccl()
EOF

python3 /tmp/test_nccl_simple.py

# 检查verl配置
echo "5. 检查verl配置相关代码:"
python3 -c "
try:
    from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout
    print('SGLangRollout class found')
    
    # 检查SGLangRollout的初始化参数
    import inspect
    sig = inspect.signature(SGLangRollout.__init__)
    print('SGLangRollout.__init__ parameters:')
    for param_name, param in sig.parameters.items():
        print(f'  {param_name}: {param}')
        
except ImportError as e:
    print(f'verl SGLangRollout not found: {e}')
except Exception as e:
    print(f'Error inspecting SGLangRollout: {e}')
"

# 分析verl配置系统
echo "6. 分析verl配置参数传递:"
find /Users/zhujiayan/Documents/Github/Agent_Foundation_Models/verl -name "*.py" -exec grep -l "disable_cuda_graph\|enable_cuda_graph" {} \; | head -5 | while read file; do
    echo "Found in: $file"
    grep -n "disable_cuda_graph\|enable_cuda_graph" "$file" | head -3
    echo "---"
done

echo "7. 生成临时修复脚本建议:"
cat > /tmp/suggested_fix.sh << 'EOF'
#!/bin/bash
# 建议的修复方案

echo "方案1: 强制使用vLLM而不是SGLang"
echo "在train_dapo_code_agent.sh中修改:"
echo "actor_rollout_ref.rollout.name=vllm_rollout"
echo ""

echo "方案2: 降低tensor_parallel_size"
echo "GEN_TP=2  # 从4改为2"
echo ""

echo "方案3: 使用单GPU测试"
echo "export CUDA_VISIBLE_DEVICES=\"0\""
echo "GEN_TP=1"
echo ""

echo "方案4: 完全跳过rollout初始化测试"
echo "在配置中添加:"
echo "trainer.val_only=true"
echo "trainer.val_before_train=false"
EOF

echo "========================================="
echo "诊断完成，建议查看 /tmp/suggested_fix.sh"
echo "========================================="

# 清理临时文件
rm -f /tmp/test_nccl_simple.py

echo "诊断脚本执行完成"
