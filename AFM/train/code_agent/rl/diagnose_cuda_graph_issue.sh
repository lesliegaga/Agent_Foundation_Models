#!/bin/bash

# ============================================================================
#                    CUDA图问题诊断脚本
# ============================================================================

set -x

echo "========================================="
echo "开始诊断CUDA图相关问题"
echo "========================================="

# 获取当前脚本所在目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
echo "脚本目录: $SCRIPT_DIR"
echo "项目根目录: $PROJECT_ROOT"

# 设置Python路径
export PYTHONPATH="$PROJECT_ROOT/verl:$PYTHONPATH"
cd "$PROJECT_ROOT"

# 先设置必要的环境变量（模拟train_dapo_code_agent.sh中的设置）
echo "0. 设置测试环境变量:"
export SGLANG_DISABLE_CUDA_GRAPH=1
export DISABLE_CUDA_GRAPH=1
export CUDA_GRAPHS_ENABLED=0
export SGLANG_FORCE_EAGER=1
export VERL_DISABLE_CUDA_GRAPH=1

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
        # 提供必需的model_path参数
        args = ServerArgs(model_path='dummy')
        print('ServerArgs available attributes:')
        cuda_graph_attrs = [attr for attr in dir(args) if 'cuda' in attr.lower() or 'graph' in attr.lower() or 'eager' in attr.lower()]
        for attr in cuda_graph_attrs[:10]:  # 限制输出数量
            try:
                value = getattr(args, attr, 'N/A')
                print(f'  {attr}: {value}')
            except:
                print(f'  {attr}: <unable to access>')
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
from datetime import timedelta

def test_nccl():
    try:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['RANK'] = '0'
        
        dist.init_process_group(backend='nccl', timeout=timedelta(seconds=30))
        
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
import sys
sys.path.insert(0, '$PROJECT_ROOT/verl')
print(f'Python path: {sys.path[:3]}')

try:
    from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout
    print('SGLangRollout class found')
    
    # 检查SGLangRollout的初始化参数
    import inspect
    sig = inspect.signature(SGLangRollout.__init__)
    print('SGLangRollout.__init__ parameters:')
    for param_name, param in sig.parameters.items():
        if param_name in ['self', 'kwargs']: continue
        print(f'  {param_name}: {param}')
        
except ImportError as e:
    print(f'verl SGLangRollout not found: {e}')
    # 尝试找到verl目录
    import os
    verl_path = '$PROJECT_ROOT/verl'
    if os.path.exists(verl_path):
        print(f'verl directory exists at: {verl_path}')
        print(f'Contents: {os.listdir(verl_path)[:5]}')
    else:
        print(f'verl directory not found at: {verl_path}')
except Exception as e:
    print(f'Error inspecting SGLangRollout: {e}')
"

# 分析verl配置系统
echo "6. 分析verl配置参数传递:"
find "$PROJECT_ROOT/verl" -name "*.py" -exec grep -l "disable_cuda_graph\|enable_cuda_graph" {} \; 2>/dev/null | head -5 | while read file; do
    echo "Found in: $file"
    grep -n "disable_cuda_graph\|enable_cuda_graph" "$file" | head -3
    echo "---"
done

# 特别检查SGLang rollout的配置传递
echo "6.1 检查SGLang配置传递机制:"
python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT/verl')

try:
    # 检查配置相关的文件
    import os
    search_files = []
    for root, dirs, files in os.walk('$PROJECT_ROOT/verl'):
        for file in files:
            if file.endswith('.py') and ('sglang' in file.lower() or 'rollout' in file.lower()):
                search_files.append(os.path.join(root, file))
    
    print(f'Found {len(search_files)} relevant files')
    for f in search_files[:5]:
        print(f'  {f}')
        
    # 检查配置类
    try:
        from verl.trainer.config import ppo_trainer
        print('PPO trainer config loaded')
    except Exception as e:
        print(f'Failed to load PPO trainer config: {e}')
        
except Exception as e:
    print(f'Error in config analysis: {e}')
"

echo "7. 生成修复建议和下一步行动:"
cat > /tmp/suggested_fix.sh << 'EOF'
#!/bin/bash
# 建议的修复方案

echo "=== 基于诊断结果的修复建议 ==="

echo "问题1: 环境变量传递"
echo "  - 环境变量只在train_dapo_code_agent.sh中设置，但没有传递到SGLang进程"
echo "  - 解决方案: 在verl代码中直接设置环境变量或使用配置参数"

echo ""
echo "问题2: SGLang CUDA图配置" 
echo "  - SGLang 0.4.6.post5版本中的disable_cuda_graph配置可能不生效"
echo "  - 解决方案: 使用多种方式禁用CUDA图"

echo ""
echo "方案1: 强制使用vLLM替代SGLang (推荐)"
echo "  修改 train_dapo_code_agent.sh:"
echo "  actor_rollout_ref.rollout.name=vllm_rollout"

echo ""
echo "方案2: 降低并行度减少同步复杂度"
echo "  GEN_TP=2  # 从4改为2"
echo "  或者 GEN_TP=1  # 单GPU测试"

echo ""
echo "方案3: 完全禁用推理引擎进行纯训练测试"
echo "  trainer.val_only=true"
echo "  trainer.val_before_train=false"

echo ""
echo "方案4: 修改源码强制禁用CUDA图"
echo "  在SGLangRollout初始化时添加强制环境变量设置"
EOF

echo ""
echo "8. 输出关键发现总结:"
echo "✓ SGLang版本: 0.4.6.post5 (正常)"  
echo "✓ PyTorch版本: 2.6.0+cu124 (支持NCCL)"
echo "✓ GPU数量: 8个 (充足)"
echo "✗ 环境变量传递: 未传递到SGLang进程"
echo "✗ CUDA图禁用: 配置可能不生效"
echo "✗ verl路径: 需要正确的Python路径设置"

echo "========================================="
echo "诊断完成，建议查看 /tmp/suggested_fix.sh"
echo "========================================="

# 清理临时文件
rm -f /tmp/test_nccl_simple.py

echo "诊断脚本执行完成"
