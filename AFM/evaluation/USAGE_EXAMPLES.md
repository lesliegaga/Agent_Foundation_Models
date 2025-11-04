# 综合测评脚本使用示例

## 目录
1. [快速开始](#快速开始)
2. [完整评估](#完整评估)
3. [单独数据集评估](#单独数据集评估)
4. [自定义配置](#自定义配置)
5. [结果分析](#结果分析)

---

## 快速开始

### 1分钟快速验证

```bash
cd /path/to/Agent_Foundation_Models

# 快速测试（每个数据集 50 个样本）
bash AFM/evaluation/quick_test.sh
```

**输出示例**：
```
===================================================================================================
快速测试模式 - 每个数据集仅评估 50 个样本
===================================================================================================
模型: /mnt/tongyan.zjy/openlm/model/Qwen/Qwen3-4B-Thinking-2507
输出: ./eval_results/quick_test_20251104_160000

Loading model from /mnt/tongyan.zjy/openlm/model/Qwen/Qwen3-4B-Thinking-2507...
Model loaded successfully on cuda

================================================================================
Evaluating Story Agent Thinking...
================================================================================
Story Agent Thinking Evaluation: 100%|██████████| 50/50 [02:15<00:00,  2.70s/it]

Story Agent Thinking Results:
  Full Match Accuracy: 45.00% (23/50)
  Answer Match Accuracy: 68.00% (34/50)
  Thinking Usage: 48/50

================================================================================
Evaluating NQ...
================================================================================
NQ Evaluation: 100%|██████████| 50/50 [01:30<00:00,  1.80s/it]

NQ EM Score: 66.00% (33/50)

===================================================================================================
快速测试完成！
===================================================================================================
```

---

## 完整评估

### 评估所有数据集（推荐）

```bash
cd /path/to/Agent_Foundation_Models

# 方式 1: 使用 Shell 脚本（推荐）
bash AFM/evaluation/run_comprehensive_eval.sh

# 方式 2: 直接运行 Python
python3 AFM/evaluation/comprehensive_eval.py \
    --model_path /mnt/tongyan.zjy/openlm/model/Qwen/Qwen3-4B-Thinking-2507 \
    --output_dir ./eval_results \
    --datasets mmlu ceval cmmlu nq story \
    --n_shot 5 \
    --batch_size 4
```

### 预期时间（4卡 A100）

| 数据集 | 样本数 | 预计时间 |
|--------|--------|----------|
| MMLU   | ~14,000 | 4 小时   |
| C-Eval | ~13,948 | 3 小时   |
| CMMLU  | ~11,500 | 3.5 小时 |
| NQ     | ~3,610  | 2 小时   |
| Story  | ~200    | 1 小时   |
| **总计** | **~43,258** | **13-15 小时** |

### 评估输出

```bash
# 查看结果目录
ls -lh eval_results/20251104_160000/

# 输出文件：
# comprehensive_eval_20251104_160512.json  - 完整JSON结果
# summary_20251104_160512.txt             - 摘要报告
```

---

## 单独数据集评估

### 示例 1：仅评估通用能力（MMLU + C-Eval + CMMLU）

```bash
python3 AFM/evaluation/comprehensive_eval.py \
    --model_path /mnt/tongyan.zjy/openlm/model/Qwen/Qwen3-4B-Thinking-2507 \
    --output_dir ./eval_results/general_capabilities \
    --datasets mmlu ceval cmmlu \
    --n_shot 5 \
    --batch_size 8
```

**预期时间**：~10 小时

### 示例 2：仅评估 QA 能力（NQ + Story）

```bash
python3 AFM/evaluation/comprehensive_eval.py \
    --model_path /mnt/tongyan.zjy/openlm/model/Qwen/Qwen3-4B-Thinking-2507 \
    --output_dir ./eval_results/qa_capabilities \
    --datasets nq story \
    --n_shot 0 \
    --batch_size 4 \
    --max_length 4096
```

**预期时间**：~3 小时

### 示例 3：仅评估 Story Agent Thinking

```bash
python3 AFM/evaluation/comprehensive_eval.py \
    --model_path /mnt/tongyan.zjy/openlm/model/Qwen/Qwen3-4B-Thinking-2507 \
    --output_dir ./eval_results/story_only \
    --datasets story \
    --story_file /mnt/tongyan.zjy/data/story_room/sft/training_samples_test.jsonl \
    --batch_size 2 \
    --max_length 8192
```

**预期时间**：~1 小时

---

## 自定义配置

### 调整 Few-shot 示例数

```bash
# Zero-shot (不使用示例)
--n_shot 0

# 3-shot (快速评估)
--n_shot 3

# 5-shot (标准评估，推荐)
--n_shot 5

# 10-shot (最佳性能，但较慢)
--n_shot 10
```

### 调整批处理大小（根据 GPU 显存）

```bash
# A100 80GB
--batch_size 16

# A100 40GB
--batch_size 8

# V100 32GB
--batch_size 4

# RTX 3090 24GB
--batch_size 2
```

### 调整序列长度

```bash
# 选择题（MMLU/C-Eval/CMMLU）
--max_length 2048

# 短问答（NQ）
--max_length 4096

# 长思维链（Story Agent）
--max_length 8192
```

### 限制样本数量（快速测试）

```bash
# 每个数据集 100 个样本
--max_samples 100

# 每个数据集 500 个样本
--max_samples 500

# 全量评估
--max_samples None  # 或不设置此参数
```

---

## 结果分析

### 查看 JSON 结果

```bash
# 格式化输出
python3 -m json.tool eval_results/*/comprehensive_eval_*.json | less

# 提取特定指标
cat eval_results/*/comprehensive_eval_*.json | jq '.results.mmlu.overall_accuracy'
```

### 查看摘要报告

```bash
# 直接查看
cat eval_results/*/summary_*.txt

# 示例输出：
================================================================================
COMPREHENSIVE EVALUATION SUMMARY
================================================================================

Model: /mnt/tongyan.zjy/openlm/model/Qwen/Qwen3-4B-Thinking-2507
Timestamp: 2025-11-04T16:05:12

--------------------------------------------------------------------------------
MMLU
--------------------------------------------------------------------------------
Overall Accuracy: 67.20%

Category Results:
  STEM: 65.00%
  Humanities: 71.00%
  Social Sciences: 69.50%
  Other: 68.20%

--------------------------------------------------------------------------------
C-EVAL
--------------------------------------------------------------------------------
Overall Accuracy: 72.50%

Category Results:
  STEM: 70.20%
  Social Science: 75.30%
  Humanities: 73.80%
  Other: 71.50%

...
```

### Python 脚本分析结果

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# 加载结果
with open("eval_results/.../comprehensive_eval_*.json") as f:
    results = json.load(f)

# 提取各数据集准确率
scores = {}
if "mmlu" in results["results"]:
    scores["MMLU"] = results["results"]["mmlu"]["overall_accuracy"]
if "ceval" in results["results"]:
    scores["C-Eval"] = results["results"]["ceval"]["overall_accuracy"]
if "cmmlu" in results["results"]:
    scores["CMMLU"] = results["results"]["cmmlu"]["overall_accuracy"]
if "nq" in results["results"]:
    scores["NQ"] = results["results"]["nq"]["em_score"]
if "story" in results["results"]:
    scores["Story"] = results["results"]["story"]["answer_accuracy"]

# 创建表格
df = pd.DataFrame([scores])
print(df.to_markdown())

# 可视化
plt.figure(figsize=(10, 6))
plt.bar(scores.keys(), [v*100 for v in scores.values()])
plt.ylabel("Accuracy (%)")
plt.title("Model Performance Across Datasets")
plt.ylim(0, 100)
for i, (k, v) in enumerate(scores.items()):
    plt.text(i, v*100 + 2, f"{v*100:.1f}%", ha='center')
plt.savefig("eval_results/performance_chart.png")
print("Chart saved to: eval_results/performance_chart.png")
```

### 对比多个模型

```python
import json
import pandas as pd

# 加载多个模型的结果
models = {
    "Qwen3-4B-Thinking": "eval_results/model1/comprehensive_eval_*.json",
    "Qwen2.5-7B": "eval_results/model2/comprehensive_eval_*.json",
    "Llama-3-8B": "eval_results/model3/comprehensive_eval_*.json"
}

data = []
for model_name, result_file in models.items():
    with open(result_file) as f:
        res = json.load(f)
    
    row = {"Model": model_name}
    if "mmlu" in res["results"]:
        row["MMLU"] = f"{res['results']['mmlu']['overall_accuracy']:.2%}"
    if "ceval" in res["results"]:
        row["C-Eval"] = f"{res['results']['ceval']['overall_accuracy']:.2%}"
    if "nq" in res["results"]:
        row["NQ"] = f"{res['results']['nq']['em_score']:.2%}"
    if "story" in res["results"]:
        row["Story"] = f"{res['results']['story']['answer_accuracy']:.2%}"
    
    data.append(row)

df = pd.DataFrame(data)
print(df.to_markdown(index=False))

# 输出示例：
# | Model                | MMLU   | C-Eval | NQ     | Story  |
# |----------------------|--------|--------|--------|--------|
# | Qwen3-4B-Thinking    | 67.20% | 72.50% | 68.50% | 67.80% |
# | Qwen2.5-7B           | 70.30% | 75.20% | 71.20% | 64.50% |
# | Llama-3-8B           | 68.50% | 69.80% | 66.30% | 62.10% |
```

---

## 高级用法

### 1. 批量评估多个 Checkpoint

```bash
#!/bin/bash

MODEL_DIR="/mnt/tongyan.zjy/model_output/AFM/AFM-StoryAgent-7B-sft"

# 遍历所有 checkpoint
for checkpoint in ${MODEL_DIR}/checkpoint-*; do
    if [ -d "$checkpoint" ]; then
        echo "Evaluating $checkpoint"
        
        checkpoint_name=$(basename $checkpoint)
        
        python3 AFM/evaluation/comprehensive_eval.py \
            --model_path "$checkpoint" \
            --output_dir "./eval_results/${checkpoint_name}" \
            --datasets story nq \
            --n_shot 5 \
            --batch_size 4
    fi
done

echo "All checkpoints evaluated!"
```

### 2. 并行评估不同数据集

```bash
#!/bin/bash

MODEL_PATH="/mnt/tongyan.zjy/openlm/model/Qwen/Qwen3-4B-Thinking-2507"

# 后台运行多个评估任务
python3 AFM/evaluation/comprehensive_eval.py \
    --model_path "$MODEL_PATH" \
    --output_dir ./eval_results/mmlu \
    --datasets mmlu \
    --n_shot 5 &

python3 AFM/evaluation/comprehensive_eval.py \
    --model_path "$MODEL_PATH" \
    --output_dir ./eval_results/ceval \
    --datasets ceval \
    --n_shot 5 &

python3 AFM/evaluation/comprehensive_eval.py \
    --model_path "$MODEL_PATH" \
    --output_dir ./eval_results/nq_story \
    --datasets nq story \
    --n_shot 5 &

# 等待所有任务完成
wait

echo "All evaluations completed!"
```

### 3. 定时评估（监控训练进度）

```bash
#!/bin/bash

MODEL_DIR="/mnt/tongyan.zjy/model_output/AFM/training"
CHECK_INTERVAL=3600  # 每小时检查一次

while true; do
    # 获取最新的 checkpoint
    LATEST_CHECKPOINT=$(ls -td ${MODEL_DIR}/checkpoint-* 2>/dev/null | head -1)
    
    if [ -n "$LATEST_CHECKPOINT" ]; then
        CHECKPOINT_NAME=$(basename $LATEST_CHECKPOINT)
        RESULT_FILE="./eval_results/${CHECKPOINT_NAME}/comprehensive_eval_*.json"
        
        # 如果还没评估过，则评估
        if [ ! -f "$RESULT_FILE" ]; then
            echo "New checkpoint detected: $CHECKPOINT_NAME"
            
            python3 AFM/evaluation/comprehensive_eval.py \
                --model_path "$LATEST_CHECKPOINT" \
                --output_dir "./eval_results/${CHECKPOINT_NAME}" \
                --datasets story nq \
                --max_samples 500
            
            echo "Evaluation completed for $CHECKPOINT_NAME"
        fi
    fi
    
    sleep $CHECK_INTERVAL
done
```

### 4. 自定义评估任务

```python
from AFM.evaluation.comprehensive_eval import ComprehensiveEvaluator

# 创建评估器
evaluator = ComprehensiveEvaluator(
    model_path="/path/to/model",
    output_dir="./custom_eval",
    batch_size=4
)

# 自定义 NQ 评估（添加更多分析）
nq_file = "/path/to/nq_test.jsonl"
results = evaluator.eval_nq(nq_file)

# 分析错误案例
errors = [
    pred for pred in results["predictions"]
    if not pred["is_correct"]
]

print(f"Total errors: {len(errors)}")
print("\nError examples:")
for i, err in enumerate(errors[:5]):
    print(f"\n{i+1}. Question: {err['question']}")
    print(f"   Golden: {err['golden_answers']}")
    print(f"   Prediction: {err['prediction']}")

# 保存错误案例分析
import json
with open("./custom_eval/error_analysis.json", "w") as f:
    json.dump(errors, f, indent=2, ensure_ascii=False)
```

---

## 常见使用场景

### 场景 1：训练完成后的完整验证

```bash
# 训练刚完成，需要全面评估模型性能
bash AFM/evaluation/run_comprehensive_eval.sh
```

### 场景 2：快速验证微调效果

```bash
# 微调了 Story Agent，想快速看效果
python3 AFM/evaluation/comprehensive_eval.py \
    --model_path /path/to/finetuned/model \
    --datasets story \
    --max_samples 100
```

### 场景 3：对比基座模型和微调模型

```bash
# 评估基座模型
python3 AFM/evaluation/comprehensive_eval.py \
    --model_path /mnt/tongyan.zjy/openlm/model/Qwen/Qwen2.5-7B \
    --output_dir ./eval_results/base_model \
    --datasets mmlu ceval

# 评估微调模型
python3 AFM/evaluation/comprehensive_eval.py \
    --model_path /mnt/tongyan.zjy/model_output/AFM/AFM-StoryAgent-7B-sft \
    --output_dir ./eval_results/finetuned_model \
    --datasets mmlu ceval

# 对比结果
python3 compare_results.py \
    ./eval_results/base_model/comprehensive_eval_*.json \
    ./eval_results/finetuned_model/comprehensive_eval_*.json
```

### 场景 4：调试数据处理流程

```bash
# 使用极小样本数验证流程
python3 AFM/evaluation/comprehensive_eval.py \
    --model_path /path/to/model \
    --datasets story \
    --max_samples 5 \
    --batch_size 1
```

---

## 总结

### 推荐工作流程

1. **快速验证**：运行 `quick_test.sh`（5分钟）
2. **单数据集测试**：评估目标数据集的 100 个样本（30分钟）
3. **完整评估**：运行全量评估（12-15小时）
4. **结果分析**：生成对比报告和可视化

### 性能优化建议

- 使用多卡并行：`export CUDA_VISIBLE_DEVICES=0,1,2,3`
- 调整批处理大小：根据显存选择合适的 `batch_size`
- 分批评估：将大数据集拆分为多个任务并行运行
- 使用 FP16/BF16：已默认启用 `torch.bfloat16`

### 获取帮助

```bash
# 查看所有参数
python3 AFM/evaluation/comprehensive_eval.py --help

# 查看文档
cat AFM/evaluation/README_COMPREHENSIVE_EVAL.md
```

