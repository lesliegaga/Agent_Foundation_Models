# 综合测评脚本使用指南

## 概述

`comprehensive_eval.py` 是一个统一的多数据集测评脚本，支持以下数据集的自动化评估：

- **MMLU**：大规模多任务语言理解（英文，57个学科）
- **C-Eval**：中文综合能力评估（52个学科）
- **CMMLU**：中文多任务理解（67个学科）
- **NQ**：Natural Questions 问答数据集
- **Story Agent Thinking**：故事智能体思维数据集（支持 think+answer 格式）

## 功能特性

### 1. 统一接口
- 单一脚本支持所有数据集
- 标准化输出格式
- 自动结果汇总和报告生成

### 2. 智能评分
- **选择题**：基于 logits 的概率选择（准确高效）
- **问答题**：Exact Match (EM) 评分
- **思维链**：支持 `<think>...</think>` 格式的分离评估

### 3. 批量推理
- 支持 GPU 批处理加速
- 自动内存管理
- 进度条实时显示

### 4. 详细报告
- JSON 格式的完整结果
- 文本格式的摘要报告
- 按类别/学科细分的准确率

## 快速开始

### 1. 环境准备

```bash
# 确保已安装必要的包
pip install torch transformers datasets numpy tqdm

# 或使用项目环境
conda activate llama_factory
```

### 2. 数据准备

#### MMLU/C-Eval/CMMLU
数据集已包含在 `LLaMA-Factory/evaluation/` 目录中，无需额外下载。

#### NQ 数据集
```bash
# 下载 NQ 数据（如果还没有）
cd AFM/data/mhqa_agent
python download.py

# 数据路径：/mnt/tongyan.zjy/data/mhqa/nq_full.jsonl
```

#### Story Agent Thinking 数据集
```bash
# 由训练脚本自动生成的测试集
# 路径：/mnt/tongyan.zjy/data/story_room/sft/training_samples_test.jsonl
```

### 3. 运行评估

#### 方式一：使用 Shell 脚本（推荐）

```bash
cd /path/to/Agent_Foundation_Models

# 赋予执行权限
chmod +x AFM/evaluation/run_comprehensive_eval.sh

# 运行完整评估
bash AFM/evaluation/run_comprehensive_eval.sh
```

#### 方式二：直接运行 Python 脚本

```bash
cd /path/to/Agent_Foundation_Models

# 评估所有数据集
python3 AFM/evaluation/comprehensive_eval.py \
    --model_path /mnt/tongyan.zjy/openlm/model/Qwen/Qwen3-4B-Thinking-2507 \
    --output_dir ./eval_results \
    --datasets mmlu ceval cmmlu nq story \
    --n_shot 5 \
    --batch_size 4

# 只评估特定数据集
python3 AFM/evaluation/comprehensive_eval.py \
    --model_path /path/to/model \
    --output_dir ./eval_results \
    --datasets mmlu ceval \
    --n_shot 5

# 快速测试（限制样本数）
python3 AFM/evaluation/comprehensive_eval.py \
    --model_path /path/to/model \
    --output_dir ./eval_results \
    --datasets story nq \
    --max_samples 100
```

## 参数说明

### 必需参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | 模型路径 | `/mnt/tongyan.zjy/openlm/model/Qwen/Qwen3-4B-Thinking-2507` |
| `--output_dir` | 结果输出目录 | `./eval_results` |

### 可选参数

| 参数 | 说明 | 默认值 | 选项 |
|------|------|--------|------|
| `--datasets` | 要评估的数据集 | `mmlu ceval cmmlu nq story` | `mmlu`, `ceval`, `cmmlu`, `nq`, `story` |
| `--n_shot` | Few-shot 示例数 | `5` | 整数 |
| `--batch_size` | 批处理大小 | `4` | 整数 |
| `--max_length` | 最大序列长度 | `2048` | 整数 |
| `--nq_file` | NQ 数据文件路径 | `/mnt/tongyan.zjy/data/mhqa/nq_full.jsonl` | 文件路径 |
| `--story_file` | Story 数据文件路径 | `/mnt/tongyan.zjy/data/story_room/sft/training_samples_test.jsonl` | 文件路径 |
| `--max_samples` | 每个数据集的最大样本数 | `None`（全量） | 整数或 `None` |
| `--device` | 计算设备 | `cuda` | `cuda`, `cpu` |

## 输出结果

### 目录结构

```
eval_results/
└── 20251104_153045/
    ├── comprehensive_eval_20251104_153512.json  # 完整 JSON 结果
    └── summary_20251104_153512.txt              # 摘要报告
```

### JSON 结果格式

```json
{
  "model": "/path/to/model",
  "timestamp": "2025-11-04T15:35:12",
  "config": {
    "n_shot": 5,
    "batch_size": 4,
    "max_length": 2048
  },
  "results": {
    "mmlu": {
      "dataset": "MMLU",
      "overall_accuracy": 0.672,
      "category_results": {
        "STEM": {"accuracy": 0.65, "count": 5234},
        "Humanities": {"accuracy": 0.71, "count": 4521}
      },
      "subject_results": {
        "abstract_algebra": {"accuracy": 0.68, "correct": 68, "total": 100}
      }
    },
    "ceval": {
      "dataset": "C-Eval",
      "overall_accuracy": 0.725,
      ...
    },
    "nq": {
      "dataset": "NQ",
      "em_score": 0.685,
      "correct": 2473,
      "total": 3610
    },
    "story": {
      "dataset": "Story Agent Thinking",
      "full_accuracy": 0.452,
      "answer_accuracy": 0.678,
      "thinking_usage": 195
    }
  }
}
```

### 摘要报告示例

```
================================================================================
COMPREHENSIVE EVALUATION SUMMARY
================================================================================

Model: /mnt/tongyan.zjy/openlm/model/Qwen/Qwen3-4B-Thinking-2507
Timestamp: 2025-11-04T15:35:12
Config: {
  "n_shot": 5,
  "batch_size": 4,
  "max_length": 2048
}

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

--------------------------------------------------------------------------------
CMMLU
--------------------------------------------------------------------------------
Overall Accuracy: 71.80%

--------------------------------------------------------------------------------
NQ
--------------------------------------------------------------------------------
EM Score: 68.50%
Correct: 2473/3610

--------------------------------------------------------------------------------
STORY
--------------------------------------------------------------------------------
Answer Accuracy: 67.80%
Full Match: 45.20%

================================================================================
EVALUATION SUMMARY
================================================================================
MMLU: 67.20%
CEVAL: 72.50%
CMMLU: 71.80%
NQ: 68.50%
STORY: 67.80%
```

## 使用示例

### 示例 1：完整评估

```bash
bash AFM/evaluation/run_comprehensive_eval.sh
```

预期时间（4卡 A100）：
- MMLU: ~4 小时
- C-Eval: ~3 小时
- CMMLU: ~3.5 小时
- NQ: ~2 小时
- Story: ~1 小时
- **总计**: ~13-15 小时

### 示例 2：快速测试

```bash
# 修改 run_comprehensive_eval.sh 中的参数
MAX_SAMPLES=100  # 每个数据集只测试 100 个样本

bash AFM/evaluation/run_comprehensive_eval.sh
```

预期时间：~30 分钟

### 示例 3：单独评估 Story Agent

```python
from comprehensive_eval import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator(
    model_path="/path/to/model",
    output_dir="./results",
    batch_size=4
)

results = evaluator.eval_story_agent_thinking(
    data_file="/path/to/test.jsonl",
    max_samples=None  # 全量评估
)

print(f"Answer Accuracy: {results['answer_accuracy']:.2%}")
```

### 示例 4：自定义数据集评估

```python
# 评估自定义 QA 数据集
evaluator = ComprehensiveEvaluator(...)

# 数据格式：[{"question": "...", "target": ["ans1", "ans2"]}]
custom_results = evaluator.eval_nq(
    data_file="/path/to/custom_qa.jsonl",
    max_samples=None
)
```

## 评分机制详解

### 1. MMLU/C-Eval/CMMLU（选择题）

**方法**：Logits-based Choice Selection

```python
# 获取 A/B/C/D 四个选项的 logits
choice_logits = last_token_logits[:, [token_A, token_B, token_C, token_D]]

# Softmax 归一化
choice_probs = softmax(choice_logits)

# 选择概率最高的选项
prediction = argmax(choice_probs)
```

**优势**：
- 不依赖生成，速度快
- 避免格式错误
- 符合标准评估协议

### 2. NQ（问答题）

**方法**：Exact Match (EM)

```python
def normalize_answer(text):
    # 1. 转小写
    text = text.lower()
    
    # 2. 移除冠词 (a, an, the)
    text = remove_articles(text)
    
    # 3. 移除标点
    text = remove_punctuation(text)
    
    # 4. 标准化空格
    return whitespace_fix(text)

# 评分
is_correct = normalize(prediction) == normalize(golden_answer)
```

**特点**：
- 标准化后的精确匹配
- 支持多个可接受答案
- 严格但公平

### 3. Story Agent Thinking

**方法**：分离式评估

```python
# 提取 think 和 answer 部分
think_part = extract_between("<think>", "</think>")
answer_part = text_after("</think>")

# 两个评分维度
full_match = (think_part + answer_part) == golden_response
answer_match = normalize(answer_part) == normalize(golden_answer)
```

**指标**：
- `full_accuracy`: 完整响应匹配率
- `answer_accuracy`: 答案部分匹配率（更重要）
- `thinking_usage`: 使用思维链的比例

## 性能优化建议

### 1. 批处理大小

```bash
# 根据 GPU 显存调整
--batch_size 8   # 对于 A100 80GB
--batch_size 4   # 对于 A100 40GB
--batch_size 2   # 对于 V100 32GB
```

### 2. 序列长度

```bash
# 选择题不需要长序列
--max_length 2048   # MMLU/C-Eval/CMMLU

# QA 任务可能需要更长上下文
--max_length 4096   # NQ

# 思维链任务需要最长序列
--max_length 8192   # Story Agent Thinking
```

### 3. 多卡并行

目前脚本使用 `device_map="auto"` 自动分配，如需手动控制：

```bash
# 使用特定 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 或在代码中修改
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map={
        "model.embed_tokens": 0,
        "model.layers.0-17": 0,
        "model.layers.18-35": 1,
        ...
    }
)
```

## 故障排查

### 问题 1：CUDA Out of Memory

**解决方案**：

```bash
# 减小 batch_size
--batch_size 1

# 减小 max_length
--max_length 1024

# 使用 CPU（慢但稳定）
--device cpu
```

### 问题 2：数据集加载失败

**检查**：

```bash
# 验证数据集文件
ls -lh LLaMA-Factory/evaluation/mmlu/mmlu.zip

# 手动解压（如果需要）
cd LLaMA-Factory/evaluation/mmlu
unzip mmlu.zip
```

### 问题 3：模型推理速度慢

**优化**：

```bash
# 使用 BF16（已默认启用）
torch_dtype=torch.bfloat16

# 使用 Flash Attention（如果支持）
pip install flash-attn --no-build-isolation

# 增大 batch_size
--batch_size 8
```

### 问题 4：Story Agent Thinking 答案提取失败

**调试**：

```python
# 检查数据格式
import json
with open("/path/to/test.jsonl") as f:
    sample = json.loads(f.readline())
    print(sample)

# 期望格式
{
  "prompt": "问题...",
  "response": "<think>思考过程</think>\n答案内容",
  # 或
  "answer": "答案内容"
}
```

## 高级用法

### 1. 自定义评估流程

```python
from comprehensive_eval import ComprehensiveEvaluator

class CustomEvaluator(ComprehensiveEvaluator):
    def eval_custom_dataset(self, data_file):
        """实现自定义数据集评估"""
        # 加载数据
        with open(data_file) as f:
            data = [json.loads(line) for line in f]
        
        # 推理和评分
        for item in data:
            prompt = self.format_prompt(item)
            response = self.generate_response(prompt)
            score = self.compute_score(response, item['answer'])
        
        return results

# 使用
evaluator = CustomEvaluator(...)
results = evaluator.eval_custom_dataset("custom.jsonl")
```

### 2. 批量评估多个模型

```bash
#!/bin/bash

MODELS=(
    "/path/to/model-v1"
    "/path/to/model-v2"
    "/path/to/model-v3"
)

for model in "${MODELS[@]}"; do
    echo "Evaluating $model"
    python3 comprehensive_eval.py \
        --model_path "$model" \
        --output_dir "./results/$(basename $model)" \
        --datasets mmlu ceval
done
```

### 3. 结果对比分析

```python
import json
import pandas as pd

# 加载多个结果
results = []
for file in ["model1.json", "model2.json", "model3.json"]:
    with open(file) as f:
        results.append(json.load(f))

# 创建对比表格
data = []
for res in results:
    data.append({
        "Model": res["model"].split("/")[-1],
        "MMLU": res["results"]["mmlu"]["overall_accuracy"],
        "C-Eval": res["results"]["ceval"]["overall_accuracy"],
        "NQ": res["results"]["nq"]["em_score"]
    })

df = pd.DataFrame(data)
print(df.to_markdown())
```

## 参考资料

- [MMLU 论文](https://arxiv.org/abs/2009.03300)
- [C-Eval 论文](https://arxiv.org/abs/2305.08322)
- [CMMLU 项目](https://github.com/haonan-li/CMMLU)
- [Natural Questions](https://ai.google.com/research/NaturalQuestions)

## 贡献与反馈

如有问题或建议，请提交 Issue 或 Pull Request。

