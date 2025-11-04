# AFM 模型测评工具集

## 📋 概述

本目录包含了 AFM (Agent Foundation Model) 项目的完整测评工具集，支持多个数据集和多种评估场景。

## 🗂️ 文件结构

```
AFM/evaluation/
├── README.md                          # 本文件
├── EVALUATION_MANUAL.md               # 完整的测评体系手册
├── README_COMPREHENSIVE_EVAL.md       # 综合测评脚本详细文档
├── USAGE_EXAMPLES.md                  # 使用示例大全
│
├── comprehensive_eval.py              # 主评估脚本（Python）
├── run_comprehensive_eval.sh          # 完整评估执行脚本（Shell）
├── quick_test.sh                      # 快速测试脚本
│
├── code_agent/
│   └── eval_code_agent.sh            # Code Agent 专项评估
├── mhqa_agent/
│   └── eval_mhqa_agent.sh            # MHQA Agent 专项评估
└── web_agent/
    ├── inference_web_agent.py        # Web Agent 推理脚本
    ├── run_qwen.sh                   # Web Agent 服务部署
    └── ...                           # 其他 Web Agent 工具
```

## 🚀 快速开始

### 1分钟快速验证

```bash
cd /path/to/Agent_Foundation_Models

# 快速测试（每个数据集 50 个样本，约 3-5 分钟）
bash AFM/evaluation/quick_test.sh
```

### 完整评估

```bash
# 评估所有数据集（约 13-15 小时）
bash AFM/evaluation/run_comprehensive_eval.sh
```

## 📊 支持的数据集

### 通用能力测评
- **MMLU**：大规模多任务语言理解（57个学科，英文）
- **C-Eval**：中文综合能力评估（52个学科）
- **CMMLU**：中文多任务理解（67个学科）

### Agent 专业能力测评
- **NQ**：Natural Questions 问答数据集
- **Story Agent Thinking**：故事智能体思维数据集（支持 think+answer 格式）
- **MHQA**：多跳问答（需使用专项脚本）
- **Code**：代码生成（需使用专项脚本）
- **Web**：网页交互（需使用专项脚本）

## 🛠️ 使用方式

### 方式一：Shell 脚本（推荐）

```bash
# 完整评估
bash AFM/evaluation/run_comprehensive_eval.sh

# 快速测试
bash AFM/evaluation/quick_test.sh
```

### 方式二：Python 直接调用

```bash
# 评估所有数据集
python3 AFM/evaluation/comprehensive_eval.py \
    --model_path /mnt/tongyan.zjy/openlm/model/Qwen/Qwen3-4B-Thinking-2507 \
    --output_dir ./eval_results \
    --datasets mmlu ceval cmmlu nq story

# 评估特定数据集
python3 AFM/evaluation/comprehensive_eval.py \
    --model_path /path/to/model \
    --datasets story nq \
    --max_samples 100
```

### 方式三：Python API

```python
from AFM.evaluation.comprehensive_eval import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator(
    model_path="/path/to/model",
    output_dir="./results",
    batch_size=4
)

# 运行评估
results = evaluator.run_comprehensive_evaluation(
    datasets=["mmlu", "ceval", "story"],
    n_shot=5
)

# 或单独评估
story_results = evaluator.eval_story_agent_thinking(
    data_file="/path/to/test.jsonl"
)
```

## 📝 核心参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--model_path` | 模型路径 | - | `/path/to/model` |
| `--output_dir` | 结果输出目录 | `./eval_results` | `./my_results` |
| `--datasets` | 要评估的数据集 | `mmlu ceval cmmlu nq story` | `mmlu story` |
| `--n_shot` | Few-shot 示例数 | `5` | `3` |
| `--batch_size` | 批处理大小 | `4` | `8` |
| `--max_samples` | 每个数据集最大样本数 | `None`（全量） | `100` |

**查看所有参数**：
```bash
python3 AFM/evaluation/comprehensive_eval.py --help
```

## 📈 评估结果

### 输出文件

```
eval_results/
└── 20251104_160000/
    ├── comprehensive_eval_20251104_160512.json  # 完整 JSON 结果
    └── summary_20251104_160512.txt              # 摘要报告
```

### 结果格式

**JSON 结果**：
```json
{
  "model": "/path/to/model",
  "timestamp": "2025-11-04T16:05:12",
  "results": {
    "mmlu": {
      "overall_accuracy": 0.672,
      "category_results": {...}
    },
    "story": {
      "answer_accuracy": 0.678,
      "full_accuracy": 0.452
    }
  }
}
```

**摘要报告**：
```
MMLU: 67.20%
C-Eval: 72.50%
CMMLU: 71.80%
NQ: 68.50%
Story: 67.80%
```

## 📚 文档索引

- **[EVALUATION_MANUAL.md](./EVALUATION_MANUAL.md)**：完整的测评体系手册
  - 测评架构详解
  - 数据格式说明
  - 评分机制原理
  - Agent 专业能力测评
  - 通用能力测评
  - 完整实战指南

- **[README_COMPREHENSIVE_EVAL.md](./README_COMPREHENSIVE_EVAL.md)**：综合测评脚本文档
  - 功能特性
  - 参数详解
  - 输出格式
  - 性能优化
  - 故障排查

- **[USAGE_EXAMPLES.md](./USAGE_EXAMPLES.md)**：实用示例大全
  - 快速开始
  - 完整评估
  - 单独数据集评估
  - 自定义配置
  - 结果分析
  - 高级用法

## 🎯 典型使用场景

### 场景 1：训练完成后的验证

```bash
# 完整评估所有能力
bash AFM/evaluation/run_comprehensive_eval.sh
```

### 场景 2：快速验证微调效果

```bash
# 只评估目标数据集
python3 AFM/evaluation/comprehensive_eval.py \
    --datasets story \
    --max_samples 100
```

### 场景 3：对比不同模型

```bash
# 评估基座模型
python3 AFM/evaluation/comprehensive_eval.py \
    --model_path /path/to/base/model \
    --output_dir ./results/base

# 评估微调模型
python3 AFM/evaluation/comprehensive_eval.py \
    --model_path /path/to/finetuned/model \
    --output_dir ./results/finetuned
```

### 场景 4：持续监控训练进度

```bash
# 定期评估最新 checkpoint
watch -n 3600 "bash AFM/evaluation/quick_test.sh"
```

## ⚡ 性能优化

### GPU 配置

```bash
# 指定 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 调整批处理大小（根据显存）
--batch_size 8   # A100 80GB
--batch_size 4   # A100 40GB
--batch_size 2   # V100 32GB
```

### 并行评估

```bash
# 同时评估不同数据集
python3 comprehensive_eval.py --datasets mmlu &
python3 comprehensive_eval.py --datasets ceval &
python3 comprehensive_eval.py --datasets story &
wait
```

### 预期时间（4卡 A100）

| 数据集 | 样本数 | 时间 |
|--------|--------|------|
| MMLU   | ~14,000 | 4h |
| C-Eval | ~13,948 | 3h |
| CMMLU  | ~11,500 | 3.5h |
| NQ     | ~3,610  | 2h |
| Story  | ~200    | 1h |
| **总计** | ~43,258 | **13-15h** |

## 🔍 故障排查

### Q1: CUDA Out of Memory

```bash
# 减小批处理大小
--batch_size 1

# 减小序列长度
--max_length 1024
```

### Q2: 数据集加载失败

```bash
# 检查数据集文件
ls -lh LLaMA-Factory/evaluation/mmlu/
ls -lh /mnt/tongyan.zjy/data/story_room/sft/
```

### Q3: 模型推理速度慢

```bash
# 增大批处理大小
--batch_size 8

# 使用更少的 few-shot 示例
--n_shot 3
```

### Q4: 结果不符合预期

```bash
# 使用小样本验证流程
--max_samples 10

# 检查日志
tail -f logs/comprehensive_eval_*.log
```

## 📖 评分机制

### 选择题（MMLU/C-Eval/CMMLU）
- **方法**：Logits-based Choice Selection
- **优势**：准确、高效、标准

### 问答题（NQ）
- **方法**：Exact Match (EM)
- **特点**：标准化后精确匹配

### 思维链（Story）
- **方法**：分离式评估
- **指标**：`full_accuracy` + `answer_accuracy`

详细说明见 [EVALUATION_MANUAL.md](./EVALUATION_MANUAL.md)

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 添加新数据集

1. 在 `ComprehensiveEvaluator` 中添加 `eval_your_dataset()` 方法
2. 更新 `run_comprehensive_evaluation()` 中的数据集列表
3. 添加相应的文档和示例

### 改进评分机制

1. 修改相应的评分函数（如 `extract_answer_from_response()`）
2. 添加单元测试
3. 更新文档

## 📞 获取帮助

- **文档**：查看 `EVALUATION_MANUAL.md` 和 `README_COMPREHENSIVE_EVAL.md`
- **示例**：参考 `USAGE_EXAMPLES.md`
- **问题**：提交 Issue

## 📄 许可证

本项目遵循 Apache 2.0 许可证。

---

**最后更新**：2025-11-04  
**维护者**：AFM Team

