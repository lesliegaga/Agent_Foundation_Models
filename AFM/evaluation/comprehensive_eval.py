#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive Evaluation Script for AFM Models
支持 MMLU, C-Eval, CMMLU, NQ, Story Agent Thinking 等多个数据集的测评
"""

import argparse
import json
import os
import re
import string
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加 LLaMA-Factory 到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "LLaMA-Factory" / "src"))


class ComprehensiveEvaluator:
    """综合测评器，支持多种数据集和评估方式"""
    
    def __init__(
        self,
        model_path: str,
        output_dir: str,
        device: str = "cuda",
        batch_size: int = 4,
        max_length: int = 2048,
    ):
        """
        初始化评估器
        
        Args:
            model_path: 模型路径
            output_dir: 结果输出目录
            device: 设备 (cuda/cpu)
            batch_size: 批处理大小
            max_length: 最大序列长度
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"  # 用于批处理
        )
        
        # 设置 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        
        print(f"Model loaded successfully on {device}")
        
        # 存储结果
        self.all_results = {}
    
    def normalize_answer(self, text: str) -> str:
        """标准化答案（用于 EM 评分）"""
        # 移除冠词
        text = re.sub(r"\b(a|an|the)\b", " ", text.lower())
        # 移除标点
        text = "".join(ch for ch in text if ch not in string.punctuation)
        # 标准化空格
        return " ".join(text.split())
    
    def extract_answer_from_response(self, response: str, mode: str = "choice") -> str:
        """
        从模型响应中提取答案
        
        Args:
            response: 模型生成的文本
            mode: 提取模式 ('choice' 用于选择题, 'qa' 用于问答题)
        
        Returns:
            提取的答案
        """
        if mode == "choice":
            # 选择题模式：提取 A/B/C/D
            # 尝试多种匹配模式
            patterns = [
                r"答案[是为]?\s*[:：]?\s*([ABCD])",
                r"选择\s*([ABCD])",
                r"^([ABCD])\s*[.。]",
                r"\b([ABCD])\b",
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
                if match:
                    return match.group(1).upper()
            
            # 如果没找到，返回响应的第一个字符（如果是 A-D）
            first_char = response.strip()[0] if response.strip() else ""
            if first_char.upper() in "ABCD":
                return first_char.upper()
            
            return ""
        
        elif mode == "qa":
            # QA 模式：提取 <answer> 标签中的内容
            answer_pattern = r'<answer>(.*?)</answer>'
            matches = list(re.finditer(answer_pattern, response, re.DOTALL))
            
            if matches:
                # 返回最后一个 answer 标签的内容
                return matches[-1].group(1).strip()
            
            # 如果没有标签，返回全部内容（清理后）
            return response.strip()
        
        return response.strip()
    
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        do_sample: bool = False
    ) -> str:
        """生成模型响应"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response
    
    def batch_generate_choice(
        self,
        prompts: List[str],
    ) -> List[str]:
        """批量生成选择题答案（使用 logits 选择）"""
        # 编码输入
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True
        ).to(self.model.device)
        
        # 获取 A/B/C/D 的 token id
        choice_tokens = {
            ch: self.tokenizer.encode(ch, add_special_tokens=False)[-1]
            for ch in "ABCD"
        }
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # 获取最后一个 token 的 logits
        lengths = inputs['attention_mask'].sum(dim=-1)
        last_logits = torch.stack([
            logits[i, lengths[i] - 1]
            for i in range(len(lengths))
        ])
        
        # 计算选择的概率
        choice_ids = [choice_tokens[ch] for ch in "ABCD"]
        choice_logits = last_logits[:, choice_ids]
        choice_probs = torch.softmax(choice_logits, dim=-1)
        
        # 选择概率最高的
        predictions = torch.argmax(choice_probs, dim=-1)
        answers = ["ABCD"[pred.item()] for pred in predictions]
        
        return answers
    
    def eval_mmlu(self, n_shot: int = 5) -> Dict[str, Any]:
        """评估 MMLU 数据集"""
        print("\n" + "="*80)
        print("Evaluating MMLU...")
        print("="*80)
        
        eval_dir = Path(__file__).parent.parent.parent / "LLaMA-Factory" / "evaluation" / "mmlu"
        
        # 加载 mapping
        with open(eval_dir / "mapping.json") as f:
            subject_mapping = json.load(f)
        
        results = {}
        category_corrects = {}
        
        for subject_key, subject_info in tqdm(subject_mapping.items(), desc="MMLU Subjects"):
            subject_name = subject_info["name"]
            category = subject_info["category"]
            
            # 加载数据集
            dataset = load_dataset(
                str(eval_dir / "mmlu.py"),
                name=subject_key,
                trust_remote_code=True
            )
            
            # 准备测试样本
            test_data = dataset["test"]
            train_data = dataset.get("dev", dataset.get("validation", []))
            
            prompts = []
            labels = []
            
            for idx in range(len(test_data)):
                # 构建 few-shot prompt
                prompt_parts = [
                    f"The following are multiple choice questions (with answers) about {subject_name}.\n\n"
                ]
                
                # 添加 few-shot 示例
                if n_shot > 0 and len(train_data) > 0:
                    few_shot_samples = train_data.shuffle(seed=42).select(
                        range(min(n_shot, len(train_data)))
                    )
                    for sample in few_shot_samples:
                        q = sample["question"]
                        choices_text = "\n".join([
                            f"{ch}. {sample[ch]}"
                            for ch in "ABCD"
                            if ch in sample
                        ])
                        ans = sample["answer"]
                        prompt_parts.append(f"{q}\n{choices_text}\nAnswer: {ans}\n\n")
                
                # 添加测试问题
                test_sample = test_data[idx]
                q = test_sample["question"]
                choices_text = "\n".join([
                    f"{ch}. {test_sample[ch]}"
                    for ch in "ABCD"
                    if ch in test_sample
                ])
                prompt_parts.append(f"{q}\n{choices_text}\nAnswer:")
                
                prompts.append("".join(prompt_parts))
                labels.append(test_sample["answer"])
            
            # 批量推理
            predictions = []
            for i in range(0, len(prompts), self.batch_size):
                batch_prompts = prompts[i:i + self.batch_size]
                batch_preds = self.batch_generate_choice(batch_prompts)
                predictions.extend(batch_preds)
            
            # 计算准确率
            corrects = np.array(predictions) == np.array(labels)
            accuracy = corrects.mean()
            
            results[subject_key] = {
                "accuracy": float(accuracy),
                "correct": int(corrects.sum()),
                "total": len(corrects)
            }
            
            # 按类别统计
            if category not in category_corrects:
                category_corrects[category] = []
            category_corrects[category].extend(corrects.tolist())
        
        # 计算类别和总体准确率
        category_results = {}
        for category, corrects in category_corrects.items():
            category_results[category] = {
                "accuracy": float(np.mean(corrects)),
                "count": len(corrects)
            }
        
        overall_accuracy = np.mean([
            result["accuracy"] for result in results.values()
        ])
        
        final_results = {
            "dataset": "MMLU",
            "n_shot": n_shot,
            "overall_accuracy": float(overall_accuracy),
            "category_results": category_results,
            "subject_results": results
        }
        
        print(f"\nMMLU Overall Accuracy: {overall_accuracy:.2%}")
        for category, res in category_results.items():
            print(f"  {category}: {res['accuracy']:.2%}")
        
        return final_results
    
    def eval_ceval(self, n_shot: int = 5) -> Dict[str, Any]:
        """评估 C-Eval 数据集"""
        print("\n" + "="*80)
        print("Evaluating C-Eval...")
        print("="*80)
        
        eval_dir = Path(__file__).parent.parent.parent / "LLaMA-Factory" / "evaluation" / "ceval"
        
        # 加载 mapping
        with open(eval_dir / "mapping.json") as f:
            subject_mapping = json.load(f)
        
        results = {}
        category_corrects = {}
        
        for subject_key, subject_info in tqdm(subject_mapping.items(), desc="C-Eval Subjects"):
            subject_name = subject_info["name"]
            category = subject_info["category"]
            
            # 加载数据集
            dataset = load_dataset(
                str(eval_dir / "ceval.py"),
                name=subject_key,
                trust_remote_code=True
            )
            
            # 准备测试样本
            test_data = dataset["val"]
            train_data = dataset.get("dev", [])
            
            prompts = []
            labels = []
            
            for idx in range(len(test_data)):
                # 构建 few-shot prompt（中文）
                prompt_parts = [
                    f"以下是中国关于{subject_name}考试的单项选择题，请选出其中的正确答案。\n\n"
                ]
                
                # 添加 few-shot 示例
                if n_shot > 0 and len(train_data) > 0:
                    few_shot_samples = train_data.shuffle(seed=42).select(
                        range(min(n_shot, len(train_data)))
                    )
                    for sample in few_shot_samples:
                        q = sample["question"]
                        choices_text = "\n".join([
                            f"{ch}. {sample[ch]}"
                            for ch in "ABCD"
                            if ch in sample
                        ])
                        ans = sample["answer"]
                        prompt_parts.append(f"{q}\n{choices_text}\n答案：{ans}\n\n")
                
                # 添加测试问题
                test_sample = test_data[idx]
                q = test_sample["question"]
                choices_text = "\n".join([
                    f"{ch}. {test_sample[ch]}"
                    for ch in "ABCD"
                    if ch in test_sample
                ])
                prompt_parts.append(f"{q}\n{choices_text}\n答案：")
                
                prompts.append("".join(prompt_parts))
                labels.append(test_sample["answer"])
            
            # 批量推理
            predictions = []
            for i in range(0, len(prompts), self.batch_size):
                batch_prompts = prompts[i:i + self.batch_size]
                batch_preds = self.batch_generate_choice(batch_prompts)
                predictions.extend(batch_preds)
            
            # 计算准确率
            corrects = np.array(predictions) == np.array(labels)
            accuracy = corrects.mean()
            
            results[subject_key] = {
                "accuracy": float(accuracy),
                "correct": int(corrects.sum()),
                "total": len(corrects)
            }
            
            # 按类别统计
            if category not in category_corrects:
                category_corrects[category] = []
            category_corrects[category].extend(corrects.tolist())
        
        # 计算类别和总体准确率
        category_results = {}
        for category, corrects in category_corrects.items():
            category_results[category] = {
                "accuracy": float(np.mean(corrects)),
                "count": len(corrects)
            }
        
        overall_accuracy = np.mean([
            result["accuracy"] for result in results.values()
        ])
        
        final_results = {
            "dataset": "C-Eval",
            "n_shot": n_shot,
            "overall_accuracy": float(overall_accuracy),
            "category_results": category_results,
            "subject_results": results
        }
        
        print(f"\nC-Eval Overall Accuracy: {overall_accuracy:.2%}")
        for category, res in category_results.items():
            print(f"  {category}: {res['accuracy']:.2%}")
        
        return final_results
    
    def eval_cmmlu(self, n_shot: int = 5) -> Dict[str, Any]:
        """评估 CMMLU 数据集"""
        print("\n" + "="*80)
        print("Evaluating CMMLU...")
        print("="*80)
        
        eval_dir = Path(__file__).parent.parent.parent / "LLaMA-Factory" / "evaluation" / "cmmlu"
        
        # 加载 mapping
        with open(eval_dir / "mapping.json") as f:
            subject_mapping = json.load(f)
        
        results = {}
        category_corrects = {}
        
        for subject_key, subject_info in tqdm(subject_mapping.items(), desc="CMMLU Subjects"):
            subject_name = subject_info["name"]
            category = subject_info["category"]
            
            # 加载数据集
            dataset = load_dataset(
                str(eval_dir / "cmmlu.py"),
                name=subject_key,
                trust_remote_code=True
            )
            
            # 准备测试样本
            test_data = dataset["test"]
            train_data = dataset.get("dev", [])
            
            prompts = []
            labels = []
            
            for idx in range(len(test_data)):
                # 构建 few-shot prompt（中文）
                prompt_parts = [
                    f"以下是中国关于{subject_name}考试的单项选择题，请选出其中的正确答案。\n\n"
                ]
                
                # 添加 few-shot 示例
                if n_shot > 0 and len(train_data) > 0:
                    few_shot_samples = train_data.shuffle(seed=42).select(
                        range(min(n_shot, len(train_data)))
                    )
                    for sample in few_shot_samples:
                        q = sample["question"]
                        choices_text = "\n".join([
                            f"{ch}. {sample[ch]}"
                            for ch in "ABCD"
                            if ch in sample
                        ])
                        ans = sample["answer"]
                        prompt_parts.append(f"{q}\n{choices_text}\n答案：{ans}\n\n")
                
                # 添加测试问题
                test_sample = test_data[idx]
                q = test_sample["question"]
                choices_text = "\n".join([
                    f"{ch}. {test_sample[ch]}"
                    for ch in "ABCD"
                    if ch in test_sample
                ])
                prompt_parts.append(f"{q}\n{choices_text}\n答案：")
                
                prompts.append("".join(prompt_parts))
                labels.append(test_sample["answer"])
            
            # 批量推理
            predictions = []
            for i in range(0, len(prompts), self.batch_size):
                batch_prompts = prompts[i:i + self.batch_size]
                batch_preds = self.batch_generate_choice(batch_prompts)
                predictions.extend(batch_preds)
            
            # 计算准确率
            corrects = np.array(predictions) == np.array(labels)
            accuracy = corrects.mean()
            
            results[subject_key] = {
                "accuracy": float(accuracy),
                "correct": int(corrects.sum()),
                "total": len(corrects)
            }
            
            # 按类别统计
            if category not in category_corrects:
                category_corrects[category] = []
            category_corrects[category].extend(corrects.tolist())
        
        # 计算类别和总体准确率
        category_results = {}
        for category, corrects in category_corrects.items():
            category_results[category] = {
                "accuracy": float(np.mean(corrects)),
                "count": len(corrects)
            }
        
        overall_accuracy = np.mean([
            result["accuracy"] for result in results.values()
        ])
        
        final_results = {
            "dataset": "CMMLU",
            "n_shot": n_shot,
            "overall_accuracy": float(overall_accuracy),
            "category_results": category_results,
            "subject_results": results
        }
        
        print(f"\nCMMLU Overall Accuracy: {overall_accuracy:.2%}")
        for category, res in category_results.items():
            print(f"  {category}: {res['accuracy']:.2%}")
        
        return final_results
    
    def eval_nq(self, data_file: str, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """评估 NQ (Natural Questions) 数据集"""
        print("\n" + "="*80)
        print("Evaluating NQ...")
        print("="*80)
        
        # 加载数据
        with open(data_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        if max_samples:
            data = data[:max_samples]
        
        correct = 0
        total = len(data)
        predictions_list = []
        
        for item in tqdm(data, desc="NQ Evaluation"):
            question = item["question"]
            # 兼容不同字段：优先使用 target，其次使用 answer
            if "target" in item:
                golden_field = item["target"]
            elif "answer" in item:
                golden_field = item["answer"]
            else:
                golden_field = []
            golden_answers = golden_field if isinstance(golden_field, list) else [golden_field]
            
            # 构建 prompt
            prompt = f"Question: {question}\nPlease provide a concise answer.\nAnswer:"
            
            # 生成答案
            response = self.generate_response(
                prompt,
                max_new_tokens=100,
                temperature=0.1
            )
            
            prediction = self.extract_answer_from_response(response, mode="qa")
            
            # EM 评分
            normalized_pred = self.normalize_answer(prediction)
            is_correct = any(
                self.normalize_answer(ans) == normalized_pred
                for ans in golden_answers
            )
            
            if is_correct:
                correct += 1
            
            predictions_list.append({
                "question": question,
                "golden_answers": golden_answers,
                "prediction": prediction,
                "is_correct": is_correct
            })
        
        accuracy = correct / total if total > 0 else 0.0
        
        results = {
            "dataset": "NQ",
            "accuracy": accuracy,
            "em_score": accuracy,
            "correct": correct,
            "total": total,
            "predictions": predictions_list[:100]  # 只保存前100个预测示例
        }
        
        print(f"\nNQ EM Score: {accuracy:.2%} ({correct}/{total})")
        
        return results
    
    def eval_story_agent_thinking(
        self,
        data_file: str,
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """评估 Story Agent Thinking 数据集（支持 think + answer 格式）"""
        print("\n" + "="*80)
        print("Evaluating Story Agent Thinking...")
        print("="*80)
        
        # 加载数据
        with open(data_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        if max_samples:
            data = data[:max_samples]
        
        total_correct = 0
        answer_correct = 0
        total = len(data)
        predictions_list = []
        
        for item in tqdm(data, desc="Story Agent Thinking Evaluation"):
            prompt = item["prompt"]
            
            # 获取标准答案（可能包含 think 和 answer）
            if "response" in item:
                # 格式: <think>...</think>\n答案内容
                full_response = item["response"]
                # 提取 answer 部分
                if "</think>" in full_response:
                    golden_answer = full_response.split("</think>")[-1].strip()
                else:
                    golden_answer = full_response
            elif "answer" in item:
                golden_answer = item["answer"]
            else:
                golden_answer = ""
            
            # 生成答案（期望模型也生成 <think>...</think>\n答案 格式）
            response = self.generate_response(
                prompt,
                max_new_tokens=2048,
                temperature=0.7,
                do_sample=True
            )
            
            # 提取模型的 answer 部分
            if "</think>" in response:
                pred_answer = response.split("</think>")[-1].strip()
            else:
                pred_answer = response
            
            # 评分：完整响应匹配 vs 仅 answer 部分匹配
            full_match = response.strip() == full_response.strip() if "response" in item else False
            
            # Answer 部分的语义匹配（简化版，实际可用 LLM Judge）
            normalized_pred = self.normalize_answer(pred_answer[:500])  # 限制长度
            normalized_golden = self.normalize_answer(golden_answer[:500])
            answer_match = normalized_pred == normalized_golden or normalized_golden in normalized_pred
            
            if full_match:
                total_correct += 1
            if answer_match:
                answer_correct += 1
            
            predictions_list.append({
                "prompt": prompt[:200] + "...",  # 截断显示
                "golden_answer": golden_answer[:200] + "...",
                "prediction": pred_answer[:200] + "...",
                "full_match": full_match,
                "answer_match": answer_match,
                "has_thinking": "</think>" in response
            })
        
        full_accuracy = total_correct / total if total > 0 else 0.0
        answer_accuracy = answer_correct / total if total > 0 else 0.0
        
        results = {
            "dataset": "Story Agent Thinking",
            "full_accuracy": full_accuracy,
            "answer_accuracy": answer_accuracy,
            "total_correct": total_correct,
            "answer_correct": answer_correct,
            "total": total,
            "thinking_usage": sum(1 for p in predictions_list if p["has_thinking"]),
            "predictions": predictions_list[:50]  # 保存前50个预测示例
        }
        
        print(f"\nStory Agent Thinking Results:")
        print(f"  Full Match Accuracy: {full_accuracy:.2%} ({total_correct}/{total})")
        print(f"  Answer Match Accuracy: {answer_accuracy:.2%} ({answer_correct}/{total})")
        print(f"  Thinking Usage: {results['thinking_usage']}/{total}")
        
        return results
    
    def run_comprehensive_evaluation(
        self,
        datasets: List[str],
        n_shot: int = 5,
        nq_file: Optional[str] = None,
        story_file: Optional[str] = None,
        max_samples_per_dataset: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        运行综合评估
        
        Args:
            datasets: 要评估的数据集列表 ['mmlu', 'ceval', 'cmmlu', 'nq', 'story']
            n_shot: Few-shot 示例数量
            nq_file: NQ 数据文件路径
            story_file: Story Agent 数据文件路径
            max_samples_per_dataset: 每个数据集的最大样本数（用于快速测试）
        
        Returns:
            所有评估结果的字典
        """
        all_results = {
            "model": self.model_path,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "n_shot": n_shot,
                "batch_size": self.batch_size,
                "max_length": self.max_length,
                "max_samples_per_dataset": max_samples_per_dataset
            },
            "results": {}
        }
        
        # MMLU
        if "mmlu" in datasets:
            try:
                mmlu_results = self.eval_mmlu(n_shot=n_shot)
                all_results["results"]["mmlu"] = mmlu_results
            except Exception as e:
                print(f"Error evaluating MMLU: {e}")
                all_results["results"]["mmlu"] = {"error": str(e)}
        
        # C-Eval
        if "ceval" in datasets:
            try:
                ceval_results = self.eval_ceval(n_shot=n_shot)
                all_results["results"]["ceval"] = ceval_results
            except Exception as e:
                print(f"Error evaluating C-Eval: {e}")
                all_results["results"]["ceval"] = {"error": str(e)}
        
        # CMMLU
        if "cmmlu" in datasets:
            try:
                cmmlu_results = self.eval_cmmlu(n_shot=n_shot)
                all_results["results"]["cmmlu"] = cmmlu_results
            except Exception as e:
                print(f"Error evaluating CMMLU: {e}")
                all_results["results"]["cmmlu"] = {"error": str(e)}
        
        # NQ
        if "nq" in datasets:
            if nq_file and os.path.exists(nq_file):
                try:
                    nq_results = self.eval_nq(nq_file, max_samples=max_samples_per_dataset)
                    all_results["results"]["nq"] = nq_results
                except Exception as e:
                    print(f"Error evaluating NQ: {e}")
                    all_results["results"]["nq"] = {"error": str(e)}
            else:
                print(f"Warning: NQ file not found at {nq_file}")
        
        # Story Agent Thinking
        if "story" in datasets:
            if story_file and os.path.exists(story_file):
                try:
                    story_results = self.eval_story_agent_thinking(
                        story_file,
                        max_samples=max_samples_per_dataset
                    )
                    all_results["results"]["story"] = story_results
                except Exception as e:
                    print(f"Error evaluating Story Agent Thinking: {e}")
                    all_results["results"]["story"] = {"error": str(e)}
            else:
                print(f"Warning: Story file not found at {story_file}")
        
        # 保存结果
        self.save_results(all_results)
        
        return all_results
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """保存评估结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细 JSON 结果
        json_file = self.output_dir / f"comprehensive_eval_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {json_file}")
        
        # 生成摘要报告
        summary_file = self.output_dir / f"summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE EVALUATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Model: {results['model']}\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Config: {json.dumps(results['config'], indent=2)}\n\n")
            
            for dataset_name, dataset_results in results["results"].items():
                f.write("-"*80 + "\n")
                f.write(f"{dataset_name.upper()}\n")
                f.write("-"*80 + "\n")
                
                if "error" in dataset_results:
                    f.write(f"Error: {dataset_results['error']}\n\n")
                    continue
                
                if "overall_accuracy" in dataset_results:
                    f.write(f"Overall Accuracy: {dataset_results['overall_accuracy']:.2%}\n")
                
                if "category_results" in dataset_results:
                    f.write("\nCategory Results:\n")
                    for cat, res in dataset_results["category_results"].items():
                        f.write(f"  {cat}: {res['accuracy']:.2%}\n")
                
                if "em_score" in dataset_results:
                    f.write(f"EM Score: {dataset_results['em_score']:.2%}\n")
                    f.write(f"Correct: {dataset_results['correct']}/{dataset_results['total']}\n")
                
                if "answer_accuracy" in dataset_results:
                    f.write(f"Answer Accuracy: {dataset_results['answer_accuracy']:.2%}\n")
                    f.write(f"Full Match: {dataset_results['full_accuracy']:.2%}\n")
                
                f.write("\n")
        
        print(f"Summary saved to: {summary_file}")
        
        # 打印到控制台
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        for dataset_name, dataset_results in results["results"].items():
            if "error" in dataset_results:
                print(f"{dataset_name.upper()}: ERROR - {dataset_results['error']}")
            elif "overall_accuracy" in dataset_results:
                print(f"{dataset_name.upper()}: {dataset_results['overall_accuracy']:.2%}")
            elif "em_score" in dataset_results:
                print(f"{dataset_name.upper()}: {dataset_results['em_score']:.2%}")
            elif "answer_accuracy" in dataset_results:
                print(f"{dataset_name.upper()}: {dataset_results['answer_accuracy']:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Model Evaluation")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/tongyan.zjy/openlm/model/Qwen/Qwen3-4B-Thinking-2507",
        help="Path to the model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["mmlu", "ceval", "cmmlu", "nq", "story"],
        choices=["mmlu", "ceval", "cmmlu", "nq", "story"],
        help="Datasets to evaluate"
    )
    parser.add_argument(
        "--n_shot",
        type=int,
        default=5,
        help="Number of few-shot examples"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--nq_file",
        type=str,
        default="/mnt/tongyan.zjy/data/mhqa/nq_full.jsonl",
        help="Path to NQ dataset file"
    )
    parser.add_argument(
        "--story_file",
        type=str,
        default="/mnt/tongyan.zjy/data/story_room/sft/training_samples_test.jsonl",
        help="Path to Story Agent Thinking test file"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples per dataset (for quick testing)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = ComprehensiveEvaluator(
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    # 运行评估
    results = evaluator.run_comprehensive_evaluation(
        datasets=args.datasets,
        n_shot=args.n_shot,
        nq_file=args.nq_file,
        story_file=args.story_file,
        max_samples_per_dataset=args.max_samples
    )
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()
