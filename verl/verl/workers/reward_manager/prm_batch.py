# Copyright 2025 Individual Contributor: Mert Unsal
# Portions of this file are modifications by OPPO PersonalAI Team.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import torch

from verl import DataProto
from verl.workers.reward_manager import register


@register("prm_batch")
class ProcessRewardManager:
    """
    步骤级 reaward manager, 批量处理数据，并且接收来自 reward_fn 的分数计算函数和 reward tensor。
    """
    def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key="data_source", **reward_kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs

    def verify(self, data):
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        responses_str = []
        prompts_str = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            prompt_str = self.tokenizer.decode(prompt_ids[i], skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses_str.append(response_str)
            prompts_str.append(prompt_str)

        ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extras = data.non_tensor_batch.get("extra_info", [None] * len(data))

        scores = self.compute_score(
            data_sources=data_sources,
            prompt_strs=prompts_str,
            solution_strs=responses_str,
            ground_truths=ground_truths,
            extra_info=extras,
        )

        return scores

    def __call__(self, data: DataProto, return_dict=False):
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch[self.reward_fn_key]

        scores = self.verify(data)

        # 为每个样本按步骤累加 token 长度，定位 step 最后一个 token 的 idx
        for i, score in enumerate(scores):
            steps = score.get("steps", [])
            start_offset = 0
            for step in steps:
                step_str = step["step_str"]
                # 分词，不添加特殊 token
                step_tokens = self.tokenizer.encode(step_str, add_special_tokens=False)
                L = len(step_tokens)
                if L > 0:
                    # idx 相对于整个序列 = 已累加长度 + 本 step 长度 - 1
                    idx = start_offset + L - 1
                    reward_tensor[i, idx] = step["step_score"]
                start_offset += L

            # 如果还要保存 data_source 或其他额外信息
            data_source = data_sources[i]


        reward_extra_info["em"] = [score.get("em", 0) for score in scores]
        reward_extra_info["llm"] = [score.get("llm", 0) for score in scores]
        # reward_extra_info['steps'] = [score.get("steps", []) for score in scores]

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor
