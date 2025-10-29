# Copyright 2025 the LlamaFactory team.
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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .supervised import SupervisedDatasetProcessor


if TYPE_CHECKING:
    pass


logger = logging.get_logger(__name__)


@dataclass
class ThinkingSupervisedDatasetProcessor(SupervisedDatasetProcessor):
    """
    Processor for thinking mode training.
    Handles data with think and answer parts, and tracks answer start positions.
    """

    def _calculate_answer_start_position(
        self, 
        response_text: str, 
        labels: list[int],
        input_ids: list[int]
    ) -> int:
        """
        Calculate the start position of the answer part in the tokenized sequence.
        
        Args:
            response_text: The response text containing "<think>...</think>\nanswer"
            labels: The label sequence where IGNORE_INDEX marks prompt parts
            input_ids: The input token sequence
            
        Returns:
            The position where answer starts (in the label sequence)
        """
        # Find the separator in the response text
        separator = self.data_args.thinking_separator
        
        # If separator not found, assume the whole response is the answer
        if separator not in response_text:
            logger.warning_rank0(
                f"Thinking separator '{separator}' not found in response. "
                "Treating entire response as answer."
            )
            # Find the first non-IGNORE_INDEX position in labels
            for i, label in enumerate(labels):
                if label != IGNORE_INDEX:
                    return i
            return 0
        
        # Split by separator to get think and answer parts
        parts = response_text.split(separator, 1)
        if len(parts) != 2:
            logger.warning_rank0(
                f"Failed to split response by separator. Using fallback."
            )
            for i, label in enumerate(labels):
                if label != IGNORE_INDEX:
                    return i
            return 0
        
        think_part = parts[0] + separator  # Include separator in think part
        
        # Tokenize the think part to get its length
        # We need to find where it ends in the input_ids
        # But we need to be careful about special tokens and template formatting
        
        # Find the start of the response (first non-IGNORE_INDEX in labels)
        response_start = 0
        for i, label in enumerate(labels):
            if label != IGNORE_INDEX:
                response_start = i
                break
        
        # Tokenize the think part alone to estimate its length
        think_tokens = self.tokenizer.encode(think_part, add_special_tokens=False)
        
        # The answer should start approximately after think_tokens from response_start
        answer_start = response_start + len(think_tokens)
        
        # Make sure we don't go out of bounds
        answer_start = min(answer_start, len(labels) - 1)
        
        return answer_start

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """
        Preprocess dataset for thinking mode training.
        Adds answer_start_positions to track where answer begins.
        """
        model_inputs = defaultdict(list)
        
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            # Get the response text
            response_text = examples["_response"][i][0]["content"]
            
            # Encode the example
            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            
            # Calculate answer start position
            answer_start_pos = self._calculate_answer_start_position(
                response_text, labels, input_ids
            )
            
            # Add to model inputs
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["answer_start_positions"].append(answer_start_pos)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        """Print a data example with answer position highlighted."""
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        answer_start = example.get("answer_start_positions", 0)
        
        print("=" * 80)
        print("THINKING MODE DATA EXAMPLE")
        print("=" * 80)
        print("input_ids:\n{}".format(example["input_ids"]))
        print("\ninputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("\nlabel_ids:\n{}".format(example["labels"]))
        print(f"\nlabels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")
        print(f"\nanswer_start_position: {answer_start}")
        
        # Try to show where answer starts
        if answer_start < len(example["labels"]):
            answer_labels = [label for label in example["labels"][answer_start:] if label != IGNORE_INDEX]
            if answer_labels:
                print(f"\nanswer part (from position {answer_start}):\n{self.tokenizer.decode(answer_labels, skip_special_tokens=False)}")
        print("=" * 80)

