# Copyright 2025 LlamaFactory team.
# Licensed under the Apache License, Version 2.0.

"""
Custom callback to trigger evaluation at the start of training.
"""

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from ..extras.logging import get_logger


logger = get_logger(__name__)


class EvalOnStartCallback(TrainerCallback):
    """
    A callback that triggers evaluation at the very beginning of training (before step 1).
    
    This is useful when you want to record baseline metrics before any training occurs.
    """
    
    def __init__(self):
        super().__init__()
        self.eval_done = False
    
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Trigger evaluation before the first training step.
        """
        # Only evaluate once at the very beginning
        if not self.eval_done and state.global_step == 0:
            logger.info("=" * 50)
            logger.info("Triggering initial evaluation before training starts...")
            logger.info("=" * 50)
            
            # Set the control flag to trigger evaluation
            control.should_evaluate = True
            self.eval_done = True
        
        return control

