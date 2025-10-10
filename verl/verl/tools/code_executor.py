#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The OPPO Inc. Personal AI team. All rights reserved.
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


import json
import logging
import os
import ray
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

from verl.tools.utils.code_executors.nsjail_sandbox import NsjailSandbox
from verl.tools.utils.code_executors.utils import truncate_content
from concurrent.futures import ProcessPoolExecutor
import resource
import ray
from multiprocessing import Process, Queue
import signal
import time
import sympy

# logger config
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("CODEEXEC_LOGGING_LEVEL", "INFO"))  # Changed from WARN to INFO to see our debug logs

class CodeExecutor(BaseTool):
    """Run the written code in nsjail sandbox env and return the result"""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize CodeExecutor and config"""
        super().__init__(config, tool_schema)

        # state save
        self._instance_dict = {}    
        
        # check execution config
        if "timeout" not in config:
            raise ValueError(f"[Error] Lack param 'timeout' in code_executor.yaml")
        if "memory_limit" not in config:
            raise ValueError(f"[Error] Lack param 'memory_limit' in code_executor.yaml")
        if "nsjail_path" not in config:
            raise ValueError(f"[Error] Lack param 'nsjail_path' in code_executor.yaml")
        if "max_obs_length" not in config:
            raise ValueError(f"[Error] Lack param 'max_obs_length' in code_executor.yaml")
        
        self.timeout = int(config["timeout"])
        self.memory_limit = int(sympy.sympify(config["memory_limit"])) 
        self.nsjail_path = config["nsjail_path"]
        self.max_obs_length = int(config["max_obs_length"]) * 3     # token-level length limit
        
        # print config
        logger.info(f"Initialized CodeExecutor with config: {config}")
        
        # Check if nsjail path exists
        if not os.path.exists(self.nsjail_path):
            logger.error(f"[CodeExecutor] nsjail path does not exist: {self.nsjail_path}")
        else:
            logger.info(f"[CodeExecutor] nsjail path verified: {self.nsjail_path}")
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""

        if self.tool_schema:
            return self.tool_schema
            
        return OpenAIFunctionToolSchema(
            type="function",
            function={
                "name": "code",
                "description": "Execute the code blob in nsjail sandbox and return corresponding results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code_blob": {
                            "type": "string",
                            "description": "The code blob written by LLMs"
                        },
                    },
                    "required": ["code_blob"]
                }
            }
        )
    
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Create a tool instance."""
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": 0,
            "results": []
        }
        return instance_id
    
    def _worker(self, q, query):
        try:
            logger.info(f"[CodeExecutor._worker] Worker process started")
            
            # Set memory limitation for the execution subprocess
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            logger.info(f"[CodeExecutor._worker] Setting memory limit: {self.memory_limit} bytes")
            resource.setrlimit(resource.RLIMIT_AS, (self.memory_limit, hard))
            
            logger.info(f"[CodeExecutor._worker] Initializing nsjail sandbox with path: {self.nsjail_path}")
            nsjail_sandbox = NsjailSandbox(self.nsjail_path)
            
            logger.info(f"[CodeExecutor._worker] Parsing code blob from query")
            _, code_blob, stdin_str = nsjail_sandbox.parse_code_blobs_stdin_answer(query)
            logger.info(f"[CodeExecutor._worker] Parsed code blob (first 100 chars): {code_blob[:100]}...")
            
            logger.info(f"[CodeExecutor._worker] Executing code in sandbox")
            succ, output = nsjail_sandbox.exec(code_blob, stdin_str)
            logger.info(f"[CodeExecutor._worker] Execution completed, success: {succ}")
            
            q.put((succ, output))
            logger.info(f"[CodeExecutor._worker] Result put in queue")
            
        except MemoryError:
            logger.error(f"[CodeExecutor._worker] Memory limit exceeded")
            q.put((False, "MEMORY_LIMIT_EXCEEDED"))
        except Exception as e:
            logger.error(f"[CodeExecutor._worker] Exception occurred: {str(e)}")
            q.put((False, f"ERROR: {str(e)}"))

    async def execute(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """Execute the code_executor tool.
        
        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing code_blob
            
        Returns:
            Tuple of (tool_response, tool_reward_score, tool_metrics)
        """
        
        logger.info(f"[CodeExecutor] Starting execution for instance {instance_id}")
        logger.info(f"[CodeExecutor] Parameters: {parameters}")
        
        # Try both parameter names for compatibility
        # query = parameters.get("query") or parameters.get("code_blob")
        query = parameters.get("query")
        if not query:
            error_msg = f"[CodeExecutor] No code found in parameters: {parameters}"
            logger.error(error_msg)
            return error_msg, 0.0, {"error": "missing_code"}

        logger.info(f"[CodeExecutor] Code to execute (first 200 chars): {query[:200]}...")

        # Check delimiters
        # if not query.startswith("```py\n") and not query.endswith("\n```"):
        #     query = f"```py\n{query}\n```"
  
        # Init queue
        logger.info(f"[CodeExecutor] Creating subprocess for execution")
        q = Queue()
        p = Process(target=self._worker, args=(q, query))
        p.start()
        logger.info(f"[CodeExecutor] Subprocess started, PID: {p.pid}")
        
        # Set timeout for the execution subprocess and start it
        logger.info(f"[CodeExecutor] Waiting for subprocess completion (timeout: {self.timeout}s)")
        p.join(timeout=self.timeout)
        
        result = None
        if p.is_alive():
            logger.warning(f"[CodeExecutor] Subprocess timed out after {self.timeout}s, terminating")
            p.terminate()  # Timeout, kill the subprocess immediately
            result = (False, "PROCESS_TIMEOUT")
        if not result:
            if not q.empty():
                result = q.get()
                logger.info(f"[CodeExecutor] Got result from subprocess: success={result[0]}")
            else:
                logger.error(f"[CodeExecutor] No result from subprocess")
                result = (False, "NO_RESULT")
        
        # Get result and cut obs length
        succ, output = result
        output = truncate_content(output, self.max_obs_length)
        
        logger.info(f"[CodeExecutor] Execution completed for instance {instance_id}, success: {succ}")
        logger.info(f"[CodeExecutor] Output (first 200 chars): {output[:200]}...")

        # TODO: prm reward and extra_info
        prm_reward = 0.0
        metrics = {}

        return output, prm_reward, metrics

    
    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate reward for the tool instance."""
        return 0.0 # TODO: self._instance_dict[instance_id]["reward"]
    
    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]