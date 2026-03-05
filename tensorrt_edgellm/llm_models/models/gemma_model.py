# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Gemma 3 Model Implementation

This module provides the Gemma 3 model implementation for efficient
accelerated generation on edge devices.
"""

from typing import Optional, Tuple, Union

import torch
from torch import nn

from .llm_model import EdgeLLMModel, EdgeLLMModelForCausalLM


class EdgeGemma3Model(EdgeLLMModel):
    """
    Edge Gemma 3 Model.
    """
    def __init__(self,
                 hf_model: nn.Module,
                 is_eagle_base: bool = False,
                 use_prompt_tuning: bool = False) -> None:
        # If this is a multimodal wrapper (Gemma 3), extract the text model
        if hasattr(hf_model, "language_model"):
            hf_model = hf_model.language_model
        elif hasattr(hf_model, "text_model"):
            hf_model = hf_model.text_model
            
        super().__init__(hf_model, is_eagle_base, use_prompt_tuning)
        # Gemma 3 specific initialization if needed
        # For example, checking for specific RoPE or Norm settings
        pass

class EdgeGemma3ModelForCausalLM(EdgeLLMModelForCausalLM):
    """
    Edge Gemma 3 Model for Causal LM.
    """
    def __init__(self,
                 hf_model: nn.Module,
                 is_eagle_base: bool = False,
                 use_prompt_tuning: bool = False,
                 reduced_vocab_size: Optional[int] = None,
                 vocab_map: Optional[torch.Tensor] = None,
                 output_hidden_states: bool = False) -> None:
        super().__init__(hf_model, is_eagle_base, use_prompt_tuning, reduced_vocab_size, vocab_map, output_hidden_states=output_hidden_states)
        
        # Override the internal model with EdgeGemma3Model if we need specific forward logic
        self.model = EdgeGemma3Model(hf_model.model, is_eagle_base, use_prompt_tuning)
