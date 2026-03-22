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
"""EdgeLLM language model wrappers."""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .. import model_utils


class EdgeLLMModel(nn.Module):
    """
    Base wrapper for EdgeLLM models.
    """

    def __init__(self, language_model: nn.Module, is_eagle_base: bool = False):
        super().__init__()
        self.language_model = language_model
        self.is_eagle_base = is_eagle_base

    def forward(self, *args, **kwargs) -> Union[Tuple, Dict[str, torch.Tensor]]:
        """Standard forward pass for EdgeLLM models."""
        return self.language_model(*args, **kwargs)


class EdgeLLMHybridModel(nn.Module):
    """
    Wrapper for hybrid models (e.g., Mamba + Attention).
    """

    def __init__(self, language_model: nn.Module):
        super().__init__()
        self.language_model = language_model

    def forward(self, *args, **kwargs) -> Union[Tuple, Dict[str, torch.Tensor]]:
        """Forward pass for hybrid models."""
        return self.language_model(*args, **kwargs)


class EdgeLLMHybridModelForCausalLM(nn.Module):
    """
    EdgeLLM Hybrid Model for Causal Language Modeling.
    """

    def __init__(self,
                 hf_model: nn.Module,
                 reduced_vocab_size: Optional[int] = None,
                 vocab_map: Optional[torch.Tensor] = None):
        super().__init__()
        self.config = hf_model.config
        self.torch_dtype = hf_model.dtype

        # Extract language model from hf_model if nested (e.g. Qwen2MambaForCausalLM)
        language_model = getattr(hf_model, "model", hf_model)
        self.model = EdgeLLMHybridModel(language_model)

        # Handle lm_head with optional vocabulary reduction
        lm_head = hf_model.lm_head
        if reduced_vocab_size is not None and vocab_map is not None:
            # Create a new lm_head with reduced vocabulary
            # Note: vocab_map should be a tensor of indices to keep
            new_lm_head = nn.Linear(lm_head.in_features,
                                    reduced_vocab_size,
                                    bias=False)
            with torch.no_grad():
                new_lm_head.weight.copy_(lm_head.weight[vocab_map])
            self.lm_head = new_lm_head
        else:
            self.lm_head = lm_head

    def forward(self, *args, **kwargs) -> Union[Tuple, Dict[str, torch.Tensor]]:
        """Forward pass for hybrid causal LM."""
        outputs = self.model(*args, **kwargs)

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        if kwargs.get("return_dict"):
            from transformers.modeling_outputs import \
                CausalLMOutputWithPast
            return CausalLMOutputWithPast(
                logits=logits,
                hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
                attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
            )

        return (logits, ) + outputs[1:]


class EdgeLLMModelForCausalLM(nn.Module):
    """
    EdgeLLM Model for Causal Language Modeling.
    """

    def __init__(self,
                 hf_model: nn.Module,
                 is_eagle_base: bool = False,
                 reduced_vocab_size: Optional[int] = None,
                 vocab_map: Optional[torch.Tensor] = None,
                 output_hidden_states: bool = False) -> None:
        """
        Initialize the EdgeLLM model for causal LM.
        """
        super().__init__()

        language_model, config = model_utils.prepare_language_model_and_config(
            hf_model)
        self.torch_dtype = hf_model.dtype
        self.config = config
        self.output_hidden_states = output_hidden_states
        self.embed_tokens = language_model.embed_tokens.to(self.torch_dtype)

        # Create EdgeLLMModel with the original model
        self.model = EdgeLLMModel(language_model, is_eagle_base)

        # Handle lm_head with optional vocabulary reduction
        lm_head = hf_model.lm_head
        if reduced_vocab_size is not None and vocab_map is not None:
            new_lm_head = nn.Linear(lm_head.in_features,
                                    reduced_vocab_size,
                                    bias=False)
            with torch.no_grad():
                new_lm_head.weight.copy_(lm_head.weight[vocab_map])
            self.lm_head = new_lm_head
        else:
            self.lm_head = lm_head

    def forward(self, *args, **kwargs) -> Union[Tuple, Dict[str, torch.Tensor]]:
        """
        Forward pass for causal LM. 
        Robustly handles both standard calls and ONNX export tracing.
        Order matches llm_export.py dummy inputs.
        """
        num_layers = self.config.num_hidden_layers
        
        # Default values
        inputs_embeds = None
        past_key_values_list = None
        rope_rotary_cos_sin = None
        context_lengths = None
        last_token_ids = None
        kvcache_start_index = None
        attention_pos_id = None
        attention_mask_input = None

        # Determine if this is a tracing call or a standard call
        if len(args) >= num_layers + 1:
            # Order: inputs_embeds, KV0, KV1, ..., KVn-1, rope, context, last, kv_start, pos, attn
            inputs_embeds = args[0]
            
            past_key_values_list = []
            for i in range(num_layers):
                past_key_values_list.append(args[1 + i])
            
            offset = 1 + num_layers
            rope_rotary_cos_sin = args[offset]
            context_lengths = args[offset + 1]
            last_token_ids = args[offset + 2]
            kvcache_start_index = args[offset + 3]
            attention_pos_id = args[offset + 4]
            attention_mask_input = args[offset + 5]
            
            # Convert to DynamicCache
            # In transformers v5, DynamicCache expects separate K and V tensors
            unstacked = [(t[:, 0, ...], t[:, 1, ...]) for t in past_key_values_list]
            from transformers.cache_utils import DynamicCache
            past_key_values = DynamicCache(tuple(unstacked))
        else:
            # Standard call
            inputs_embeds = kwargs.get("inputs_embeds")
            past_key_values = kwargs.get("past_key_values")
            attention_pos_id = kwargs.get("attention_pos_id", kwargs.get("position_ids"))
            attention_mask_input = kwargs.get("attention_mask_input", kwargs.get("attention_mask"))
            
            if len(args) > 0: inputs_embeds = args[0]
            if len(args) > 1: past_key_values = args[1]
            
            input_ids = kwargs.get("input_ids")
            if inputs_embeds is None and input_ids is not None:
                inputs_embeds = self.embed_tokens(input_ids)

        should_output_hidden_states = self.output_hidden_states or kwargs.get("output_hidden_states", False)

        # Call underlying model using keyword arguments
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask_input,
            position_ids=attention_pos_id,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_hidden_states=should_output_hidden_states,
            return_dict=True,
            rope_rotary_cos_sin=rope_rotary_cos_sin, # Pass RoPE tensor!
            context_lengths=context_lengths,
            kvcache_start_index=kvcache_start_index,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        
        # Add a dummy operation using last_token_ids to keep it in the graph
        if last_token_ids is not None:
            # We must use it in a way that doesn't change the output but forces inclusion
            logits = logits + 0 * last_token_ids.view(-1)[0].to(logits.dtype)
        
        # Combine outputs in the expected order for llm_export.py
        final_outputs = [logits]
        
        if should_output_hidden_states:
            # Clone to ensure it's a distinct node in the ONNX graph
            final_outputs.append(hidden_states.clone())
        
        if outputs.past_key_values is not None:
            # DynamicCache is iterable in transformers v5
            for kv_tuple in outputs.past_key_values:
                k, v = kv_tuple[0], kv_tuple[1]
                # Stack K and V along dimension 1 to match expected 5D present_key_values
                final_outputs.append(torch.stack([k, v], dim=1))
        
        return tuple(final_outputs)
