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


from transformers.cache_utils import Cache

class MockQwen35Cache(Cache):
    def __init__(self, states, layer_types):
        self.layer_types = layer_types
        self.key_cache = [None] * len(states)
        self.value_cache = [None] * len(states)
        self.conv_states = [None] * len(states)
        self.recurrent_states = [None] * len(states)
        for i, state in enumerate(states):
            if isinstance(state, tuple):
                if len(state) == 2 and state[0].dim() == 3:
                    self.conv_states[i] = state[0]
                    self.recurrent_states[i] = state[1]
                else:
                    self.key_cache[i], self.value_cache[i] = state
            elif state.dim() == 5:
                self.key_cache[i] = state
                self.value_cache[i] = state
            else:
                self.recurrent_states[i] = state
    def __len__(self): return len(self.key_cache)
    def get_seq_length(self, layer_idx=0):
        # Find first attention layer to get seq len
        for k in self.key_cache:
            if k is not None: return k.shape[2]
        return 0
    def get_mask_sizes(self, query_length: int, layer_idx: int) -> Tuple[int, int]:
        return query_length + self.get_seq_length(layer_idx), 0
    @property
    def has_previous_state(self):
        return any(k is not None for k in self.key_cache)
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        return key_states, value_states

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
        # Clean up kwargs to avoid double-passing
        for key in ["input_ids", "attention_mask", "position_ids", "past_key_values", 
                    "inputs_embeds", "use_cache", "output_hidden_states", "return_dict"]:
            kwargs.pop(key, None)
            
        # Mapping for Qwen 3.5 TextModel.forward
        # During jit.trace, args has the interleaved structure from llm_export.py
        # It's now a FLAT list of states.
        if len(args) > 5:
            num_layers = self.language_model.config.num_hidden_layers
            layer_types = self.language_model.config.layer_types
            inputs_embeds = args[0]
            
            # Extract states based on layer types
            state_idx = 1
            past_key_values_list = []
            for ltype in layer_types:
                if "full_attention" in ltype:
                    past_key_values_list.append(args[state_idx])
                    state_idx += 1
                else:
                    # GatedDeltaNet has TWO states: conv and recurrent
                    past_key_values_list.append((args[state_idx], args[state_idx+1]))
                    state_idx += 2
            
            # Custom tensors
            custom_names = ["rope_rotary_cos_sin", "context_lengths", "last_token_ids", 
                            "kvcache_start_index", "position_ids", "attention_mask"]
            for i, name in enumerate(custom_names):
                offset = state_idx + i
                if len(args) > offset:
                    kwargs[name] = args[offset]
            
            # Wrap in Mock Cache
            past_key_values = MockQwen35Cache(past_key_values_list, layer_types)
                
            # Directly call forward to bypass any remaining decorators
            # We must use keyword arguments to be safe
            return self.language_model.forward(
                input_ids=None,
                attention_mask=kwargs.pop("attention_mask", None),
                position_ids=kwargs.pop("position_ids", None),
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=True,
                **kwargs
            )

        return self.language_model(*args, **kwargs)


class EdgeLLMHybridModelForCausalLM(nn.Module):
    """
    EdgeLLM Hybrid Model for Causal Language Modeling.
    """

    def __init__(self,
                 hf_model: nn.Module,
                 reduced_vocab_size: Optional[int] = None,
                 vocab_map: Optional[torch.Tensor] = None,
                 trt_native_ops: bool = False):
        super().__init__()
        self.config = hf_model.config
        self.torch_dtype = hf_model.dtype

        # Replace layers with TRT native versions if requested
        if trt_native_ops:
            from ..layers.layers import replace_decoder_layers_with_trt_native
            hf_model = replace_decoder_layers_with_trt_native(hf_model, trt_native_ops=True)

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

    @property
    def num_attention_layers(self):
        config = self.config
        if hasattr(config, "text_config"):
            config = config.text_config
        if hasattr(config, "layer_types"):
            return sum(1 for t in config.layer_types if "full_attention" in t)
        return getattr(config, "num_attention_layers", 0)

    @property
    def num_mamba_layers(self):
        config = self.config
        if hasattr(config, "text_config"):
            config = config.text_config
        if hasattr(config, "layer_types"):
            return sum(1 for t in config.layer_types if "linear_attention" in t or "mamba" in t)
        return getattr(config, "num_mamba_layers", 0)

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

        # Combine outputs in the expected order for llm_export.py
        final_outputs = [logits]
        
        # Handle hidden states if requested
        should_output_hidden_states = getattr(self, "output_hidden_states", False) or kwargs.get("output_hidden_states", False)
        if should_output_hidden_states:
            final_outputs.append(hidden_states)
            
        # Extract present states from Cache object
        if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
            cache = outputs.past_key_values
            # The order must match the layer_types order
            num_layers = self.model.language_model.config.num_hidden_layers
            layer_types = self.model.language_model.config.layer_types
            for i in range(num_layers):
                ltype = layer_types[i]
                if "full_attention" in ltype:
                    # key_cache and value_cache might be 4D or 5D. 
                    # If 5D, they are already stacked.
                    k = cache.key_cache[i]
                    v = cache.value_cache[i]
                    if k.dim() == 4:
                        final_outputs.append(torch.stack([k, v], dim=1))
                    else:
                        final_outputs.append(k) # Already stacked
                else:
                    # GatedDeltaNet has TWO states: conv and recurrent
                    final_outputs.append(cache.conv_states[i])
                    final_outputs.append(cache.recurrent_states[i])
        
        return tuple(final_outputs)


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

        # Determine total state tensors based on layer types
        is_qwen3_5 = getattr(self.model.config, "model_type", "") in ["qwen3_5", "qwen3_5_text", "qwen3_5_moe_text"]
        layer_types = getattr(self.model.config, "layer_types", [])
        
        # In Qwen 3.5, each layer has exactly ONE state (either 5D KV or 4D Recurrent)
        # For older Mamba, some might have 2. For now, assume 1:1 mapping for Qwen 3.5.
        num_layers = len(layer_types) if layer_types else self.config.num_hidden_layers
        
        if len(args) >= num_layers + 1:
            # Order: inputs_embeds, State0, State1, ..., StateN-1, rope, context, last, kv_start, pos, attn
            inputs_embeds = args[0]
            
            # Construct interleaved past_key_values for transformers
            # We'll use a custom list that contains both 5D and 4D states
            interleaved_states = []
            for i in range(num_layers):
                state = args[1 + i]
                if state.dim() == 5:
                    # Attention KV: [B, 2, H, S, D] -> Unstack to (K, V)
                    interleaved_states.append((state[:, 0, ...], state[:, 1, ...]))
                else:
                    # Recurrent state: [B, H, D, D] or [B, D, K]
                    interleaved_states.append(state)
            
            from transformers.cache_utils import DynamicCache
            # Note: We may need a custom Cache class if transformers doesn't like 4D tensors in DynamicCache
            past_key_values = interleaved_states 
            
            offset = 1 + num_layers
            rope_rotary_cos_sin = args[offset]
            context_lengths = args[offset + 1]
            last_token_ids = args[offset + 2]
            kvcache_start_index = args[offset + 3]
            attention_pos_id = args[offset + 4]
            attention_mask_input = args[offset + 5]
            
            # Convert to DynamicCache for transformer layers
            # Recurrent states will be handled separately by the model forward
            unstacked = [(t[:, 0, ...], t[:, 1, ...]) for t in past_key_values_list]
            from transformers.cache_utils import DynamicCache
            past_key_values = DynamicCache(tuple(unstacked))
            
            # Pack recurrent states if needed
            # (Assuming the model forward knows how to handle these as kwargs)
            kwargs["conv_states"] = conv_states
            kwargs["ssm_states"] = ssm_states
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
        # Thoroughly clean kwargs to avoid multiple values for the same argument
        # These are all explicitly passed as keyword arguments below
        for key in ["input_ids", "attention_mask", "position_ids", "past_key_values", 
                    "inputs_embeds", "use_cache", "output_hidden_states", "return_dict"]:
            kwargs.pop(key, None)
        
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask_input,
            position_ids=attention_pos_id,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
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
        
        if context_lengths is not None:
            logits = logits + 0 * context_lengths.view(-1)[0].to(logits.dtype)
            
        if kvcache_start_index is not None:
            logits = logits + 0 * kvcache_start_index.view(-1)[0].to(logits.dtype)
        
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
