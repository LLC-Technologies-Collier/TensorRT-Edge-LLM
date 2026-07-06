# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# SPDX-FileCopyrightText: Copyright 2026 Google LLC and contributors
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import torch
import torch.nn as nn

from ...config import ModelConfig
from ..default.modeling_default import (MLP, Attention, CausalLM, RMSNorm,
                                        Transformer)
from ..qwen3_moe.modeling_qwen3_moe import Qwen3SparseMoeBlock


class Gemma4Attention(Attention):

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        gemma4_layer_types = getattr(config, "gemma4_layer_types", None)
        layer_type = gemma4_layer_types[
            layer_idx] if gemma4_layer_types else "sliding_attention"

        orig_num_kv_heads = config.num_key_value_heads
        orig_num_attn_heads = config.num_attention_heads
        orig_head_dim = config.head_dim

        if layer_type == "full_attention":
            config.num_attention_heads = orig_num_attn_heads
            config.head_dim = getattr(config, "global_head_dim", 512) or 512
            config.num_key_value_heads = getattr(config,
                                                 "num_global_key_value_heads",
                                                 2)
        else:
            config.num_attention_heads = orig_num_attn_heads
            config.head_dim = orig_head_dim
            config.num_key_value_heads = orig_num_kv_heads

        super().__init__(config, layer_idx=layer_idx)

        if getattr(config, "attention_k_eq_v", False):
            self.v_proj = self.k_proj

        config.num_key_value_heads = orig_num_kv_heads
        config.num_attention_heads = orig_num_attn_heads
        config.head_dim = orig_head_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        attention_mask: "torch.Tensor | None" = None,
        attention_pos_id: "torch.Tensor | None" = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        head_dim = int(512) if int(self.head_dim) == 512 else int(256)
        num_kv_heads = int(2) if int(self.num_kv_heads) == 2 else int(8)
        num_heads = int(16)

        if self.q_norm is not None:
            query_states = self.q_norm(
                query_states.reshape(-1, hidden_states.size(1), num_heads,
                                     head_dim)).reshape(
                                         -1, hidden_states.size(1),
                                         num_heads * head_dim)

        if self.k_norm is not None:
            key_states = self.k_norm(
                key_states.reshape(-1, hidden_states.size(1), num_kv_heads,
                                   head_dim)).reshape(-1,
                                                      hidden_states.size(1),
                                                      num_kv_heads * head_dim)

        k_reshaped = key_states.reshape(-1, hidden_states.size(1),
                                        num_kv_heads,
                                        head_dim).transpose(1, 2)
        v_reshaped = value_states.reshape(-1, hidden_states.size(1),
                                          num_kv_heads,
                                          head_dim).transpose(1, 2)

        # Update KV cache cleanly using TensorScatter mapping
        k_cache = past_key_value[:, 0]
        v_cache = past_key_value[:, 1]

        # Concatenate the new keys and values with the past KV cache
        # to ensure the attention block topologically connects to the cache.
        k_full = torch.cat([k_cache, k_reshaped], dim=2)
        v_full = torch.cat([v_cache, v_reshaped], dim=2)
        present_key_value = torch.stack([k_full, v_full], dim=1)

        q = query_states.reshape(-1, hidden_states.size(1), num_heads,
                                 head_dim).transpose(1, 2)

        if num_heads != num_kv_heads:
            k_full = k_full.repeat_interleave(num_heads // num_kv_heads, dim=1)
            v_full = v_full.repeat_interleave(num_heads // num_kv_heads, dim=1)

        attn_weights = torch.matmul(q, k_full.transpose(-1, -2)) / (head_dim**
                                                                    0.5)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_probs = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_probs, v_full)
        attn_output = attn_output.transpose(1,
                                            2).reshape(-1,
                                                       hidden_states.size(1),
                                                       num_heads * head_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output, present_key_value


class Gemma4DecoderLayer(nn.Module):

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        gemma4_layer_types = getattr(config, "gemma4_layer_types", None)
        layer_type = gemma4_layer_types[
            layer_idx] if gemma4_layer_types else "sliding_attention"

        self.self_attn = Gemma4Attention(config, layer_idx=layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       config.rms_norm_eps,
                                       config=config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                config.rms_norm_eps,
                                                config=config)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size,
                                                 config.rms_norm_eps,
                                                 config=config)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size,
                                                  config.rms_norm_eps,
                                                  config=config)

        if layer_type == "full_attention":
            self.mlp = MLP(config, layer_idx=layer_idx)
        else:
            self.moe = Qwen3SparseMoeBlock(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        attention_mask: "torch.Tensor | None" = None,
        attention_pos_id: "torch.Tensor | None" = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        is_moe = hasattr(self, "moe")

        residual = hidden_states
        attn_output, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            past_key_value,
            rope_rotary_cos_sin,
            context_lengths,
            kvcache_start_index,
            attention_mask=attention_mask,
            attention_pos_id=attention_pos_id,
        )
        hidden_states = residual + self.post_attention_layernorm(attn_output)

        residual = hidden_states
        if is_moe:
            hidden_states = residual + self.post_feedforward_layernorm(
                self.moe(self.pre_feedforward_layernorm(hidden_states)))
        else:
            hidden_states = residual + self.post_feedforward_layernorm(
                self.mlp(self.pre_feedforward_layernorm(hidden_states)))

        return hidden_states, present_key_value


class Gemma4Transformer(Transformer):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.layers = nn.ModuleList([
            Gemma4DecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])


class Gemma4CausalLM(CausalLM):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.model = Gemma4Transformer(config)
