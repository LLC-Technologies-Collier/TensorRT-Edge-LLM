# Copyright 2026 Google LLC
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

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
from ..model_utils import get_config_attr
from ...onnx_export.onnx_utils import (kv_cache_update_onnx, rope_onnx,
                                      attention_onnx, ONNX_OPSET_VERSION)


class EdgeLLMAttentionTRTNative(nn.Module):
    """
    TensorRT native operations attention computation module.
    
    This module encapsulates the attention computation using TensorRT's
    native operations, optimized for export to ONNX and later TensorRT engines.
    
    Attributes:
        qkv_proj: Combined Q, K, V projection layer
        o_proj: Output projection layer
        qk_norm: QK normalization layer
        hidden_size: Hidden dimension size
        num_key_value_heads: Number of key-value heads
        num_attention_heads: Number of attention heads
        head_dim: Dimension of each attention head
        max_position_embeddings: Maximum sequence length for positional embeddings
        qk_scale: Scaling factor for Q@K^T
    """

    def __init__(self, attention_module: nn.Module,
                 eagle3_draft: bool) -> None:
        """
        Initialize the EdgeLLMAttentionTRTNative module.
        
        Args:
            attention_module: Original attention module to extract components from
        """
        super().__init__()
        from .layer_utils import EdgeLLMQKVProj, EdgeLLMQKNorm
        try:
            self.torch_dtype = next(attention_module.parameters()).dtype
        except StopIteration:
            self.torch_dtype = torch.float16

        # Copy projection layers from original attention module
        self.qkv_proj = EdgeLLMQKVProj(attention_module, eagle3_draft)
        self.o_proj = getattr(attention_module, "o_proj", 
                             getattr(attention_module, "out_proj", None))
        if self.o_proj is None:
            raise AttributeError(f"Could not find output projection in {type(attention_module)}")
            
        # Qwen3.5 MoE specific layers
        self.z_proj = getattr(attention_module, "in_proj_z", None)
        self.norm = getattr(attention_module, "norm", None)

        self.qk_norm = EdgeLLMQKNorm(attention_module)
        self.eagle3_draft = eagle3_draft

        # Robust attribute discovery for Qwen variants
        self.hidden_size = get_config_attr(attention_module, "hidden_size", 3072)
        
        # Standard vs Linear Attention head mapping
        self.num_attention_heads = get_config_attr(attention_module, "num_attention_heads", 
                                                 getattr(attention_module, "num_v_heads", 32))
        self.num_key_value_heads = get_config_attr(attention_module, "num_key_value_heads",
                                                 getattr(attention_module, "num_k_heads", 2))
        
        # Head Dim fallback
        self.head_dim = get_config_attr(attention_module, "head_dim",
                                      getattr(attention_module, "head_v_dim", 
                                              self.hidden_size // self.num_attention_heads if self.num_attention_heads > 0 else 128))
        
        self.max_position_embeddings = get_config_attr(attention_module, "max_position_embeddings", 32768)

        assert self.num_attention_heads % self.num_key_value_heads == 0, \
            f"num_attention_heads ({self.num_attention_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})"
        self.num_key_value_groups: int = self.num_attention_heads // self.num_key_value_heads

        # Compute QK scale factor
        self.qk_scale: float = 1.0 / (self.head_dim**0.5)

    def _get_expected_output_dim(self) -> int:
        """
        Determines the expected input dimension for the output projection layer.
        This ensures meta tensors and computed tensors remain consistent.
        """
        if hasattr(self.o_proj, "in_features"):
            return self.o_proj.in_features
        # Fallback for 122B Gated Delta Attention (32 heads @ 512 reduced to 16 groups @ 512 = 8192)
        if self.num_attention_heads == 32 and self.head_dim == 512:
            return 8192
        return self.hidden_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for TensorRT native operations attention computation.
        """
        bsz, q_len, _ = hidden_states.size()
        
        # Handle meta tensors during jit.trace for offloaded modules
        if hidden_states.device.type == 'meta':
            out_dim = self._get_expected_output_dim()
            attn_output = torch.empty(bsz, q_len, out_dim, device='meta', dtype=self.torch_dtype)
            return attn_output, k_cache, v_cache

        # Apply Q, K, V projections
        query_states, key_states, value_states = self.qkv_proj(hidden_states)

        norm_shape = [bsz, q_len, -1, self.head_dim]
        query_states, key_states = self.qk_norm(query_states, key_states,
                                                norm_shape)

        # Convert to FP16 for TensorRT compatibility
        compute_type = torch.float16
        io_type = torch.float16
        query_states = query_states.to(io_type)
        key_states = key_states.to(io_type)
        value_states = value_states.to(io_type)

        if k_cache.dtype != io_type:
            k_cache = k_cache.to(io_type)
        if v_cache.dtype != io_type:
            v_cache = v_cache.to(io_type)

        # Reshape Q, K, V
        q_heads = query_states.shape[-1] // self.head_dim
        kv_heads = key_states.shape[-1] // self.head_dim
        
        query_states = query_states.view(bsz, q_len, q_heads,
                                         self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, kv_heads,
                                     self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, kv_heads,
                                         self.head_dim).transpose(1, 2)

        half_dim = self.head_dim // 2
        rope_cos = rope_rotary_cos_sin[:, :, :half_dim]
        rope_sin = rope_rotary_cos_sin[:, :, half_dim:]
        rope_cos = rope_cos[0:1, :, :]
        rope_sin = rope_sin[0:1, :, :]
        rope_cos = rope_cos.reshape(-1, half_dim)
        rope_sin = rope_sin.reshape(-1, half_dim)
        rope_cos = rope_cos.to(compute_type)
        rope_sin = rope_sin.to(compute_type)

        if position_ids is None:
            position_ids = kvcache_start_index.unsqueeze(1) + torch.arange(
                q_len,
                device=kvcache_start_index.device,
                dtype=kvcache_start_index.dtype).unsqueeze(0)

        query_states = self._apply_rope(query_states.to(compute_type), rope_cos, rope_sin, position_ids)
        key_states = self._apply_rope(key_states.to(compute_type), rope_cos, rope_sin, position_ids)
        
        query_states = query_states.to(io_type)
        key_states = key_states.to(io_type)
        value_states = value_states.to(io_type)

        present_k_cache = kv_cache_update_onnx(k_cache, key_states, kvcache_start_index)
        present_v_cache = kv_cache_update_onnx(v_cache, value_states, kvcache_start_index)

        present_length = torch.max(kvcache_start_index) + torch.max(context_lengths)
        k_present = present_k_cache[:, :, :present_length, :]
        v_present = present_v_cache[:, :, :present_length, :]

        query_states = query_states * self.qk_scale
        attn_output = self._compute_attention(query_states, k_present, v_present, attention_mask)

        # Handle Gated Delta Attention (Qwen3.5 MoE)
        if self.z_proj is not None:
            z_gate = self.z_proj(hidden_states)
            if z_gate.device.type != 'meta' and attn_output.device.type != 'meta':
                z_gate = z_gate.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
                attn_output = attn_output * z_gate

        if self.norm is not None:
            is_materialized = True
            if hasattr(self.norm, 'weight') and self.norm.weight.device.type == 'meta':
                is_materialized = False
            if self.z_proj is not None and z_gate.device.type == 'meta':
                is_materialized = False

            if is_materialized:
                attn_output = attn_output.transpose(1, 2).reshape(-1, self.head_dim)
                if self.z_proj is not None:
                    z_gate_flat = z_gate.transpose(1, 2).reshape(-1, self.head_dim)
                    attn_output = self.norm(attn_output, z_gate_flat)
                else:
                    attn_output = self.norm(attn_output)
                attn_output = attn_output.view(bsz, q_len, self.num_attention_heads * self.head_dim)
            else:
                attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)

            # Reductions for GatedDeltaNet
            expected_in = self._get_expected_output_dim()
            if self.num_attention_heads > self.num_key_value_heads and attn_output.shape[-1] > expected_in:
                attn_output = attn_output.view(bsz, q_len, self.num_key_value_heads, -1, self.head_dim)
                attn_output = attn_output.sum(dim=3).view(bsz, q_len, -1)
        else:
            # Standard attention path
            if self.num_attention_heads > self.num_key_value_heads:
                attn_output = attn_output.view(bsz, self.num_key_value_heads, -1, q_len, self.head_dim)
                attn_output = attn_output.sum(dim=2)
            
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)

        # Apply output projection
        attn_output = attn_output.to(self.torch_dtype)
        attn_output = self.o_proj(attn_output)

        # Ensure final output hidden dimension matches self.hidden_size
        if attn_output.shape[-1] != self.hidden_size:
             pad_size = self.hidden_size - attn_output.shape[-1]
             if pad_size > 0:
                 attn_output = torch.nn.functional.pad(attn_output, (0, pad_size))
             else:
                 attn_output = attn_output[..., :self.hidden_size]

        return attn_output, present_k_cache, present_v_cache

    def _apply_rope(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary position embeddings to input tensor."""
        return rope_onnx(x, cos, sin, position_ids)

    def _compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute scaled dot-product attention using TensorRT custom op."""
        is_causal = False
        if mask is None:
            batch_size, num_heads, seq_q, head_dim = query.shape
            seq_k = key.shape[2]
            row_indices = torch.arange(seq_q, device=query.device, dtype=torch.int32).reshape(seq_q, 1)
            col_indices = torch.arange(seq_k, device=query.device, dtype=torch.int32).reshape(1, seq_k)
            offset = seq_k - seq_q
            causal_mask = col_indices > (row_indices + offset)
            mask = torch.where(
                causal_mask,
                torch.tensor(float('-inf'), device=query.device, dtype=query.dtype),
                torch.tensor(0.0, device=query.device, dtype=query.dtype))
            mask = mask.reshape(1, 1, seq_q, seq_k)

        return attention_onnx(query, key, value, attn_mask=mask, is_causal=is_causal, scale=1.0)


def register_trt_native_attention_onnx_symbolic_functions() -> None:
    """Register symbolic functions for ONNX export of TensorRT native operations."""
    from torch.onnx import register_custom_op_symbolic
    register_custom_op_symbolic("trt::kv_cache_update_onnx", symbolic_kv_cache_update, ONNX_OPSET_VERSION)
    register_custom_op_symbolic("trt::rope_onnx", symbolic_rope, ONNX_OPSET_VERSION)
    register_custom_op_symbolic("trt::attention_onnx", symbolic_attention, ONNX_OPSET_VERSION)
    print("Registered ONNX symbolic functions for TensorRT native kv cache update, rope, and attention")
