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
        self.config = getattr(attention_module, "config", None)
        self.hidden_size = get_config_attr(attention_module, "hidden_size", 3072)
        
        # Standard vs Linear Attention head mapping
        self.num_attention_heads = get_config_attr(attention_module, "num_attention_heads", 
                                                 getattr(self.config, "num_attention_heads", 32))
        self.num_key_value_heads = get_config_attr(attention_module, "num_key_value_heads",
                                                 getattr(self.config, "num_key_value_heads", 2))
        
        # Head Dim MUST exactly match what o_proj expects to avoid broken ONNX graphs
        # For Qwen 3.5 MoE Standard Attention, in_features is 8192, num_heads is 32 -> head_dim = 256
        if hasattr(self.o_proj, "in_features") and self.num_attention_heads > 0:
            self.head_dim = self.o_proj.in_features // self.num_attention_heads
        else:
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
        Determines the expected output dimension for the attention layer.
        This ensures meta tensors and computed tensors remain consistent.
        """
        if hasattr(self.o_proj, "out_features"):
            return self.o_proj.out_features
        # Fallback
        return self.hidden_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for TensorRT native operations attention computation.
        """
        bsz, q_len, _ = hidden_states.size()
        
        # Apply Q, K, V projections
        query_states, key_states, value_states = self.qkv_proj(hidden_states)

        # Dynamic head detection for normalization
        # This handles unified 64-head inputs by allowing the norm block to see the actual size
        current_q_heads = query_states.shape[-1] // self.head_dim
        current_k_heads = key_states.shape[-1] // self.head_dim
        
        q_norm_shape = [bsz, q_len, current_q_heads, self.head_dim]
        k_norm_shape = [bsz, q_len, current_k_heads, self.head_dim]
        
        query_states, key_states = self.qk_norm(query_states, key_states,
                                                q_norm_shape, k_norm_shape)

        # Convert to FP16 for TensorRT compatibility
        compute_type = torch.float16
        io_type = torch.float16
        query_states = query_states.to(io_type)
        key_states = key_states.to(io_type)
        value_states = value_states.to(io_type)

        if kv_cache.dtype != io_type:
            kv_cache = kv_cache.to(io_type)

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

        # Note: In the fused AttentionPlugin, RoPE and KV cache update are handled internally.
        # But for tracing, we still need to pass the inputs.
        
        query_states = query_states * self.qk_scale
        
        # Qwen 3.5 hybrid models use a unified super-shape for KV cache (64 heads, 256 dim)
        # to satisfy the TRT 10.x parser. We must slice to the layer's actual heads here.
        if kv_cache.shape[2] > self.num_key_value_heads:
            kv_cache = kv_cache[:, :, :self.num_key_value_heads, :, :]
        if kv_cache.shape[4] > self.head_dim:
            kv_cache = kv_cache[..., :self.head_dim]

        # Use fake_attention custom op for tracing
        # We pass the original tensors and let the surgeon handle broadcasting
        attn_output, present_kv_cache = fake_attention(
            query_states, key_states, value_states, 
            kv_cache,
            attention_mask,
            q_heads, kv_heads, self.head_dim
        )
        
        # Add contributions to trace for all unused meta tensors
        if attn_output.device.type == 'meta':
            dummy = (context_lengths.sum() + rope_rotary_cos_sin.sum() + kvcache_start_index.sum()) * 0
            attn_output = attn_output + dummy

        # Ensure attn_output is float32 for any subsequent norms/math
        attn_output = attn_output.to(torch.float32)

        # Handle Gated Delta Attention (Qwen3.5 MoE)
        if self.z_proj is not None:
            z_gate = self.z_proj(hidden_states).to(torch.float32)
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

            # Reductions for GatedDeltaNet - ONLY if dimension exceeds expected
            expected_out_dim = self._get_expected_output_dim()
            if attn_output.shape[-1] > expected_out_dim:
                # This handles cases where num_attention_heads * head_dim != hidden_size
                # by reducing along the head dimension groups
                num_groups = attn_output.shape[-1] // expected_out_dim
                attn_output = attn_output.view(bsz, q_len, expected_out_dim, num_groups).sum(dim=-1)
        else:
            # Standard attention path: just reshape to (bsz, q_len, num_heads * head_dim)
            out_dim = self.num_attention_heads * self.head_dim
            
            # Collapse heads to match num_attention_heads if they are expanded
            if attn_output.shape[1] > self.num_attention_heads:
                num_groups = attn_output.shape[1] // self.num_attention_heads
                attn_output = attn_output.view(bsz, self.num_attention_heads, num_groups, q_len, self.head_dim).sum(dim=2)

            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, out_dim)

        # Apply output projection
        # Finally convert to torch_dtype (BF16) only after all attention math is done
        attn_output = attn_output.to(self.torch_dtype)
        
        # Ensure attn_output matches self.o_proj.in_features exactly.
        # Hybrid models like Qwen 3.5 MoE might have varying head counts.
        target_features = self.o_proj.in_features
        current_features = attn_output.shape[-1]
        
        if current_features != target_features:
            if current_features < target_features:
                ratio = target_features // current_features
                attn_output = attn_output.repeat_interleave(ratio, dim=-1)
            else:
                attn_output = attn_output[..., :target_features]

        attn_output = self.o_proj(attn_output)

        # Ensure final output hidden dimension matches self.hidden_size
        if attn_output.shape[-1] != self.hidden_size:
             pad_size = self.hidden_size - attn_output.shape[-1]
             if pad_size > 0:
                 attn_output = torch.nn.functional.pad(attn_output, (0, pad_size))
             else:
                 attn_output = attn_output[..., :self.hidden_size]

        return attn_output, present_kv_cache

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

        return attention_onnx(query, key, value, self.num_attention_heads, self.num_key_value_heads, self.head_dim, attn_mask=mask, is_causal=is_causal, scale=1.0)


def symbolic_kv_cache_update(g, past_key_value, new_key_value, kvcache_start_index):
    return g.op("trt::kv_cache_update_onnx", past_key_value, new_key_value, kvcache_start_index)

def symbolic_rope(g, x, cos, sin, position_ids):
    return g.op("trt::rope_onnx", x, cos, sin, position_ids)

def fake_attention_symbolic(g, query, key, value, kv_cache, mask=None, q_heads=64, kv_heads=2, head_dim=128):
    # This is where we map the fake op to the actual AttentionPlugin
    # The tracer sees fake_attention, but ONNX sees AttentionPlugin
    from torch.onnx.symbolic_helper import _parse_arg
    
    q_heads_val = _parse_arg(q_heads, 'i') if hasattr(q_heads, 'node') else int(q_heads)
    kv_heads_val = _parse_arg(kv_heads, 'i') if hasattr(kv_heads, 'node') else int(kv_heads)
    head_dim_val = _parse_arg(head_dim, 'i') if hasattr(head_dim, 'node') else int(head_dim)

    return g.op("trt::AttentionPlugin", query, key, value, kv_cache, 
                g.op("Constant", value_t=torch.tensor([0], dtype=torch.int32)), # context_lengths placeholder
                g.op("Constant", value_t=torch.tensor([0], dtype=torch.float32)), # rope cos_sin placeholder
                g.op("Constant", value_t=torch.tensor([0], dtype=torch.int32)), # start_index placeholder
                num_q_heads_i=q_heads_val, num_kv_heads_i=kv_heads_val, head_size_i=head_dim_val,
                outputs=2)

@torch.library.custom_op("trt::fake_attention", mutates_args=())
def fake_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    q_heads: int = 64,
    kv_heads: int = 2,
    head_dim: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    return query.clone(), kv_cache.clone()

@fake_attention.register_fake
def fake_attention_fake(query, key, value, kv_cache, mask=None, q_heads=64, kv_heads=2, head_dim=128):
    # During tracing, physically return the shape needed
    # The output of attention ALWAYS has the same number of heads as Q
    from torch.onnx.symbolic_helper import _parse_arg
    
    q_h_val = _parse_arg(q_heads, 'i') if hasattr(q_heads, 'node') else int(q_heads)
    h_dim_val = _parse_arg(head_dim, 'i') if hasattr(head_dim, 'node') else int(head_dim)
    
    bsz = query.shape[0]
    q_len = query.shape[2]
    
    return torch.empty((bsz, q_h_val, q_len, h_dim_val), device=query.device, dtype=query.dtype), torch.empty_like(kv_cache)

def symbolic_attention(g, query, key, value, kv_cache, context_lengths, rope_rotary_cos_sin, kvcache_start_index, 
                       num_q_heads, num_kv_heads, head_size, attn_mask=None, is_causal=True, scale=1.0):
    # Standard attention decomposition for ONNX
    # This avoids using the missing AttentionPlugin kernels while allowing TensorRT to optimize natively
    
    try:
        from torch.onnx.symbolic_helper import _parse_arg
        scale_val = _parse_arg(scale, "f") if hasattr(scale, "node") else float(scale)
    except:
        scale_val = 1.0

    # 1. Scale query: [B, H, S, D] * scale
    q_scaled = g.op("Mul", query, g.op("Constant", value_t=torch.tensor([scale_val], dtype=torch.float32)))
    
    # 2. Matmul QK: [B, H, S, D] x [B, H, D, S] -> [B, H, S, S]
    attn_weights = g.op("MatMul", q_scaled, g.op("Transpose", key, perm_i=[0, 1, 3, 2]))
    
    # 3. Add mask if exists
    mask_str = str(attn_mask).lower()
    if attn_mask is not None and "none" not in mask_str and mask_str != "":
        try:
            attn_weights = g.op("Add", attn_weights, attn_mask)
        except:
            pass
        
    # 4. Softmax: MUST be float32 for parser compatibility
    attn_probs = g.op("Softmax", attn_weights, axis_i=-1)
    
    # 5. Matmul PV: [B, H, S, S] x [B, H, S, D] -> [B, H, S, D]
    attn_output = g.op("MatMul", attn_probs, value)
    
    # Return attention output and KV cache (as-is for tracing)
    return attn_output, kv_cache

def register_trt_native_attention_onnx_symbolic_functions() -> None:
    """Register symbolic functions for ONNX export of TensorRT native operations."""
    from torch.onnx import register_custom_op_symbolic
    register_custom_op_symbolic("trt::kv_cache_update_onnx", symbolic_kv_cache_update, ONNX_OPSET_VERSION)
    register_custom_op_symbolic("trt::rope_onnx", symbolic_rope, ONNX_OPSET_VERSION)
    register_custom_op_symbolic("trt::fake_attention", fake_attention_symbolic, ONNX_OPSET_VERSION)
    print("Registered ONNX symbolic functions for TensorRT native kv cache update, rope, and attention")
