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
from torch import nn
from typing import Optional, Tuple, List, Union
import numpy as np

# Constant for ONNX opset version
ONNX_OPSET_VERSION = 17

def qwen3_delta_attention_symbolic(g, q, k, v, gate, beta, kv_cache, q_heads=16, v_heads=64, head_dim=128):
    print("MONSTER SYMBOLIC EXECUTION: qwen3_delta_attention_symbolic")
    from torch.onnx.symbolic_helper import _parse_arg
    qh = _parse_arg(q_heads, 'i') if hasattr(q_heads, 'node') else int(q_heads)
    vh = _parse_arg(v_heads, 'i') if hasattr(v_heads, 'node') else int(v_heads)
    hd = _parse_arg(head_dim, 'i') if hasattr(head_dim, 'node') else int(head_dim)
    
    ctx_len = g.op("Constant", value_t=torch.tensor([0], dtype=torch.int32))
    rope = g.op("Constant", value_t=torch.tensor([0], dtype=torch.float32))
    start_idx = g.op("Constant", value_t=torch.tensor([0], dtype=torch.int32))

    return g.op("trt::AttentionPlugin", q, k, v, kv_cache, 
                ctx_len, rope, start_idx, gate, beta,
                num_q_heads_i=qh, num_kv_heads_i=vh, head_size_i=hd,
                enable_tree_attention_i=0, enable_fp8_kv_cache_i=0, sliding_window_size_i=-1,
                outputs=2)

@torch.library.custom_op("trt::qwen3_delta_attention", mutates_args=())
def qwen3_delta_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    kv_cache: torch.Tensor,
    q_heads: int = 16,
    v_heads: int = 64,
    head_dim: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    return q.clone(), kv_cache.clone()

@qwen3_delta_attention.register_fake
def qwen3_delta_attention_fake(q, k, v, g, beta, kv_cache, q_heads=16, v_heads=64, head_dim=128):
    from torch.onnx.symbolic_helper import _parse_arg
    qh = _parse_arg(q_heads, 'i') if hasattr(q_heads, 'node') else int(q_heads)
    hd = _parse_arg(head_dim, 'i') if hasattr(head_dim, 'node') else int(head_dim)
    bsz = q.shape[0]
    q_len = q.shape[1]
    return torch.empty((bsz, q_len, qh, hd), device=q.device, dtype=q.dtype), kv_cache.clone()

def symbolic_kv_cache_update(g, past_key_value, new_key_value, kvcache_start_index):
    return g.op("trt::kv_cache_update_onnx", past_key_value, new_key_value, kvcache_start_index)

def symbolic_rope(g, x, cos, sin, position_ids):
    return g.op("trt::rope_onnx", x, cos, sin, position_ids)

def fake_attention_symbolic(g, query, key, value, kv_cache, mask=None, q_heads=64, kv_heads=2, head_dim=128, forget_g=None, beta=None):
    print("MONSTER SYMBOLIC EXECUTION: fake_attention_symbolic")
    from torch.onnx.symbolic_helper import _parse_arg
    q_heads_val = _parse_arg(q_heads, 'i') if hasattr(q_heads, 'node') else int(q_heads)
    kv_heads_val = _parse_arg(kv_heads, 'i') if hasattr(kv_heads, 'node') else int(kv_heads)
    head_dim_val = _parse_arg(head_dim, 'i') if hasattr(head_dim, 'node') else int(head_dim)

    ctx_len = g.op("Constant", value_t=torch.tensor([0], dtype=torch.int32))
    rope = g.op("Constant", value_t=torch.tensor([0], dtype=torch.float32))
    start_idx = g.op("Constant", value_t=torch.tensor([0], dtype=torch.int32))
    
    # Plugin creator expects 9 inputs: [q, k, v, kv_cache, ctx_len, rope, start_idx, g, beta]
    # If g/beta are missing, provide dummy ones (1.0 and 0.0)
    if forget_g is None:
        forget_g = g.op("Constant", value_t=torch.tensor([1.0], dtype=torch.float32))
    if beta is None:
        beta = g.op("Constant", value_t=torch.tensor([0.0], dtype=torch.float32))

    return g.op("trt_edgellm::AttentionPlugin", query, key, value, kv_cache, 
                ctx_len, rope, start_idx, forget_g, beta,
                num_q_heads_i=q_heads_val, num_kv_heads_i=kv_heads_val, head_size_i=head_dim_val,
                enable_tree_attention_i=0, enable_fp8_kv_cache_i=0, sliding_window_size_i=-1,
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
    head_dim: int = 128,
    g: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    return query.clone(), kv_cache.clone()

@fake_attention.register_fake
def fake_attention_fake(query, key, value, kv_cache, mask=None, q_heads=64, kv_heads=2, head_dim=128, g=None, beta=None):
    from torch.onnx.symbolic_helper import _parse_arg
    q_h_val = _parse_arg(q_heads, 'i') if hasattr(q_heads, 'node') else int(q_heads)
    h_dim_val = _parse_arg(head_dim, 'i') if hasattr(head_dim, 'node') else int(head_dim)
    bsz = query.shape[0]
    q_len = query.shape[2]
    return torch.empty((bsz, q_h_val, q_len, h_dim_val), device=query.device, dtype=query.dtype), torch.empty_like(kv_cache)

class EdgeLLMAttentionTRTNative(nn.Module):
    def __init__(self, attention_module: nn.Module, eagle3_draft: bool = False):
        super().__init__()
        # Use config if available
        config = getattr(attention_module, 'config', None)
        
        self.hidden_size = getattr(attention_module, 'hidden_size', 
                                  getattr(attention_module.q_proj, 'in_features', 3072))
        
        # Priority for head_dim: explicit attribute -> config attribute -> 256 default
        self.head_dim = getattr(attention_module, 'head_dim', 
                               getattr(config, 'head_dim', 256))
        
        # Explicitly create scale as a CPU float16 tensor.
        self.qk_scale = torch.tensor((self.head_dim**-0.5), dtype=torch.float16, device="cpu")
        
        self.num_attention_heads = getattr(attention_module, 'num_attention_heads', 
                                          getattr(attention_module, 'num_heads', 
                                                 getattr(config, 'num_attention_heads', 16)))
        self.num_key_value_heads = getattr(attention_module, 'num_key_value_heads', 
                                          getattr(attention_module, 'num_kv_heads', 
                                                 getattr(config, 'num_key_value_heads', 16)))
        self.torch_dtype = next(attention_module.parameters()).dtype
        
        self.q_proj = attention_module.q_proj
        self.k_proj = attention_module.k_proj
        self.v_proj = attention_module.v_proj
        self.o_proj = attention_module.o_proj
        self.eagle3_draft = eagle3_draft

    def forward(self, hidden_states, rope_rotary_cos_sin, context_lengths, kvcache_start_index, kv_cache, attention_mask=None, position_ids=None, inputs_embeds=None):
        bsz, q_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Derive heads from projected volume to handle varying architectural widths
        q_heads = query_states.shape[-1] // self.head_dim
        kv_heads = key_states.shape[-1] // self.head_dim
        print(f"DEBUG TRTNative: forward Q_heads={q_heads} KV_heads={kv_heads} D={self.head_dim} Q_shape={query_states.shape}")
        
        query_states = query_states.view(bsz, q_len, q_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, kv_heads, self.head_dim).transpose(1, 2)
        
        # GQA PADDING: If kv_heads < target_heads, we must expand KV to match global KV cache shape
        # For Qwen 3.5 0.8B, we use 32 KV heads globally to ensure plugin attribute consistency.
        target_kv_heads = 32
        if kv_heads < target_kv_heads:
            print(f"DEBUG TRTNative: PADDING KV heads from {kv_heads} to {target_kv_heads}")
            repeats = target_kv_heads // kv_heads
            # Use unsqueeze + expand + reshape for trace-safe head expansion
            key_states = key_states.unsqueeze(2).expand(-1, -1, repeats, -1, -1).reshape(bsz, target_kv_heads, q_len, self.head_dim)
            value_states = value_states.unsqueeze(2).expand(-1, -1, repeats, -1, -1).reshape(bsz, target_kv_heads, q_len, self.head_dim)
            kv_heads = target_kv_heads

        query_states = query_states * self.qk_scale
        
        attn_output, _ = fake_attention(query_states, key_states, value_states, kv_cache, attention_mask, q_heads, kv_heads, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, q_heads * self.head_dim)
        
        # Final projection volume check
        target_in = self.o_proj.in_features
        current = attn_output.shape[-1]
        if current != target_in:
            if current < target_in:
                attn_output = attn_output.repeat_interleave(target_in // current, dim=-1)
            else:
                attn_output = attn_output[..., :target_in]

        return self.o_proj(attn_output), kv_cache

def register_trt_native_attention_onnx_symbolic_functions() -> None:
    from torch.onnx import register_custom_op_symbolic
    register_custom_op_symbolic("trt::kv_cache_update_onnx", symbolic_kv_cache_update, ONNX_OPSET_VERSION)
    register_custom_op_symbolic("trt::rope_onnx", symbolic_rope, ONNX_OPSET_VERSION)
    register_custom_op_symbolic("trt::fake_attention", fake_attention_symbolic, ONNX_OPSET_VERSION)
    register_custom_op_symbolic("trt::qwen3_delta_attention", qwen3_delta_attention_symbolic, ONNX_OPSET_VERSION)
    print("Registered ONNX symbolic functions for Qwen 3.5 Gated Delta Attention")
