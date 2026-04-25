# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from typing import Optional, List, Tuple, Any

class Qwen3DeltaAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, state_in, z, norm_weight, q_heads, v_heads, head_dim, eps):
        return q.clone(), state_in.clone()

    @staticmethod
    def symbolic(g_graph, q, k, v, g, beta, state_in, z, norm_weight, q_heads, v_heads, head_dim, eps):
        # The Qwen3DeltaAttentionPlugin (C++) now expects 8 inputs:
        # [0] q, [1] k, [2] v, [3] g, [4] beta, [5] state_in, [6] z, [7] norm_weight
        return g_graph.op("trt::qwen3_delta_attention", q, k, v, g, beta, state_in, z, norm_weight,
                    num_q_heads_i=int(q_heads), num_kv_heads_i=int(v_heads), head_size_i=int(head_dim),
                    eps_f=float(eps), outputs=2)

class Qwen3_5MoeGatedDeltaNetTRTNative(nn.Module):
    """
    TensorRT native implementation of Qwen 3.5 MoE Gated Delta Attention.
    Uses a 1D-Neutralized trace to bypass parser volume bugs.
    """

    def __init__(self, attention_module: nn.Module, eagle3_draft: bool = False, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        print(f"DEBUG GatedDeltaNet init: layer_idx={layer_idx}")
        import sys; sys.stdout.flush()
        # Use config if available, otherwise fallback to attributes or 122B defaults
        config = getattr(attention_module, 'config', None)
        
        self.hidden_size = getattr(attention_module, 'hidden_size', 
                                  getattr(config, 'hidden_size', 3072))
        
        self.head_dim = 128
        
        self.num_v_heads = getattr(attention_module, 'num_v_heads',
                                  getattr(config, 'linear_num_value_heads', 16))
        self.num_k_heads = getattr(attention_module, 'num_k_heads',
                                  getattr(config, 'linear_num_key_heads', 16))
        self.head_k_dim = getattr(attention_module, 'head_k_dim',
                                 getattr(config, 'linear_key_head_dim', 128))
        self.head_v_dim = getattr(attention_module, 'head_v_dim',
                                 getattr(config, 'linear_value_head_dim', 128))
        
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        
        # Standardization check:
        self.num_q_heads = self.key_dim // self.head_dim
        
        device = next(attention_module.parameters()).device
        dtype = next(attention_module.parameters()).dtype
        
        # Original projections from the module
        # qkvz_proj: [Q, K, V, Z] (122B style) OR separate: [QKV], [Z], [B], [A] (0.8B style)
        if hasattr(attention_module, 'in_proj_qkvz'):
            self.in_proj_qkvz = attention_module.in_proj_qkvz
            self.has_unified_qkvz = True
        else:
            self.in_proj_qkv = attention_module.in_proj_qkv
            self.in_proj_z = attention_module.in_proj_z
            self.has_unified_qkvz = False
            
        if hasattr(attention_module, 'in_proj_ba'):
            self.in_proj_ba = attention_module.in_proj_ba
            self.has_unified_ba = True
        else:
            self.in_proj_b = attention_module.in_proj_b
            self.in_proj_a = attention_module.in_proj_a
            self.has_unified_ba = False
        
        # Discretization parameters
        self.dt_bias = attention_module.dt_bias
        self.A_log = attention_module.A_log
        
        # Normalization and output projection
        self.norm = attention_module.norm
        self.out_proj = attention_module.out_proj
        self.conv1d = getattr(attention_module, "conv1d", None)
        self.activation = getattr(attention_module, "activation", "silu")

    def initialize_unique_weights(self):
        """
        Populate model with unique random noise to prevent TensorRT deduplication.
        """
        print("DEBUG: Initializing unique random weights for GatedDeltaNet...")
        for name, param in self.named_parameters():
            seed = sum(ord(c) for c in name) % 1000000
            rng = torch.Generator(device='cpu')
            rng.manual_seed(seed)
            data = torch.randn(param.shape, generator=rng, dtype=torch.float16, device='cpu') * 1e-3
            if param.is_meta:
                new_param = torch.nn.Parameter(data, requires_grad=False)
                parts = name.split('.')
                curr = self
                for p in parts[:-1]: curr = getattr(curr, p)
                setattr(curr, parts[:-1], new_param)
            else:
                with torch.no_grad(): param.copy_(data)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor = None,
        rope_rotary_cos_sin: torch.Tensor = None,
        context_lengths: torch.Tensor = None,
        kvcache_start_index: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,
        past_key_value: Optional[Any] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.shape
        
        # Extract states from past_key_value if provided
        conv_state = None
        recurrent_state = None
        layer_idx = getattr(self, "layer_idx", 0)

        if past_key_value is not None:
            if hasattr(past_key_value, "conv_states"):
                conv_state = past_key_value.conv_states[layer_idx]
                recurrent_state = past_key_value.recurrent_states[layer_idx]
            else:
                # Handle flattened or other formats if needed
                pass

        # 1. Project QKV
        if self.has_unified_qkvz:
            projected_states_qkvz = self.in_proj_qkvz(hidden_states)
            split_qkvz = torch.split(projected_states_qkvz, [self.key_dim, self.key_dim, self.value_dim, self.value_dim], dim=-1)
            mixed_qkv, z = torch.cat(split_qkvz[:3], dim=-1), split_qkvz[3]
        else:
            mixed_qkv = self.in_proj_qkv(hidden_states)
            z = self.in_proj_z(hidden_states)

        # 2. Causal Conv1d
        from ..layers.mamba_plugin import causal_conv1d_plugin
        # Transpose to [B, D, S] for conv
        mixed_qkv = mixed_qkv.transpose(1, 2)
        # causal_conv1d_plugin(x, weight, bias, conv_state, stride, padding, dilation, groups)
        mixed_qkv, conv_state_out = causal_conv1d_plugin(
            mixed_qkv,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            conv_state,
            1, # stride
            self.conv1d.kernel_size[0] - 1, # padding
            1, # dilation
            mixed_qkv.shape[1], # groups
        )
        # Transpose back to [B, S, D]
        mixed_qkv = mixed_qkv.transpose(1, 2)
        
        # Apply SiLU activation (Qwen 3.5 style)
        mixed_qkv = torch.nn.functional.silu(mixed_qkv)

        # 3. Split Q, K, V
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        
        if self.has_unified_ba:
            projected_states_ba = self.in_proj_ba(hidden_states)
            split_ba = torch.split(projected_states_ba, [self.num_v_heads, self.num_v_heads], dim=-1)
            b, a = split_ba
        else:
            b = self.in_proj_b(hidden_states)
            a = self.in_proj_a(hidden_states)

        # 4. Discretization
        g = -self.A_log.float().exp() * torch.nn.functional.softplus(a.float() + self.dt_bias)
        g = g.to(query.dtype)

        # 5. 1D NEUTRALIZATION: Flatten to 1D to bypass TRT volume bugs
        q_flat = query.reshape(-1)
        k_flat = key.reshape(-1)
        v_flat = value.reshape(-1)
        g_flat = g.reshape(-1)
        beta_flat = b.reshape(-1)
        z_flat = z.reshape(-1)

        # 6. Call Delta Attention plugin
        # Order: q, k, v, g, beta, state_in, z, norm_weight
        attn_output, recurrent_state_out = Qwen3DeltaAttentionFunction.apply(
            q_flat, k_flat, v_flat, g_flat, beta_flat, recurrent_state, z_flat, self.norm.weight,
            self.num_v_heads, self.num_k_heads, self.head_dim, self.norm.variance_epsilon
        )

        # 7. Final Reconstruction
        attn_output = attn_output.reshape(bsz, q_len, self.value_dim)
        attn_output = self.out_proj(attn_output)
        
        return attn_output, (conv_state_out, recurrent_state_out)
