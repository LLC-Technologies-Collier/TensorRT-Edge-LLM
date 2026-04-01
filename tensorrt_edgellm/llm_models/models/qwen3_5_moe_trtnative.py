# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from typing import Optional, List, Tuple

class Qwen3DeltaAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, kv_cache, q_heads, v_heads, head_dim):
        return q.clone(), kv_cache.clone()

    @staticmethod
    def symbolic(g_graph, q, k, v, g, beta, kv_cache, q_heads, v_heads, head_dim):
        # We pass EVERYTHING as 1D to bypass TRT 10.x parser reshape bugs
        ctx_len = g_graph.op("Constant", value_t=torch.tensor([0], dtype=torch.int32))
        rope = g_graph.op("Constant", value_t=torch.tensor([0], dtype=torch.float32))
        start_idx = g_graph.op("Constant", value_t=torch.tensor([0], dtype=torch.int32))

        return g_graph.op("trt_edgellm::AttentionPlugin", q, k, v, kv_cache,
                    ctx_len, rope, start_idx, g, beta,
                    num_q_heads_i=int(q_heads), num_kv_heads_i=int(v_heads), head_size_i=int(head_dim),
                    enable_tree_attention_i=0, enable_fp8_kv_cache_i=0, sliding_window_size_i=-1,
                    outputs=2)

class Qwen3_5MoeGatedDeltaNetTRTNative(nn.Module):
    """
    TensorRT native implementation of Qwen 3.5 MoE Gated Delta Attention.
    Uses a 1D-Neutralized trace to bypass parser volume bugs.
    """

    def __init__(self, attention_module: nn.Module, eagle3_draft: bool = False):
        super().__init__()
        # Use config if available, otherwise fallback to attributes or 122B defaults
        config = getattr(attention_module, 'config', None)
        
        self.hidden_size = getattr(attention_module, 'hidden_size', 
                                  getattr(config, 'hidden_size', 3072))
        
        # SHAPE PARADOX: All layers must match the global head_dim (max 128) and global KV head count (32).
        self.head_dim = 128
        self.num_kv_heads = 32 # Global max KV heads for Qwen 0.8B
        
        # Calculate volumes from original config
        orig_q_heads = getattr(attention_module, 'num_k_heads',
                              getattr(config, 'linear_num_key_heads', 16))
        orig_v_heads = getattr(attention_module, 'num_v_heads',
                              getattr(config, 'linear_num_value_heads', 64))
        orig_hd = getattr(attention_module, 'head_v_dim',
                         getattr(config, 'linear_value_head_dim', 128))
        
        q_vol = orig_q_heads * orig_hd
        v_vol = orig_v_heads * orig_hd
        
        # Natural heads before padding
        self.num_q_heads = q_vol // self.head_dim
        self.num_v_heads_natural = v_vol // self.head_dim
        
        print(f"DEBUG Qwen35Wrapper: Q_heads={self.num_q_heads} V_heads_natural={self.num_v_heads_natural} target_KV_heads={self.num_kv_heads} D={self.head_dim}")
        
        self.q_dim = q_vol
        self.k_dim = q_vol
        self.v_dim = v_vol
        
        device = next(attention_module.parameters()).device
        dtype = next(attention_module.parameters()).dtype
        
        # Projections include Z-gate and G-gate
        self.qkvz_proj = nn.Linear(self.hidden_size, self.q_dim + self.k_dim + self.v_dim * 2, bias=False, dtype=dtype).to(device)
        self.ba_proj = nn.Linear(self.hidden_size, self.num_kv_heads * 2, bias=False, dtype=dtype).to(device)
        self.out_proj = attention_module.out_proj

    def initialize_unique_weights(self):
        """
        Populate model with unique random noise to prevent TensorRT deduplication.
        Handles meta-tensors by replacing them with real CPU tensors during this phase.
        """
        print("DEBUG: Initializing unique random weights for refit slots...")
        for name, param in self.named_parameters():
            seed = sum(ord(c) for c in name) % 1000000
            rng = torch.Generator(device='cpu')
            rng.manual_seed(seed)
            
            # Create unique data
            data = torch.randn(param.shape, generator=rng, dtype=torch.float16, device='cpu') * 1e-3
            
            # Replace param data
            if param.is_meta:
                # Replace meta tensor with real CPU tensor containing unique noise
                # This is critical for the tracer to see unique storage
                new_param = torch.nn.Parameter(data, requires_grad=False)
                # Find the parent module and set the attribute
                parts = name.split('.')
                curr = self
                for p in parts[:-1]: curr = getattr(curr, p)
                setattr(curr, parts[-1], new_param)
            else:
                with torch.no_grad():
                    param.copy_(data)
        print("DEBUG: Unique weight initialization complete.")

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,
    ):
        bsz, q_len, _ = hidden_states.shape
        
        # 1. Project
        qkvz = self.qkvz_proj(hidden_states)
        ba = self.ba_proj(hidden_states)

        # 2. 1D NEUTRALIZATION: Flatten everything to 1D before passing to plugin
        # This removes all Reshape/Shuffle nodes from the custom branches
        q = qkvz[..., :self.q_dim].reshape(-1)
        k = qkvz[..., self.q_dim : self.q_dim+self.k_dim].reshape(-1)
        v = qkvz[..., self.q_dim+self.k_dim : self.q_dim+self.k_dim+self.v_dim].reshape(-1)
        g = qkvz[..., self.q_dim+self.k_dim+self.v_dim :].reshape(-1)
        beta = ba[..., :self.num_kv_heads].reshape(-1)

        # 3. Call plugin
        # The plugin will handle the 4D reconstruction internally
        attn_output_1d, present_kv_cache = Qwen3DeltaAttentionFunction.apply(
            q, k, v, g, beta,
            kv_cache,
            self.num_q_heads, self.num_kv_heads, self.head_dim
        )

        # 4. Final Reconstruction for Out-Proj
        # We must reshape the 1D output back to 3D for the linear layer
        attn_output = attn_output_1d.reshape(bsz, q_len, self.num_q_heads * self.head_dim)
        
        target_in = self.out_proj.in_features
        current = attn_output.shape[-1]
        if current < target_in:
            attn_output = attn_output.repeat_interleave(target_in // current, dim=-1)
            
        attn_output = self.out_proj(attn_output)
        
        return attn_output, present_kv_cache
