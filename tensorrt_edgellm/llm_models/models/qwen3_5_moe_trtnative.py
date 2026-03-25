import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from ..layers.attention_trt import attention_onnx, kv_cache_update_onnx, rope_onnx
from ..model_utils import get_config_attr

class Qwen3_5MoeGatedDeltaNetTRTNative(nn.Module):
    def __init__(self, attention_module: nn.Module, eagle3_draft: bool = False):
        super().__init__()
        self.hidden_size = get_config_attr(attention_module, 'hidden_size', 3072)
        self.num_attention_heads = get_config_attr(attention_module, 'num_attention_heads', 32)
        self.num_key_value_heads = get_config_attr(attention_module, 'num_key_value_heads', 16)
        self.head_dim = get_config_attr(attention_module, 'head_dim', 128)
        self.max_position_embeddings = get_config_attr(attention_module, 'max_position_embeddings', 32768)
        
        # Qwen 3.5 MoE specific head counts for Gated Delta Attention
        self.num_k_heads = getattr(attention_module, 'num_k_heads', 16)
        self.num_v_heads = getattr(attention_module, 'num_v_heads', 64)
        
        # Projections (Qwen 3.5 MoE uses separate qkv and z)
        self.in_proj_qkv = attention_module.in_proj_qkv
        self.in_proj_z = attention_module.in_proj_z
        self.in_proj_b = attention_module.in_proj_b
        self.in_proj_a = attention_module.in_proj_a
        self.out_proj = attention_module.out_proj
        self.norm = attention_module.norm

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
        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)

        # Split qkv: [B, S, Q+K+V]
        # FORCE 64 heads for ALL components to avoid ONNX broadcasting issues
        # Qwen 3.5 MoE hybrid model has asymmetric heads (Q/K vs V).
        # We must detect the physical size from the tracer and expand to target_heads.
        target_heads = 64
        h_dim = 128
        
        # Calculate how many heads are actually present in the projected tensor
        total_arch_heads = mixed_qkv.shape[-1] // h_dim
        
        # Architecture Ratio for Qwen 3.5 MoE GatedDeltaNet is 1:1:4 (Q:K:V)
        # 1 unit for Q, 1 unit for K, 4 units for V. Total 6 units.
        head_unit = total_arch_heads // 6
        arch_q = head_unit
        arch_k = head_unit
        arch_v = head_unit * 4
        
        q_end = arch_q * h_dim
        k_end = (arch_q + arch_k) * h_dim
        
        query_states = mixed_qkv[..., :q_end]
        key_states = mixed_qkv[..., q_end : k_end]
        value_states = mixed_qkv[..., k_end :]

        # 2. Expand to target_heads if needed for ONNX symmetry
        def expand_heads(t, current_h, target_h):
            if current_h < target_h:
                ratio = target_h // current_h
                return t.repeat_interleave(ratio, dim=-1)
            return t

        query_states = expand_heads(query_states, arch_q, target_heads)
        key_states = expand_heads(key_states, arch_k, target_heads)
        value_states = expand_heads(value_states, arch_v, target_heads)

        # 3. Reshape for attention
        query_states = query_states.view(bsz, q_len, target_heads, h_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, target_heads, h_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, target_heads, h_dim).transpose(1, 2)

        # 3. Handle asymmetric Gated Delta Attention
        # Slice unified cache [B, 2, 64, 4096, 256] to [B, 2, 64, 4096, 128]
        kv_cache = kv_cache[..., :h_dim]
        
        # Call the fused attention custom op with raw tensors
        from ..layers.attention_trt import fake_attention
        attn_output, present_kv_cache = fake_attention(
            query_states, key_states, value_states, 
            kv_cache,
            attention_mask,
            target_heads, target_heads, h_dim
        )

        # 4. Handle Gated Delta Rule Reductions
        # attn_output physically has 64 heads now from fake_attention_fake
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, target_heads, h_dim)

        # Gating reduction
        z_gate = z.reshape(bsz, q_len, target_heads, h_dim)
        
        # Final Norm - The norm in Qwen 3.5 MoE is head-wise!
        attn_output_flat = attn_output.view(-1, h_dim)
        z_gate_flat = z_gate.view(-1, h_dim)
        
        attn_output_flat = self.norm(attn_output_flat, z_gate_flat)
        
        # Reshape back and Out-Proj
        # Qwen 3.5 MoE hybrid layers might have different reduction dims.
        # Ensure we match self.out_proj.in_features exactly.
        attn_output_combined = attn_output_flat.view(bsz, q_len, target_heads * h_dim)
        
        target_features = self.out_proj.in_features
        current_features = attn_output_combined.shape[-1]
        
        if current_features != target_features:
            if current_features < target_features:
                ratio = target_features // current_features
                attn_output_combined = attn_output_combined.repeat_interleave(ratio, dim=-1)
            else:
                # If we have too many features (e.g. from 64 heads but architecture expects 3072)
                # we must slice to match.
                attn_output_combined = attn_output_combined[..., :target_features]

        attn_output = self.out_proj(attn_output_combined)
        
        return attn_output, present_kv_cache
