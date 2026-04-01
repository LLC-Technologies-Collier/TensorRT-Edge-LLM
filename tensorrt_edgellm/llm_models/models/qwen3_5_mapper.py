import torch
import re
import math
from collections import defaultdict

def _pack_projection_tensor(tensors, num_groups):
    reference_shape = tensors[0].shape[1:]
    reshaped = [
        tensor.reshape(num_groups, tensor.shape[0] // num_groups, *reference_shape)
        for tensor in tensors
    ]
    return torch.cat(reshaped, dim=1).reshape(-1, *reference_shape).contiguous()

def preprocess_qwen3_5_weights(weights, config):
    """
    Preprocess weights for Qwen 3.5 MoE models.
    Matches the official upstream grouped-interleaved layout for hybrid layers.
    """
    normalized_weights = {}
    for key, tensor in weights.items():
        # Remove model.language_model. prefix if present
        if key.startswith('model.language_model.'):
            key = 'model.' + key[len('model.language_model.'):]
        normalized_weights[key] = tensor

    # Pack split linear-attention projections into grouped-interleaved layout
    num_k_groups = getattr(config, 'linear_num_key_heads', 16)
    
    # Pattern for Qwen 3.5 linear attention projections
    proj_pattern = re.compile(r'^(.*\.linear_attn)\.in_proj_(q|k|v|z|b|a)\.(weight|bias|weight_scale_inv)$')
    
    grouped_projs = defaultdict(dict)
    final_weights = {}
    
    for name, tensor in normalized_weights.items():
        match = proj_pattern.match(name)
        if match:
            prefix, proj_type, suffix = match.groups()
            grouped_projs[(prefix, suffix)][proj_type] = tensor
        else:
            final_weights[name] = tensor
            
    for (prefix, suffix), projs in grouped_projs.items():
        # Qwen 3.5 MoE uses split q, k, v, z, b, a. 
        # Layout: Q, K, V, Z, Beta, A (Unified Merged Projection)
        if all(k in projs for k in ['q', 'k', 'v', 'z', 'b', 'a']):
            print(f"DEBUG Mapper: Packing Merged Projection for {prefix} ({suffix})")
            # Concat everything along the output dimension (dim 0)
            # Volume: Q_dim + K_dim + V_dim + V_dim + V_heads + V_heads
            # For 122B: (16*128) + (16*128) + (64*128) + (64*128) + 64 + 64 = 20480 + 128 = 20608
            packed_merged = torch.cat([projs['q'], projs['k'], projs['v'], projs['z'], projs['b'], projs['a']], dim=0)
            final_weights[f'{prefix}.merged_weight.{suffix}'] = packed_merged
        else:
            # Fallback for standard layers or partially mapped components
            for p_type, p_tensor in projs.items():
                final_weights[f'{prefix}.in_proj_{p_type}.{suffix}'] = p_tensor
            
    # Move finalized weights to CPU
    return {k: v.cpu() if v.device.type != 'meta' else torch.zeros(v.shape, dtype=v.dtype) 
            for k, v in final_weights.items()}
