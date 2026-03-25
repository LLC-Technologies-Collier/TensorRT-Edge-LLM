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
    Adapted from TensorRT-LLM PR #12302 and transformers modeling.
    """
    normalized_weights = {}
    for key, tensor in weights.items():
        # Remove model.language_model. prefix if present
        if key.startswith('model.language_model.'):
            key = 'model.' + key[len('model.language_model.'):]
        normalized_weights[key] = tensor

    # Pack split linear-attention projections
    num_k_groups = getattr(config, 'linear_num_key_heads', 16)
    
    # Pattern for Qwen 3.5 linear attention projections
    proj_pattern = re.compile(r'^(.*\.linear_attn)\.in_proj_(q|k|v|z|b|a)\.weight$')
    
    grouped_projs = defaultdict(dict)
    final_weights = {}
    
    for name, tensor in normalized_weights.items():
        match = proj_pattern.match(name)
        if match:
            prefix, proj_type = match.groups()
            grouped_projs[prefix][proj_type] = tensor
        else:
            final_weights[name] = tensor
            
    for prefix, projs in grouped_projs.items():
        # Qwen 3.5 MoE uses separate q, k, v, z in HF, but we expect in_proj_qkv and in_proj_z
        if all(k in projs for k in ['q', 'k', 'v']):
            # Q(4096) + K(2048) + V(8192) = 14336. Wait, the 122B has 12288 total?
            # Re-verifying from error: size 12288. Split as 32/16/48? 
            # 32*128=4096, 16*128=2048, 48*128=6144. Total = 12288.
            # We cat them to form in_proj_qkv
            final_weights[f'{prefix}.in_proj_qkv.weight'] = torch.cat([projs['q'], projs['k'], projs['v']], dim=0)
        
        if 'z' in projs:
            final_weights[f'{prefix}.in_proj_z.weight'] = projs['z']
            
        if all(k in projs for k in ['b', 'a']):
            packed_ba = _pack_projection_tensor([projs['b'], projs['a']], num_k_groups)
            final_weights[f'{prefix}.in_proj_ba.weight'] = packed_ba
            
    # Handle MoE experts and shared expert
    expert_final_weights = {}
    for key, tensor in final_weights.items():
        new_key = key
        # Move weights to CPU to materialize data before set_module_tensor_to_device
        if tensor.device.type == 'meta':
            expert_final_weights[new_key] = torch.zeros(tensor.shape, dtype=tensor.dtype, device='cpu')
        else:
            expert_final_weights[new_key] = tensor.to('cpu')
            
    return expert_final_weights
