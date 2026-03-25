import torch
import torch.nn as nn
import torch.nn.functional as F

class Qwen3_5MoeRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Norm before gate
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        
        # Reshape weight if needed for broadcasting (handle B*S vs H mismatch)
        weight = self.weight
        if hidden_states.ndim == 2 and weight.ndim == 1:
             # Standard case: [TotalTokens, HiddenSize] * [HiddenSize]
             pass
             
        hidden_states = weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32)).to(input_dtype)

        return hidden_states.to(input_dtype)
