import sys
import torch
import torch.distributed.fsdp

# Patch FSDPModule for modelopt compatibility
if not hasattr(torch.distributed.fsdp, 'FSDPModule'):
    print("Patching FSDPModule...")
    if hasattr(torch.distributed.fsdp, 'FullyShardedDataParallel'):
        torch.distributed.fsdp.FSDPModule = torch.distributed.fsdp.FullyShardedDataParallel
    else:
        # Dummy if FSDP is missing entirely (unlikely)
        class FSDPModule: pass
        torch.distributed.fsdp.FSDPModule = FSDPModule

# Patch mamba_ssm
try:
    import mamba_ssm.ops.triton.layernorm_gated
    print("Patching mamba_ssm rmsnorm_fn...")
    
    def naive_rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-6, group_size=None, norm_before_gate=False, is_rms_norm=True):
        dtype = x.dtype
        x = x.float()
        if z is not None:
            z = z.float()
            x = x * torch.nn.functional.silu(z) 
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        if weight is not None:
            weight = weight.float()
            x = x * weight
        if bias is not None:
            bias = bias.float()
            x = x + bias
        return x.to(dtype)

    mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn = naive_rmsnorm_fn
except ImportError:
    pass
except Exception as e:
    print(f"Failed to patch mamba_ssm: {e}")

# Run export
from tensorrt_edgellm.scripts.export_llm import main
if __name__ == "__main__":
    main()
