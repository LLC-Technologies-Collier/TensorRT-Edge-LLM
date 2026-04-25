/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "qwen3DeltaAttention.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>

namespace trt_edgellm
{
namespace kernels
{

__device__ __forceinline__ float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float silu(float x)
{
    return x * sigmoid(x);
}

template <int HEAD_SIZE>
__global__ void qwen3DeltaAttentionKernel(Qwen3DeltaAttentionParams params)
{
    // Each block handles one [batch, head]
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;

    if (b >= params.batch_size || h >= params.num_q_heads) return;

    // Head mapping for GQA (if num_q_heads > num_kv_heads)
    int kv_h = h / (params.num_q_heads / params.num_kv_heads);

    // State pointer: [B, H, D, D]
    float state_local[HEAD_SIZE]; // One row per thread (row tid)
    float* state_ptr = reinterpret_cast<float*>(params.state) + (b * params.num_q_heads + h) * HEAD_SIZE * HEAD_SIZE;
    
    // Load initial state
    for(int j=0; j<HEAD_SIZE; ++j) {
        state_local[j] = state_ptr[tid * HEAD_SIZE + j];
    }

    // Load norm weights if provided
    float w_val = 1.0f;
    if (params.norm_weight) {
        w_val = __half2float(reinterpret_cast<const half*>(params.norm_weight)[h * HEAD_SIZE + tid]);
    }
    
    __shared__ float s_q[HEAD_SIZE];
    __shared__ float s_k[HEAD_SIZE];
    __shared__ float s_v[HEAD_SIZE];
    __shared__ float s_sq_sum;

    for (int t = 0; t < params.seq_len; ++t)
    {
        // 1. Load inputs for this timestep
        const half* q_ptr = reinterpret_cast<const half*>(params.q) + ((b * params.seq_len + t) * params.num_q_heads + h) * HEAD_SIZE;
        const half* k_ptr = reinterpret_cast<const half*>(params.k) + ((b * params.seq_len + t) * params.num_kv_heads + kv_h) * HEAD_SIZE;
        const half* v_ptr = reinterpret_cast<const half*>(params.v) + ((b * params.seq_len + t) * params.num_kv_heads + kv_h) * HEAD_SIZE;
        const half* g_ptr = reinterpret_cast<const half*>(params.g) + (b * params.seq_len + t) * params.num_q_heads + h;
        const half* beta_ptr = reinterpret_cast<const half*>(params.beta) + (b * params.seq_len + t) * params.num_q_heads + h;
        const half* z_ptr = params.z ? (reinterpret_cast<const half*>(params.z) + ((b * params.seq_len + t) * params.num_q_heads + h) * HEAD_SIZE) : nullptr;

        float g_val = __half2float(*g_ptr);
        float beta_val = __half2float(*beta_ptr);
        float exp_g = expf(g_val);
        float sig_beta = sigmoid(beta_val);

        // Load Q, K, V into shared memory
        s_q[tid] = __half2float(q_ptr[tid]);
        s_k[tid] = __half2float(k_ptr[tid]);
        s_v[tid] = __half2float(v_ptr[tid]);
        __syncthreads();
        
        // Parallel L2 Norm for Q and K
        float q_val = s_q[tid];
        float k_val = s_k[tid];
        
        // Sum of squares for Q
        if (tid == 0) s_sq_sum = 0.0f;
        __syncthreads();
        atomicAdd(&s_sq_sum, q_val * q_val);
        __syncthreads();
        float q_inv_norm = rsqrtf(s_sq_sum + 1e-6f);
        s_q[tid] *= q_inv_norm;
        
        // Sum of squares for K
        if (tid == 0) s_sq_sum = 0.0f;
        __syncthreads();
        atomicAdd(&s_sq_sum, k_val * k_val);
        __syncthreads();
        float k_inv_norm = rsqrtf(s_sq_sum + 1e-6f);
        s_k[tid] *= k_inv_norm;
        __syncthreads();

        // 2. State update
        // a. Decay hidden state: h *= exp(g)
        for (int j = 0; j < HEAD_SIZE; ++j)
        {
            state_local[j] *= exp_g;
        }
        
        // b. Compute delta rule update: v = (v - h*k) * sigmoid(beta)
        float dot_h_k = 0.0f;
        for (int j = 0; j < HEAD_SIZE; ++j)
        {
            dot_h_k += state_local[j] * s_k[j];
        }
        
        float v_update = (s_v[tid] - dot_h_k) * sig_beta;
        __syncthreads();
        s_v[tid] = v_update; // reuse shared memory for the updated v
        __syncthreads();

        // c. Update hidden state: h += k_outer_v
        float sk_reg = s_k[tid];
        for (int j = 0; j < HEAD_SIZE; ++j)
        {
            state_local[j] += sk_reg * s_v[j];
        }
        __syncthreads();

        // 3. Compute Output: o = h * q
        float o_val = 0.0f;
        for (int j = 0; j < HEAD_SIZE; ++j)
        {
            o_val += state_local[j] * s_q[j];
        }
        
        // 4. Post-processing: Gated RMSNorm
        if (tid == 0) s_sq_sum = 0.0f;
        __syncthreads();
        atomicAdd(&s_sq_sum, o_val * o_val);
        __syncthreads();
        
        float rms = rsqrtf(s_sq_sum / HEAD_SIZE + params.eps);
        o_val = (o_val * rms) * w_val;
        
        if (z_ptr) {
            float z_val = __half2float(z_ptr[tid]);
            o_val *= silu(z_val);
        }
        
        half* out_ptr = reinterpret_cast<half*>(params.output) + ((b * params.seq_len + t) * params.num_q_heads + h) * HEAD_SIZE;
        out_ptr[tid] = __float2half(o_val * params.scale);
    }
    
    // Write back final state
    for(int j=0; j<HEAD_SIZE; ++j) {
        state_ptr[tid * HEAD_SIZE + j] = state_local[j];
    }
}

void invokeQwen3DeltaAttention(Qwen3DeltaAttentionParams const& params, cudaStream_t stream)
{
    dim3 grid(params.batch_size, params.num_q_heads);
    dim3 block(params.head_size); 

    if (params.head_size == 128)
    {
        qwen3DeltaAttentionKernel<128><<<grid, block, 0, stream>>>(params);
    }
    else if (params.head_size == 256)
    {
        qwen3DeltaAttentionKernel<256><<<grid, block, 0, stream>>>(params);
    }
    else
    {
        // Fallback or other sizes
    }
}

} // namespace kernels
} // namespace trt_edgellm
