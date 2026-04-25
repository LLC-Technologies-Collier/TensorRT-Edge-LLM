/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernels
{

struct Qwen3DeltaAttentionParams
{
    void const* q;
    void const* k;
    void const* v;
    void const* g;
    void const* beta;
    void const* z;           // [batch, seq_len, num_q_heads, head_size]
    void const* norm_weight; // [num_q_heads, head_size]
    void* state;             // [batch, num_q_heads, head_size, head_size]
    void* output;

    int32_t batch_size;
    int32_t seq_len;
    int32_t num_q_heads;
    int32_t num_kv_heads;
    int32_t head_size;
    float scale;
    float eps;
    bool is_prefill;
};

void invokeQwen3DeltaAttention(Qwen3DeltaAttentionParams const& params, cudaStream_t stream);

} // namespace kernels
} // namespace trt_edgellm
