
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// Macro to check for cuda errors.
#ifndef CUTE_DSL_CUDA_ERROR_CHECK
#define CUTE_DSL_CUDA_ERROR_CHECK(err)                                                                                 \
    {                                                                                                                  \
        if ((err) != cudaSuccess)                                                                                      \
        {                                                                                                              \
            printf("Got Cuda Error %s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));                         \
        }                                                                                                              \
    }

#endif

typedef struct
{
    cudaLibrary_t module;
} fmha_d64_sw_fp8_Kernel_Module_t;

#ifdef __cplusplus
extern "C"
{
#endif
    void _mlir_fmha_d64_sw_fp8_cuda_init(void**);
    void _mlir_fmha_d64_sw_fp8_cuda_load_to_device(void**);
    static inline void fmha_d64_sw_fp8_Kernel_Module_Load(fmha_d64_sw_fp8_Kernel_Module_t* module)
    {
        cudaLibrary_t* libraryPtr = &(module->module);
        cudaError_t ret;
        struct
        {
            cudaLibrary_t** libraryPtr;
            cudaError_t* ret;
        } initArgs = {&libraryPtr, &ret};
        _mlir_fmha_d64_sw_fp8_cuda_init((void**) (&initArgs));
        CUTE_DSL_CUDA_ERROR_CHECK(ret);
        int32_t device_id = 0;
        struct
        {
            cudaLibrary_t** library;
            int32_t* device_id;
            cudaError_t* ret;
        } loadArgs = {&libraryPtr, &device_id, &ret};
        int32_t device_count;
        CUTE_DSL_CUDA_ERROR_CHECK(cudaGetDeviceCount(&device_count));
        for (int32_t i = 0; i < device_count; i++)
        {
            device_id = i;
            _mlir_fmha_d64_sw_fp8_cuda_load_to_device((void**) (&loadArgs));
            CUTE_DSL_CUDA_ERROR_CHECK(ret);
        }
    }

    static inline void fmha_d64_sw_fp8_Kernel_Module_Unload(fmha_d64_sw_fp8_Kernel_Module_t* module)
    {
        CUTE_DSL_CUDA_ERROR_CHECK(cudaLibraryUnload(module->module));
    }

#ifdef __cplusplus
}
#endif

typedef struct
{
    void* data;
    int32_t dynamic_shapes[4];
    int64_t dynamic_strides[3];
} fmha_d64_sw_fp8_Tensor_q_tensor_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[5];
    int64_t dynamic_strides[4];
} fmha_d64_sw_fp8_Tensor_kv_cache_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[4];
    int64_t dynamic_strides[3];
} fmha_d64_sw_fp8_Tensor_o_tensor_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[1];
} fmha_d64_sw_fp8_Tensor_cum_seqlen_k_t;

#ifdef __cplusplus
extern "C"
#endif
    void
    _mlir_fmha_d64_sw_fp8__mlir_ciface_cutlass___call_____main__BlackwellFusedMultiHeadAttentionForward_object_at__Tensorgmemoi64i64i641_Tensorgmemoi64i64i64i641_Tensorgmemoi64i64i641_Tensorgmemo1__10_10_10_10_CUstream0(
        void** args, int32_t num_args);

static inline int32_t cute_dsl_fmha_d64_sw_fp8_wrapper(fmha_d64_sw_fp8_Kernel_Module_t* module,
    fmha_d64_sw_fp8_Tensor_q_tensor_t* q_tensor, fmha_d64_sw_fp8_Tensor_kv_cache_t* kv_cache,
    fmha_d64_sw_fp8_Tensor_o_tensor_t* o_tensor, fmha_d64_sw_fp8_Tensor_cum_seqlen_k_t* cum_seqlen_k,
    int32_t window_size_left, float scale_q, float scale_k, float scale_v, float inv_scale_o, cudaStream_t stream)
{
    int32_t ret;
    void* args[11] = {q_tensor, kv_cache, o_tensor, cum_seqlen_k, &window_size_left, &scale_q, &scale_k, &scale_v,
        &inv_scale_o, &stream, &ret};
    _mlir_fmha_d64_sw_fp8__mlir_ciface_cutlass___call_____main__BlackwellFusedMultiHeadAttentionForward_object_at__Tensorgmemoi64i64i641_Tensorgmemoi64i64i64i641_Tensorgmemoi64i64i641_Tensorgmemo1__10_10_10_10_CUstream0(
        args, 11);
    return ret;
}
