
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
} vit_fmha_d128_Kernel_Module_t;

#ifdef __cplusplus
extern "C"
{
#endif
    void _mlir_vit_fmha_d128_cuda_init(void**);
    void _mlir_vit_fmha_d128_cuda_load_to_device(void**);
    static inline void vit_fmha_d128_Kernel_Module_Load(vit_fmha_d128_Kernel_Module_t* module)
    {
        cudaLibrary_t* libraryPtr = &(module->module);
        cudaError_t ret;
        struct
        {
            cudaLibrary_t** libraryPtr;
            cudaError_t* ret;
        } initArgs = {&libraryPtr, &ret};
        _mlir_vit_fmha_d128_cuda_init((void**) (&initArgs));
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
            _mlir_vit_fmha_d128_cuda_load_to_device((void**) (&loadArgs));
            CUTE_DSL_CUDA_ERROR_CHECK(ret);
        }
    }

    static inline void vit_fmha_d128_Kernel_Module_Unload(vit_fmha_d128_Kernel_Module_t* module)
    {
        CUTE_DSL_CUDA_ERROR_CHECK(cudaLibraryUnload(module->module));
    }

#ifdef __cplusplus
}
#endif

typedef struct
{
    void* data;
    int32_t dynamic_shapes[3];
    int64_t dynamic_strides[2];
} vit_fmha_d128_Tensor_q_tensor_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[3];
    int64_t dynamic_strides[2];
} vit_fmha_d128_Tensor_k_tensor_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[3];
    int64_t dynamic_strides[2];
} vit_fmha_d128_Tensor_v_tensor_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[3];
    int64_t dynamic_strides[2];
} vit_fmha_d128_Tensor_o_tensor_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[1];
} vit_fmha_d128_Tensor_cu_seqlens_t;

#ifdef __cplusplus
extern "C"
#endif
    void
    _mlir_vit_fmha_d128__mlir_ciface_cutlass___call_vit_____main__BlackwellFusedMultiHeadAttentionForward_object_at__Tensorgmemoi64i641_Tensorgmemoi64i641_Tensorgmemoi64i641_Tensorgmemoi64i641_Tensorgmemo1__0127517430(
        void** args, int32_t num_args);

static inline int32_t cute_dsl_vit_fmha_d128_wrapper(vit_fmha_d128_Kernel_Module_t* module,
    vit_fmha_d128_Tensor_q_tensor_t* q_tensor, vit_fmha_d128_Tensor_k_tensor_t* k_tensor,
    vit_fmha_d128_Tensor_v_tensor_t* v_tensor, vit_fmha_d128_Tensor_o_tensor_t* o_tensor,
    vit_fmha_d128_Tensor_cu_seqlens_t* cu_seqlens, int32_t max_seqlen, float scale_softmax_log2, float scale_softmax,
    float scale_output, cudaStream_t stream)
{
    int32_t ret;
    void* args[11] = {q_tensor, k_tensor, v_tensor, o_tensor, cu_seqlens, &max_seqlen, &scale_softmax_log2,
        &scale_softmax, &scale_output, &stream, &ret};
    _mlir_vit_fmha_d128__mlir_ciface_cutlass___call_vit_____main__BlackwellFusedMultiHeadAttentionForward_object_at__Tensorgmemoi64i641_Tensorgmemoi64i641_Tensorgmemoi64i641_Tensorgmemoi64i641_Tensorgmemo1__0127517430(
        args, 11);
    return ret;
}
