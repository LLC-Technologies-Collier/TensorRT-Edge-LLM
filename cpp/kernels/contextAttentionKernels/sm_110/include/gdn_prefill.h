
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
} gdn_prefill_Kernel_Module_t;

#ifdef __cplusplus
extern "C"
{
#endif
    void _mlir_gdn_prefill_cuda_init(void**);
    void _mlir_gdn_prefill_cuda_load_to_device(void**);
    static inline void gdn_prefill_Kernel_Module_Load(gdn_prefill_Kernel_Module_t* module)
    {
        cudaLibrary_t* libraryPtr = &(module->module);
        cudaError_t ret;
        struct
        {
            cudaLibrary_t** libraryPtr;
            cudaError_t* ret;
        } initArgs = {&libraryPtr, &ret};
        _mlir_gdn_prefill_cuda_init((void**) (&initArgs));
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
            _mlir_gdn_prefill_cuda_load_to_device((void**) (&loadArgs));
            CUTE_DSL_CUDA_ERROR_CHECK(ret);
        }
    }

    static inline void gdn_prefill_Kernel_Module_Unload(gdn_prefill_Kernel_Module_t* module)
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
} gdn_prefill_Tensor_q_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[4];
    int64_t dynamic_strides[3];
} gdn_prefill_Tensor_k_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[4];
    int64_t dynamic_strides[3];
} gdn_prefill_Tensor_v_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[3];
    int64_t dynamic_strides[2];
} gdn_prefill_Tensor_a_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[3];
    int64_t dynamic_strides[2];
} gdn_prefill_Tensor_b_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[1];
} gdn_prefill_Tensor_A_log_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[1];
} gdn_prefill_Tensor_dt_bias_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[2];
    int64_t dynamic_strides[1];
} gdn_prefill_Tensor_h0_source_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[1];
} gdn_prefill_Tensor_context_lengths_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[4];
    int64_t dynamic_strides[3];
} gdn_prefill_Tensor_o_t;

#ifdef __cplusplus
extern "C"
#endif
    void
    _mlir_gdn_prefill__mlir_ciface_cutlass_run_prefill_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i641_Tensorgmemoi64i641_Tensorgmemo1_Tensorgmemo1_Tensorgmemo128128i64div1638416(
        void** args, int32_t num_args);

static inline int32_t cute_dsl_gdn_prefill_wrapper(gdn_prefill_Kernel_Module_t* module, gdn_prefill_Tensor_q_t* q,
    gdn_prefill_Tensor_k_t* k, gdn_prefill_Tensor_v_t* v, gdn_prefill_Tensor_a_t* a, gdn_prefill_Tensor_b_t* b,
    gdn_prefill_Tensor_A_log_t* A_log, gdn_prefill_Tensor_dt_bias_t* dt_bias, gdn_prefill_Tensor_h0_source_t* h0_source,
    gdn_prefill_Tensor_context_lengths_t* context_lengths, gdn_prefill_Tensor_o_t* o, int32_t seq_len,
    cudaStream_t stream)
{
    int32_t ret;
    void* args[13] = {q, k, v, a, b, A_log, dt_bias, h0_source, context_lengths, o, &seq_len, &stream, &ret};
    _mlir_gdn_prefill__mlir_ciface_cutlass_run_prefill_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i641_Tensorgmemoi64i641_Tensorgmemo1_Tensorgmemo1_Tensorgmemo128128i64div1638416(
        args, 13);
    return ret;
}
