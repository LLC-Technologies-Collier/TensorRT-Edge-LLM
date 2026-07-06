
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
} ffpa_d512_causal_gqa8_Kernel_Module_t;

#ifdef __cplusplus
extern "C"
{
#endif
    void _mlir_ffpa_d512_causal_gqa8_cuda_init(void**);
    void _mlir_ffpa_d512_causal_gqa8_cuda_load_to_device(void**);
    static inline void ffpa_d512_causal_gqa8_Kernel_Module_Load(ffpa_d512_causal_gqa8_Kernel_Module_t* module)
    {
        cudaLibrary_t* libraryPtr = &(module->module);
        cudaError_t ret;
        struct
        {
            cudaLibrary_t** libraryPtr;
            cudaError_t* ret;
        } initArgs = {&libraryPtr, &ret};
        _mlir_ffpa_d512_causal_gqa8_cuda_init((void**) (&initArgs));
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
            _mlir_ffpa_d512_causal_gqa8_cuda_load_to_device((void**) (&loadArgs));
            CUTE_DSL_CUDA_ERROR_CHECK(ret);
        }
    }

    static inline void ffpa_d512_causal_gqa8_Kernel_Module_Unload(ffpa_d512_causal_gqa8_Kernel_Module_t* module)
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
} ffpa_d512_causal_gqa8_Tensor_mQ_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[3];
    int64_t dynamic_strides[2];
} ffpa_d512_causal_gqa8_Tensor_mK_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[3];
    int64_t dynamic_strides[2];
} ffpa_d512_causal_gqa8_Tensor_mV_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[3];
    int64_t dynamic_strides[2];
} ffpa_d512_causal_gqa8_Tensor_mO_t;

#ifdef __cplusplus
extern "C"
#endif
    void
    _mlir_ffpa_d512_causal_gqa8__mlir_ciface_cutlass___call_____main__FFPAFmhaAmpere_object_at__Tensorgmemo512i64div512i64div5125121_Tensorgmemo512i64div512i64div5125121_Tensorgmemo512i64div512i64div5125121_Tensorgmemo512i64d(
        void** args, int32_t num_args);

static inline int32_t cute_dsl_ffpa_d512_causal_gqa8_wrapper(ffpa_d512_causal_gqa8_Kernel_Module_t* module,
    ffpa_d512_causal_gqa8_Tensor_mQ_t* mQ, ffpa_d512_causal_gqa8_Tensor_mK_t* mK, ffpa_d512_causal_gqa8_Tensor_mV_t* mV,
    ffpa_d512_causal_gqa8_Tensor_mO_t* mO, float softmax_scale, int32_t num_kv_heads, cudaStream_t stream)
{
    int32_t ret;
    void* args[8] = {mQ, mK, mV, mO, &softmax_scale, &num_kv_heads, &stream, &ret};
    _mlir_ffpa_d512_causal_gqa8__mlir_ciface_cutlass___call_____main__FFPAFmhaAmpere_object_at__Tensorgmemo512i64div512i64div5125121_Tensorgmemo512i64div512i64div5125121_Tensorgmemo512i64div512i64div5125121_Tensorgmemo512i64d(
        args, 8);
    return ret;
}
