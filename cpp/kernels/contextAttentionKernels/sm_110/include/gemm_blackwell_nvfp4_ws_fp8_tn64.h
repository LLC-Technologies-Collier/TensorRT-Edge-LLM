
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
} gemm_blackwell_nvfp4_ws_fp8_tn64_Kernel_Module_t;

#ifdef __cplusplus
extern "C"
{
#endif
    void _mlir_gemm_blackwell_nvfp4_ws_fp8_tn64_cuda_init(void**);
    void _mlir_gemm_blackwell_nvfp4_ws_fp8_tn64_cuda_load_to_device(void**);
    static inline void gemm_blackwell_nvfp4_ws_fp8_tn64_Kernel_Module_Load(
        gemm_blackwell_nvfp4_ws_fp8_tn64_Kernel_Module_t* module)
    {
        cudaLibrary_t* libraryPtr = &(module->module);
        cudaError_t ret;
        struct
        {
            cudaLibrary_t** libraryPtr;
            cudaError_t* ret;
        } initArgs = {&libraryPtr, &ret};
        _mlir_gemm_blackwell_nvfp4_ws_fp8_tn64_cuda_init((void**) (&initArgs));
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
            _mlir_gemm_blackwell_nvfp4_ws_fp8_tn64_cuda_load_to_device((void**) (&loadArgs));
            CUTE_DSL_CUDA_ERROR_CHECK(ret);
        }
    }

    static inline void gemm_blackwell_nvfp4_ws_fp8_tn64_Kernel_Module_Unload(
        gemm_blackwell_nvfp4_ws_fp8_tn64_Kernel_Module_t* module)
    {
        CUTE_DSL_CUDA_ERROR_CHECK(cudaLibraryUnload(module->module));
    }

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"
#endif
    void
    _mlir_gemm_blackwell_nvfp4_ws_fp8_tn64__mlir_ciface_cutlass_wrapper___main__GemmBlackwellNvFp4WS_object_at__Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_128_256_128_16_CUstream0x0(
        void** args, int32_t num_args);

static inline int32_t cute_dsl_gemm_blackwell_nvfp4_ws_fp8_tn64_wrapper(
    gemm_blackwell_nvfp4_ws_fp8_tn64_Kernel_Module_t* module, void* a_ptr, void* b_ptr, void* sfa_ptr, void* sfb_ptr,
    void* c_ptr, int64_t m, int64_t n, int64_t k, cudaStream_t stream)
{
    int32_t ret;
    void* args[10] = {&a_ptr, &b_ptr, &sfa_ptr, &sfb_ptr, &c_ptr, &m, &n, &k, &stream, &ret};
    _mlir_gemm_blackwell_nvfp4_ws_fp8_tn64__mlir_ciface_cutlass_wrapper___main__GemmBlackwellNvFp4WS_object_at__Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_128_256_128_16_CUstream0x0(
        args, 10);
    return ret;
}
