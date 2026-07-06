
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
} nvfp4_moe_sm110_fc2_n128_fp16_Kernel_Module_t;

#ifdef __cplusplus
extern "C"
{
#endif
    void _mlir_nvfp4_moe_sm110_fc2_n128_fp16_cuda_init(void**);
    void _mlir_nvfp4_moe_sm110_fc2_n128_fp16_cuda_load_to_device(void**);
    static inline void nvfp4_moe_sm110_fc2_n128_fp16_Kernel_Module_Load(
        nvfp4_moe_sm110_fc2_n128_fp16_Kernel_Module_t* module)
    {
        cudaLibrary_t* libraryPtr = &(module->module);
        cudaError_t ret;
        struct
        {
            cudaLibrary_t** libraryPtr;
            cudaError_t* ret;
        } initArgs = {&libraryPtr, &ret};
        _mlir_nvfp4_moe_sm110_fc2_n128_fp16_cuda_init((void**) (&initArgs));
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
            _mlir_nvfp4_moe_sm110_fc2_n128_fp16_cuda_load_to_device((void**) (&loadArgs));
            CUTE_DSL_CUDA_ERROR_CHECK(ret);
        }
    }

    static inline void nvfp4_moe_sm110_fc2_n128_fp16_Kernel_Module_Unload(
        nvfp4_moe_sm110_fc2_n128_fp16_Kernel_Module_t* module)
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
    _mlir_nvfp4_moe_sm110_fc2_n128_fp16__mlir_ciface_cutlass_single_b_wrapper_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_16384_256_128_128_128_8_128_16_20_CUstream0x0(
        void** args, int32_t num_args);

static inline int32_t cute_dsl_nvfp4_moe_sm110_fc2_n128_fp16_wrapper(
    nvfp4_moe_sm110_fc2_n128_fp16_Kernel_Module_t* module, void* a_ptr, void* b_ptr, void* a_sf_ptr, void* b_sf_ptr,
    void* c_ptr, void* alpha_ptr, void* down_input_scale_ptr, void* tile_idx_to_group_idx_ptr,
    void* tile_idx_to_mn_limit_ptr, void* permuted_idx_to_expanded_idx_ptr, void* num_non_exiting_tiles_ptr,
    void* token_final_scales_ptr, int64_t m, int64_t n, int64_t k, int64_t l, int64_t num_tokens, int64_t top_k,
    cudaStream_t stream)
{
    int32_t ret;
    void* args[20] = {&a_ptr, &b_ptr, &a_sf_ptr, &b_sf_ptr, &c_ptr, &alpha_ptr, &down_input_scale_ptr,
        &tile_idx_to_group_idx_ptr, &tile_idx_to_mn_limit_ptr, &permuted_idx_to_expanded_idx_ptr,
        &num_non_exiting_tiles_ptr, &token_final_scales_ptr, &m, &n, &k, &l, &num_tokens, &top_k, &stream, &ret};
    _mlir_nvfp4_moe_sm110_fc2_n128_fp16__mlir_ciface_cutlass_single_b_wrapper_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_16384_256_128_128_128_8_128_16_20_CUstream0x0(
        args, 20);
    return ret;
}
