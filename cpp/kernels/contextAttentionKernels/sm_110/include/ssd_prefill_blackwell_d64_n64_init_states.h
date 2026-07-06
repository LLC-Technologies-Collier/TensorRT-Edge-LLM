
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
} ssd_prefill_blackwell_d64_n64_init_states_Kernel_Module_t;

#ifdef __cplusplus
extern "C"
{
#endif
    void _mlir_ssd_prefill_blackwell_d64_n64_init_states_cuda_init(void**);
    void _mlir_ssd_prefill_blackwell_d64_n64_init_states_cuda_load_to_device(void**);
    static inline void ssd_prefill_blackwell_d64_n64_init_states_Kernel_Module_Load(
        ssd_prefill_blackwell_d64_n64_init_states_Kernel_Module_t* module)
    {
        cudaLibrary_t* libraryPtr = &(module->module);
        cudaError_t ret;
        struct
        {
            cudaLibrary_t** libraryPtr;
            cudaError_t* ret;
        } initArgs = {&libraryPtr, &ret};
        _mlir_ssd_prefill_blackwell_d64_n64_init_states_cuda_init((void**) (&initArgs));
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
            _mlir_ssd_prefill_blackwell_d64_n64_init_states_cuda_load_to_device((void**) (&loadArgs));
            CUTE_DSL_CUDA_ERROR_CHECK(ret);
        }
    }

    static inline void ssd_prefill_blackwell_d64_n64_init_states_Kernel_Module_Unload(
        ssd_prefill_blackwell_d64_n64_init_states_Kernel_Module_t* module)
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
} ssd_prefill_blackwell_d64_n64_init_states_Tensor_x_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[3];
    int64_t dynamic_strides[2];
} ssd_prefill_blackwell_d64_n64_init_states_Tensor_dt_in_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[1];
} ssd_prefill_blackwell_d64_n64_init_states_Tensor_A_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[4];
    int64_t dynamic_strides[3];
} ssd_prefill_blackwell_d64_n64_init_states_Tensor_B_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[4];
    int64_t dynamic_strides[3];
} ssd_prefill_blackwell_d64_n64_init_states_Tensor_C_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[1];
} ssd_prefill_blackwell_d64_n64_init_states_Tensor_Dvec_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[1];
} ssd_prefill_blackwell_d64_n64_init_states_Tensor_dt_bias_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[4];
    int64_t dynamic_strides[3];
} ssd_prefill_blackwell_d64_n64_init_states_Tensor_output_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[4];
    int64_t dynamic_strides[3];
} ssd_prefill_blackwell_d64_n64_init_states_Tensor_state_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[4];
    int64_t dynamic_strides[3];
} ssd_prefill_blackwell_d64_n64_init_states_Tensor_dA_cumsum_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[4];
    int64_t dynamic_strides[3];
} ssd_prefill_blackwell_d64_n64_init_states_Tensor_dt_proc_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[5];
    int64_t dynamic_strides[4];
} ssd_prefill_blackwell_d64_n64_init_states_Tensor_y_ws_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[2];
    int64_t dynamic_strides[1];
} ssd_prefill_blackwell_d64_n64_init_states_Tensor_seq_idx_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[1];
} ssd_prefill_blackwell_d64_n64_init_states_Tensor_chunk_indices_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[1];
} ssd_prefill_blackwell_d64_n64_init_states_Tensor_chunk_offsets_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[1];
} ssd_prefill_blackwell_d64_n64_init_states_Tensor_seq_chunk_cumsum_t;

typedef struct
{
    void* data;
    int32_t dynamic_shapes[1];
} ssd_prefill_blackwell_d64_n64_init_states_Tensor_valid_lens_t;

#ifdef __cplusplus
extern "C"
#endif
    void
    _mlir_ssd_prefill_blackwell_d64_n64_init_states__mlir_ciface_cutlass_ssd_blackwell_aot_Tensorgmemoi64i64i641_Tensorgmemoi64i641_Tensorgmemo1_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemo1_Tensorgmemo1_Tensorgmemoi64i64i641_Tensorgm(
        void** args, int32_t num_args);

static inline int32_t cute_dsl_ssd_prefill_blackwell_d64_n64_init_states_wrapper(
    ssd_prefill_blackwell_d64_n64_init_states_Kernel_Module_t* module,
    ssd_prefill_blackwell_d64_n64_init_states_Tensor_x_t* x,
    ssd_prefill_blackwell_d64_n64_init_states_Tensor_dt_in_t* dt_in,
    ssd_prefill_blackwell_d64_n64_init_states_Tensor_A_t* A, ssd_prefill_blackwell_d64_n64_init_states_Tensor_B_t* B,
    ssd_prefill_blackwell_d64_n64_init_states_Tensor_C_t* C,
    ssd_prefill_blackwell_d64_n64_init_states_Tensor_Dvec_t* Dvec,
    ssd_prefill_blackwell_d64_n64_init_states_Tensor_dt_bias_t* dt_bias,
    ssd_prefill_blackwell_d64_n64_init_states_Tensor_output_t* output,
    ssd_prefill_blackwell_d64_n64_init_states_Tensor_state_t* state,
    ssd_prefill_blackwell_d64_n64_init_states_Tensor_dA_cumsum_t* dA_cumsum,
    ssd_prefill_blackwell_d64_n64_init_states_Tensor_dt_proc_t* dt_proc,
    ssd_prefill_blackwell_d64_n64_init_states_Tensor_y_ws_t* y_ws,
    ssd_prefill_blackwell_d64_n64_init_states_Tensor_seq_idx_t* seq_idx,
    ssd_prefill_blackwell_d64_n64_init_states_Tensor_chunk_indices_t* chunk_indices,
    ssd_prefill_blackwell_d64_n64_init_states_Tensor_chunk_offsets_t* chunk_offsets,
    ssd_prefill_blackwell_d64_n64_init_states_Tensor_seq_chunk_cumsum_t* seq_chunk_cumsum,
    ssd_prefill_blackwell_d64_n64_init_states_Tensor_valid_lens_t* valid_lens, int32_t seq_len_val, int32_t nchunks_val,
    int32_t num_logical_chunks_val, int32_t num_seqs_val, cudaStream_t stream)
{
    int32_t ret;
    void* args[23] = {x, dt_in, A, B, C, Dvec, dt_bias, output, state, dA_cumsum, dt_proc, y_ws, seq_idx, chunk_indices,
        chunk_offsets, seq_chunk_cumsum, valid_lens, &seq_len_val, &nchunks_val, &num_logical_chunks_val, &num_seqs_val,
        &stream, &ret};
    _mlir_ssd_prefill_blackwell_d64_n64_init_states__mlir_ciface_cutlass_ssd_blackwell_aot_Tensorgmemoi64i64i641_Tensorgmemoi64i641_Tensorgmemo1_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemo1_Tensorgmemo1_Tensorgmemoi64i64i641_Tensorgm(
        args, 23);
    return ret;
}
