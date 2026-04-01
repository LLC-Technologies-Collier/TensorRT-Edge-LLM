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

#include "applyRopeWriteKV.h"
#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "kernels/common/vectorizedTypes.cuh"

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>

namespace trt_edgellm
{
namespace kernel
{

template <typename T>
__device__ __forceinline__ T applyRope(T const& x, T const& y, float const& cos, float const& sin, bool const isLeft);

template <>
__device__ __forceinline__ half applyRope<half>(
    half const& x, half const& y, float const& cos, float const& sin, bool const isLeft)
{
    float val
        = isLeft ? (__half2float(x) * cos - __half2float(y) * sin) : (__half2float(x) * cos + __half2float(y) * sin);
    return __float2half(val);
}

template <>
__device__ __forceinline__ __nv_bfloat16 applyRope<__nv_bfloat16>(
    __nv_bfloat16 const& x, __nv_bfloat16 const& y, float const& cos, float const& sin, bool const isLeft)
{
    float val
        = isLeft ? (__bfloat162float(x) * cos - __bfloat162float(y) * sin) : (__bfloat162float(x) * cos + __bfloat162float(y) * sin);
    return __float2bfloat16(val);
}

template <typename T>
__device__ __forceinline__ DVec<T> vecApplyRopeNonInterleave(
    T const* dataPtr, DVec<float> const& cosVec, DVec<float> const& sinVec, uint32_t const rotaryDim)
{
    DVec<T> result;
    DVec<T> input;
    DVec<T> permuteInput;

    uint32_t const vecOffset = threadIdx.x * DVec<T>::vec_size;
    input.load(dataPtr + vecOffset);

    if (vecOffset < rotaryDim)
    {
        uint32_t const permuteOffset
            = (vecOffset < rotaryDim / 2) ? vecOffset + rotaryDim / 2 : vecOffset - rotaryDim / 2;
        permuteInput.load(dataPtr + permuteOffset);

#pragma unroll
        for (uint32_t i = 0; i < DVec<T>::vec_size; ++i)
        {
            result[i] = applyRope(input[i], permuteInput[i], cosVec[i], sinVec[i], (vecOffset < rotaryDim / 2));
        }
        return result;
    }
    else
    {
        return input;
    }
}

template <typename T, typename TCache>
__device__ __forceinline__ void storeVec(
    TCache* dst, int base, DVec<T> const& vec, float const* const scaleQuantOrig)
{
    if constexpr (std::is_same_v<TCache, T>)
    {
        // Save directly to dst
        vec.store(dst + base);
    }
#if SUPPORTS_FP8
    else if constexpr (std::is_same_v<TCache, __nv_fp8_e4m3>)
    {
        // Quantize and store to dst.
        DVec<__nv_fp8_e4m3> out;
        float const invScale = (scaleQuantOrig != nullptr) ? (1.0f / scaleQuantOrig[0]) : 1.0f;
#pragma unroll
        for (uint32_t i = 0; i < DVec<T>::vec_size; ++i)
        {
            float const scaled = static_cast<float>(vec[i]) * invScale;
            out[i] = __nv_fp8_e4m3(scaled);
        }
        out.store(dst + base);
    }
#endif
}

template <typename T, typename TCache>
__global__ void applyRopeWriteKV(T* q, T* k, T const* v, TCache* kvCache, float const* cosSinCache,
    int32_t const* kvCacheEndLens, int32_t const* tokenPosIds, float const* kvScaleQuantOrig, int32_t qSeqLen,
    int32_t totalNumTokens, int32_t kvCacheCapacity, uint32_t numQHead, uint32_t numKVHead, uint32_t headDim,
    uint32_t rotaryDim, int32_t cosSinCacheBatchSize, int32_t cosSinCacheSeqLen, bool writeKInPlace)
{
    uint32_t const bIdx = blockIdx.x;
    uint32_t const bIdy = blockIdx.y;
    uint32_t const tIdx = threadIdx.x;
    uint32_t const tIdy = threadIdx.y;

    uint32_t const bDimY = blockDim.y;
    uint32_t const tokenIdx = bIdx * bDimY + tIdy;
    if (tokenIdx >= totalNumTokens)
    {
        return;
    }

    int32_t const batchIdx = tokenIdx / qSeqLen;

    int32_t sinCosCachePos{};
    bool const isPaddingToken = (tokenPosIds != nullptr && tokenPosIds[tokenIdx] == -1);
    if (tokenPosIds != nullptr)
    {
        sinCosCachePos = tokenPosIds[tokenIdx];
        if (sinCosCachePos < 0)
        {
            sinCosCachePos = 0;
        }
    }
    else
    {
        int32_t const posStartId = kvCacheEndLens != nullptr ? kvCacheEndLens[batchIdx] - qSeqLen : 0;
        sinCosCachePos = posStartId + tokenIdx % qSeqLen;
    }

    uint32_t const sinOffset = rotaryDim / 2;
    uint32_t cosOffset;
    DVec<float> cosVec;
    DVec<float> sinVec;
    cosOffset = (tIdx * DVec<float>::vec_size) % (rotaryDim / 2);
    int32_t const cosSinCacheBatchIdx = (cosSinCacheBatchSize == 1) ? 0 : batchIdx;
    int32_t const cosSinCacheOffset = cosSinCacheBatchIdx * cosSinCacheSeqLen * rotaryDim + sinCosCachePos * rotaryDim;
    cosVec.load(cosSinCache + cosSinCacheOffset + cosOffset);
    sinVec.load(cosSinCache + cosSinCacheOffset + (cosOffset + sinOffset));

    if (bIdy < numQHead)
    {
        int32_t const qHeadIdx = bIdy;
        int32_t const qOffset = tokenIdx * numQHead * headDim + qHeadIdx * headDim;
        T* qPtr = q + qOffset;
        DVec<T> qRoped;

        if (isPaddingToken)
        {
#pragma unroll
            for (uint32_t i = 0; i < DVec<T>::vec_size; ++i)
            {
                qRoped[i] = T(0);
            }
        }
        else
        {
            qRoped = vecApplyRopeNonInterleave(qPtr, cosVec, sinVec, rotaryDim);
        }
        qRoped.store(qPtr + DVec<T>::vec_size * tIdx);
    }
    else
    {
        int32_t const kvHeadIdx = bIdy - numQHead;
        int32_t const kvOffset = tokenIdx * numKVHead * headDim + kvHeadIdx * headDim;
        T* kPtr = k + kvOffset;
        T const* vPtr = v + kvOffset;

        int32_t const kvCacheStartIdx = kvCacheEndLens != nullptr ? kvCacheEndLens[batchIdx] - qSeqLen : 0;
        int32_t const tokenIdxInCache = kvCacheStartIdx + tokenIdx % qSeqLen;
        int32_t const cacheOffsetSequence = batchIdx * 2 * numKVHead * kvCacheCapacity * headDim;

        DVec<T> kRoped;
        kRoped = vecApplyRopeNonInterleave(kPtr, cosVec, sinVec, rotaryDim);

        if (writeKInPlace)
        {
            kRoped.store(kPtr + DVec<T>::vec_size * tIdx);
        }

        DVec<T> vSrc;
        vSrc.load(vPtr + DVec<T>::vec_size * tIdx);

        if (!isPaddingToken)
        {
            int32_t cacheOffsetK = cacheOffsetSequence + kvHeadIdx * kvCacheCapacity * headDim
                + tokenIdxInCache * headDim + DVec<T>::vec_size * tIdx;
            int32_t cacheOffsetV = cacheOffsetSequence + (numKVHead + kvHeadIdx) * kvCacheCapacity * headDim
                + tokenIdxInCache * headDim + DVec<T>::vec_size * tIdx;
            float const* kScaleQuantOrig = kvScaleQuantOrig;
            float const* vScaleQuantOrig = (kvScaleQuantOrig != nullptr) ? (kvScaleQuantOrig + 1) : nullptr;
            storeVec<T, TCache>(kvCache, cacheOffsetK, kRoped, kScaleQuantOrig);
            storeVec<T, TCache>(kvCache, cacheOffsetV, vSrc, vScaleQuantOrig);
        }
    }
}

static void launchApplyRopeWriteKVKernel(rt::Tensor& q, rt::Tensor& k, rt::Tensor const& v, rt::Tensor& kvCache,
    rt::Tensor const& cosSinCache, rt::OptionalInputTensor kvCacheEndLens, rt::OptionalInputTensor tokenPosIds,
    rt::Tensor const& kvScaleQuantOrig, cudaStream_t stream, bool writeKInPlace)
{
    auto const dt = kvCache.getDataType();
    auto const q_dt = q.getDataType();

    uint32_t const runtimeBatchSize = static_cast<uint32_t>(q.getShape()[0]);
    uint32_t const runtimeSeqLen = static_cast<uint32_t>(q.getShape()[1]);
    uint32_t const numQHeads = static_cast<uint32_t>(q.getShape()[2]);
    uint32_t const headDim = static_cast<uint32_t>(q.getShape()[3]);
    uint32_t const numKVHeads = static_cast<uint32_t>(kvCache.getShape()[2]);
    uint32_t const kvCacheCapacity = static_cast<uint32_t>(kvCache.getShape()[3]);
    uint32_t const totalNumTokens = runtimeBatchSize * runtimeSeqLen;

    uint32_t const cosSinCacheBatchSize = static_cast<uint32_t>(cosSinCache.getShape()[0]);
    uint32_t const cosSinCacheSeqLen = static_cast<uint32_t>(cosSinCache.getShape()[1]);
    uint32_t const rotaryDim = static_cast<uint32_t>(cosSinCache.getShape()[2]);

    float const* kvScaleQuantOrigPtr = kvScaleQuantOrig.isEmpty() ? nullptr : kvScaleQuantOrig.dataPointer<float>();
    float const* cosSinCachePtr = cosSinCache.dataPointer<float>();

    int32_t const* kvCacheEndLensPtr
        = kvCacheEndLens.has_value() ? kvCacheEndLens.value().get().dataPointer<int32_t>() : nullptr;
    int32_t const* tokenPosIdsPtr
        = tokenPosIds.has_value() ? tokenPosIds.value().get().dataPointer<int32_t>() : nullptr;

    uint32_t const kVEC_SIZE = 8; 
    uint32_t const kTHREADS_PER_CTA = 128;
    uint32_t const tokenPerCTA = kTHREADS_PER_CTA * kVEC_SIZE / headDim;
    uint32_t const bDimX = headDim / kVEC_SIZE;
    uint32_t const bDimY = tokenPerCTA;
    uint32_t const gDimX = (totalNumTokens + tokenPerCTA - 1) / tokenPerCTA;
    uint32_t const gDimY = numQHeads + numKVHeads;

    dim3 grid(gDimX, gDimY);
    dim3 block(bDimX, bDimY);

    if (q_dt == nvinfer1::DataType::kHALF)
    {
        half* qPtr = q.dataPointer<half>();
        half* kPtr = k.dataPointer<half>();
        half const* vPtr = v.dataPointer<half>();

        if (dt == nvinfer1::DataType::kHALF)
        {
            applyRopeWriteKV<half, half><<<grid, block, 0, stream>>>(qPtr, kPtr, vPtr, kvCache.dataPointer<half>(), cosSinCachePtr,
                kvCacheEndLensPtr, tokenPosIdsPtr, kvScaleQuantOrigPtr, runtimeSeqLen, totalNumTokens, kvCacheCapacity,
                numQHeads, numKVHeads, headDim, rotaryDim, cosSinCacheBatchSize, cosSinCacheSeqLen, writeKInPlace);
        }
#if SUPPORTS_FP8
        else if (dt == nvinfer1::DataType::kFP8)
        {
            applyRopeWriteKV<half, __nv_fp8_e4m3><<<grid, block, 0, stream>>>(qPtr, kPtr, vPtr, kvCache.dataPointer<__nv_fp8_e4m3>(), cosSinCachePtr,
                kvCacheEndLensPtr, tokenPosIdsPtr, kvScaleQuantOrigPtr, runtimeSeqLen, totalNumTokens, kvCacheCapacity,
                numQHeads, numKVHeads, headDim, rotaryDim, cosSinCacheBatchSize, cosSinCacheSeqLen, writeKInPlace);
        }
#endif
    }
    else if (q_dt == nvinfer1::DataType::kBF16)
    {
        __nv_bfloat16* qPtr = q.dataPointer<__nv_bfloat16>();
        __nv_bfloat16* kPtr = k.dataPointer<__nv_bfloat16>();
        __nv_bfloat16 const* vPtr = v.dataPointer<__nv_bfloat16>();

        if (dt == nvinfer1::DataType::kBF16)
        {
            applyRopeWriteKV<__nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(qPtr, kPtr, vPtr, kvCache.dataPointer<__nv_bfloat16>(), cosSinCachePtr,
                kvCacheEndLensPtr, tokenPosIdsPtr, kvScaleQuantOrigPtr, runtimeSeqLen, totalNumTokens, kvCacheCapacity,
                numQHeads, numKVHeads, headDim, rotaryDim, cosSinCacheBatchSize, cosSinCacheSeqLen, writeKInPlace);
        }
    }
}

void launchApplyRopeWriteKV(rt::Tensor const& cosSinCache, rt::OptionalInputTensor kvCacheEndLens, rt::Tensor& q,
    rt::Tensor& k, rt::Tensor const& v, rt::Tensor& kvCache, rt::Tensor const& kvScaleQuantOrig, cudaStream_t stream,
    bool writeKInPlace)
{
    rt::OptionalInputTensor tokenPosIds{std::nullopt};
    launchApplyRopeWriteKVKernel(q, k, v, kvCache, cosSinCache, kvCacheEndLens, tokenPosIds, kvScaleQuantOrig, stream, writeKInPlace);
}

void launchApplyRopeWriteKVTreeDecoding(rt::Tensor const& cosSinCache, rt::Tensor const& kvCacheEndLens,
    rt::Tensor const& tokenPosIds, rt::Tensor& q, rt::Tensor& k, rt::Tensor const& v, rt::Tensor& kvCache,
    rt::Tensor const& kvScaleQuantOrig, cudaStream_t stream)
{
    launchApplyRopeWriteKVKernel(q, k, v, kvCache, cosSinCache, kvCacheEndLens, tokenPosIds, kvScaleQuantOrig, stream, false);
}

template <typename T, typename TCache>
__global__ void applyRopeWriteKVSplitQKVKernel(T* __restrict__ q, T const* __restrict__ k, T const* __restrict__ v,
    TCache* __restrict__ kvCache, float const* __restrict__ cosSinCache, int32_t const* __restrict__ kvCacheEndLens,
    float const* __restrict__ kvScaleQuantOrig, int32_t qSeqLen, int32_t totalNumTokens, int32_t kvCacheCapacity,
    uint32_t numQHead, uint32_t numKVHead, uint32_t headDim, uint32_t rotaryDim, int32_t cosSinCacheBatchSize,
    int32_t cosSinCacheSeqLen)
{
    uint32_t const tIdx = threadIdx.x;
    uint32_t const tIdy = threadIdx.y;
    uint32_t const tokenIdx = blockIdx.x * blockDim.y + tIdy;

    if (tokenIdx >= totalNumTokens)
    {
        return;
    }

    int32_t const batchIdx = tokenIdx / qSeqLen;
    int32_t const posStartId = kvCacheEndLens[batchIdx] - qSeqLen;
    int32_t const sinCosCachePos = posStartId + tokenIdx % qSeqLen;

    uint32_t const sinOffset = rotaryDim / 2;
    uint32_t const cosOffset = (tIdx * DVec<float>::vec_size) % (rotaryDim / 2);
    int32_t const cosSinCacheBatchIdx = (cosSinCacheBatchSize == 1) ? 0 : batchIdx;
    int32_t const cosSinCacheOffset = cosSinCacheBatchIdx * cosSinCacheSeqLen * rotaryDim + sinCosCachePos * rotaryDim;
    DVec<float> cosVec;
    DVec<float> sinVec;
    cosVec.load(cosSinCache + cosSinCacheOffset + cosOffset);
    sinVec.load(cosSinCache + cosSinCacheOffset + cosOffset + sinOffset);

    uint32_t const headIdx = blockIdx.y;

    if (headIdx < numQHead)
    {
        int32_t const qOffset = tokenIdx * numQHead * headDim + headIdx * headDim;
        T* qPtr = q + qOffset;
        DVec<T> qRoped = vecApplyRopeNonInterleave(qPtr, cosVec, sinVec, rotaryDim);
        qRoped.store(qPtr + DVec<T>::vec_size * tIdx);
    }
    else
    {
        uint32_t const kvHeadIdx = headIdx - numQHead;
        int32_t const kvInputOffset = tokenIdx * numKVHead * headDim + kvHeadIdx * headDim;
        T const* kPtr = k + kvInputOffset;
        T const* vPtr = v + kvInputOffset;

        DVec<T> kRoped = vecApplyRopeNonInterleave(kPtr, cosVec, sinVec, rotaryDim);
        DVec<T> vSrc;
        vSrc.load(vPtr + DVec<T>::vec_size * tIdx);

        int32_t const tokenIdxInCache = kvCacheEndLens[batchIdx] - qSeqLen + tokenIdx % qSeqLen;
        int64_t const cacheBase = static_cast<int64_t>(batchIdx) * 2 * numKVHead * kvCacheCapacity * headDim;
        int32_t const vecBase = DVec<T>::vec_size * tIdx;
        int64_t const cacheOffsetK = cacheBase + static_cast<int64_t>(kvHeadIdx) * kvCacheCapacity * headDim
            + tokenIdxInCache * headDim + vecBase;
        int64_t const cacheOffsetV = cacheBase + static_cast<int64_t>(numKVHead + kvHeadIdx) * kvCacheCapacity * headDim
            + tokenIdxInCache * headDim + vecBase;

        float const* kScaleQuantOrig = kvScaleQuantOrig;
        float const* vScaleQuantOrig = (kvScaleQuantOrig != nullptr) ? (kvScaleQuantOrig + 1) : nullptr;
        storeVec<T, TCache>(kvCache, cacheOffsetK, kRoped, kScaleQuantOrig);
        storeVec<T, TCache>(kvCache, cacheOffsetV, vSrc, vScaleQuantOrig);
    }
}

void launchApplyRopeWriteKVSplitQKV(rt::Tensor const& cosSinCache, rt::Tensor const& kvCacheEndLens, rt::Tensor& q,
    rt::Tensor const& k, rt::Tensor const& v, rt::Tensor& kvCache, rt::Tensor const& kvScaleQuantOrig,
    cudaStream_t stream)
{
    auto const dt = kvCache.getDataType();
    auto const q_dt = q.getDataType();

    uint32_t const runtimeBatchSize = static_cast<uint32_t>(q.getShape()[0]);
    uint32_t const runtimeSeqLen = static_cast<uint32_t>(q.getShape()[1]);
    uint32_t const numQHeads = static_cast<uint32_t>(q.getShape()[2]);
    uint32_t const headDim = static_cast<uint32_t>(q.getShape()[3]);
    uint32_t const numKVHeads = static_cast<uint32_t>(kvCache.getShape()[2]);
    uint32_t const kvCacheCapacity = static_cast<uint32_t>(kvCache.getShape()[3]);
    uint32_t const totalNumTokens = runtimeBatchSize * runtimeSeqLen;

    uint32_t const cosSinCacheBatchSize = static_cast<uint32_t>(cosSinCache.getShape()[0]);
    uint32_t const cosSinCacheSeqLen = static_cast<uint32_t>(cosSinCache.getShape()[1]);
    uint32_t const rotaryDim = static_cast<uint32_t>(cosSinCache.getShape()[2]);

    float const* cosSinCachePtr = cosSinCache.dataPointer<float>();
    int32_t const* kvCacheEndLensPtr = kvCacheEndLens.dataPointer<int32_t>();
    float const* kvScaleQuantOrigPtr = kvScaleQuantOrig.isEmpty() ? nullptr : kvScaleQuantOrig.dataPointer<float>();

    uint32_t const kVEC_SIZE = 8; 
    uint32_t const kTHREADS_PER_CTA = 128;
    uint32_t const tokenPerCTA = kTHREADS_PER_CTA * kVEC_SIZE / headDim;
    uint32_t const bDimX = headDim / kVEC_SIZE;
    uint32_t const bDimY = tokenPerCTA;
    uint32_t const gDimX = (totalNumTokens + tokenPerCTA - 1) / tokenPerCTA;
    uint32_t const gDimY = numQHeads + numKVHeads;

    dim3 grid(gDimX, gDimY);
    dim3 block(bDimX, bDimY);

    if (q_dt == nvinfer1::DataType::kHALF)
    {
        if (dt == nvinfer1::DataType::kHALF)
        {
            applyRopeWriteKVSplitQKVKernel<half, half><<<grid, block, 0, stream>>>(q.dataPointer<half>(), k.dataPointer<half>(), v.dataPointer<half>(), kvCache.dataPointer<half>(),
                cosSinCachePtr, kvCacheEndLensPtr, kvScaleQuantOrigPtr, runtimeSeqLen, totalNumTokens, kvCacheCapacity,
                numQHeads, numKVHeads, headDim, rotaryDim, cosSinCacheBatchSize, cosSinCacheSeqLen);
        }
#if SUPPORTS_FP8
        else if (dt == nvinfer1::DataType::kFP8)
        {
            applyRopeWriteKVSplitQKVKernel<half, __nv_fp8_e4m3><<<grid, block, 0, stream>>>(q.dataPointer<half>(), k.dataPointer<half>(), v.dataPointer<half>(), kvCache.dataPointer<__nv_fp8_e4m3>(),
                cosSinCachePtr, kvCacheEndLensPtr, kvScaleQuantOrigPtr, runtimeSeqLen, totalNumTokens, kvCacheCapacity,
                numQHeads, numKVHeads, headDim, rotaryDim, cosSinCacheBatchSize, cosSinCacheSeqLen);
        }
#endif
    }
    else if (q_dt == nvinfer1::DataType::kBF16)
    {
        if (dt == nvinfer1::DataType::kBF16)
        {
            applyRopeWriteKVSplitQKVKernel<__nv_bfloat16, __nv_bfloat16><<<grid, block, 0, stream>>>(q.dataPointer<__nv_bfloat16>(), k.dataPointer<__nv_bfloat16>(), v.dataPointer<__nv_bfloat16>(), kvCache.dataPointer<__nv_bfloat16>(),
                cosSinCachePtr, kvCacheEndLensPtr, kvScaleQuantOrigPtr, runtimeSeqLen, totalNumTokens, kvCacheCapacity,
                numQHeads, numKVHeads, headDim, rotaryDim, cosSinCacheBatchSize, cosSinCacheSeqLen);
        }
    }
}

} // namespace kernel
} // namespace trt_edgellm
