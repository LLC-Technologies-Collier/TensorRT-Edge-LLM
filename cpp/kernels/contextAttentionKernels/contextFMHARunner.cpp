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

#include "contextFMHARunner.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "cubin/fmha_cubin.h"
#include "fmhaParams_v2.h"

#include <cstring>
#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <memory>
#include <mutex>
#include <unordered_map>

using namespace nvinfer1;
using namespace trt_edgellm;

using FMHADataType = fmha_v2::Data_type;

namespace
{
union __half2_uint32_t_union
{
    half2 fp162;
    uint32_t u32;
};

union __float_uint32_t_union
{
    float fp32;
    uint32_t u32;
};

//! @throws std::runtime_error if alpha value has an unsupported type
static inline void set_alpha(uint32_t& alpha, float norm, FMHADataType dtype)
{
    if (dtype == FMHADataType::DATA_TYPE_FP16)
    {
        // Convert the float value into two fp16 value and pack into the uint32_t buffer.
        __half2_uint32_t_union temp;
        temp.fp162 = __float2half2_rn(norm);
        alpha = temp.u32;
    }
    else if (dtype == FMHADataType::DATA_TYPE_FP32)
    {
        __float_uint32_t_union temp;
        temp.fp32 = norm;
        alpha = temp.u32;
    }
    else if (dtype == FMHADataType::DATA_TYPE_INT32)
    {
        int32_t inorm = static_cast<int32_t>(norm);
        std::memcpy(&alpha, &inorm, sizeof(alpha));
    }
    else if (dtype == FMHADataType::DATA_TYPE_BF16)
    {
        // TODO HACK!! BF16 Outputs are computed in FP32 for FP8.
        // This is because cublas does not allow current FP32 output.
        std::memcpy(&alpha, &norm, sizeof(alpha));
    }
    else
    {
        check::check(false, "Unsupported type for alpha value");
    }
}

//! @throws std::runtime_error if FMHA datatype is unsupported
FMHADataType trtToFMHADataType(nvinfer1::DataType type)
{
    FMHADataType fmhaType{FMHADataType::DATA_TYPE_FP16};
    switch (type)
    {
    case nvinfer1::DataType::kFLOAT: fmhaType = FMHADataType::DATA_TYPE_FP32; break;
    case nvinfer1::DataType::kHALF: fmhaType = FMHADataType::DATA_TYPE_FP16; break;
    case nvinfer1::DataType::kBF16: fmhaType = FMHADataType::DATA_TYPE_BF16; break;
    case nvinfer1::DataType::kFP8: fmhaType = FMHADataType::DATA_TYPE_E4M3; break;
    default: throw std::runtime_error("Unsupported datatype for FMHA_v2.");
    }
    return fmhaType;
}

int32_t attentionMaskTypeToInt(ContextAttentionMaskType type) noexcept
{
    int32_t result{};
    switch (type)
    {
    case ContextAttentionMaskType::PADDING: result = 0; break;
    case ContextAttentionMaskType::CAUSAL: result = 1; break;
    case ContextAttentionMaskType::SLIDING_OR_CHUNKED_CAUSAL: result = 2; break;
    case ContextAttentionMaskType::CUSTOM_MASK: result = 3; break;
    }
    return result;
}

int32_t attentionInputLayoutToInt(AttentionInputLayout layout) noexcept
{
    int32_t result{};
    switch (layout)
    {
    case AttentionInputLayout::PACKED_QKV: result = 0; break;
    case AttentionInputLayout::CONTIGUOUS_Q_KV: result = 1; break;
    case AttentionInputLayout::Q_PAGED_KV: result = 2; break;
    case AttentionInputLayout::SEPARATE_Q_K_V: result = 3; break;
    }
    return result;
}

struct FMHAKernelLoadHashKey
{
    FMHADataType data_type;
    int32_t sm;

    bool operator==(FMHAKernelLoadHashKey const& other) const noexcept
    {
        return data_type == other.data_type && sm == other.sm;
    }
};

struct FMHAKernelLoadHasher
{
    size_t operator()(FMHAKernelLoadHashKey const& s) const noexcept
    {
        size_t key = s.data_type;
        key <<= 16;
        key ^= s.sm;
        return key;
    }
};

struct FMHAKernelHashKey
{
    FMHADataType data_type;
    int32_t sequenceLen;
    int32_t headSize;
    int32_t stepQ;
    int32_t stepKV;
    bool unroll;
    bool force_fp32_acc;
    bool flash_attention;
    int32_t attention_mask_type;
    bool tiled;
    int32_t attention_input_layout;
    bool interleaved;
    bool warp_specialization;
    bool alibi_supported;

    bool operator==(FMHAKernelHashKey const& other) const noexcept
    {
        // Flash attention kernel supports any sequence length. So for this set of kernel, we will match any sequence
        // length.
        bool seqMatch = (sequenceLen == other.sequenceLen);
        if (flash_attention && other.flash_attention) seqMatch = true;

        return data_type == other.data_type && seqMatch
            && headSize == other.headSize && unroll == other.unroll && force_fp32_acc == other.force_fp32_acc
            && flash_attention == other.flash_attention && attention_mask_type == other.attention_mask_type
            && tiled == other.tiled && attention_input_layout == other.attention_input_layout
            && interleaved == other.interleaved && warp_specialization == other.warp_specialization
            && alibi_supported == other.alibi_supported && stepQ == other.stepQ && stepKV == other.stepKV;
    }
};

struct FMHAKernelHasher
{
    size_t operator()(FMHAKernelHashKey const& hashKey) const noexcept
    {
        // flash attention support unlimited-sequence length. 
        // We MUST use s=0 for ALL flash attention kernels to match wildcard behavior.
        int32_t s = hashKey.flash_attention ? 0 : hashKey.sequenceLen;
        
        size_t key = (size_t)hashKey.data_type;
        key <<= 16;
        key ^= (size_t)s;
        key <<= 16;
        key ^= (size_t)hashKey.headSize;
        key <<= 1;
        key ^= (size_t)hashKey.unroll;
        key <<= 1;
        key ^= (size_t)hashKey.force_fp32_acc;
        key <<= 1;
        key ^= (size_t)hashKey.flash_attention;
        key <<= 3;
        key ^= (size_t)hashKey.attention_mask_type;
        key <<= 1;
        key ^= (size_t)hashKey.tiled;
        key <<= 3;
        key ^= (size_t)hashKey.attention_input_layout;
        key <<= 1;
        key ^= (size_t)hashKey.interleaved;
        key <<= 1;
        key ^= (size_t)hashKey.warp_specialization;
        key <<= 1;
        key ^= (size_t)hashKey.alibi_supported;
        key <<= 8;
        key ^= (size_t)hashKey.stepQ;
        key <<= 8;
        key ^= (size_t)hashKey.stepKV;
        
        return key;
    }
};

struct FMHAKernelFuncInfo
{
    uint32_t mThreadsPerCTA;
    uint32_t mUnrollStep;
    uint32_t mStepQ{0};
    uint32_t mStepKV{0};
    uint32_t mSharedMemBytes{0};
    CUfunction mDeviceFunction{0};
    std::string mFuncName{};
};

class FMHAKernelList
{
    using TKernelMetaInfo = fmha_v2::FusedMultiHeadAttentionKernelMetaInfoV2;

public:
    FMHAKernelList(FMHADataType type, uint32_t sm)
        : mDataType(type)
        , mSMVersion(sm)
    {
        fprintf(stderr, "[DEBUG] FMHAKernelList constructor: DataType=%d, sm=%d\n", (int)type, (int)sm);
        mKernelMeta = &(fmha_v2::sMhaKernelMetaInfosV2[0]);








        mKernelMetaCount = sizeof(fmha_v2::sMhaKernelMetaInfosV2) / sizeof(fmha_v2::sMhaKernelMetaInfosV2[0]);
    }

    //! @throws std::runtime_error if a CUDA driver error occurs
    void loadFMHAKernels()
    {
        fprintf(stderr, "[DEBUG] loadFMHAKernels: sm=%d, dataType=%d, count=%d\n", (int)mSMVersion, (int)mDataType, (int)mKernelMetaCount);
        if (!mFunctions.empty())
        {
            fprintf(stderr, "[DEBUG] loadFMHAKernels: Map not empty, already loaded.\n");
            return;
        }
        for (int32_t i = 0; i < mKernelMetaCount; ++i)
        {
            auto const& kernelMeta = mKernelMeta[i];
            
            // Debug first few kernels
            if (i < 5) {
                fprintf(stderr, "[DEBUG] loadFMHAKernels[%d]: %s, SM=%d, In=%d, Out=%d\n", 
                    i, kernelMeta.mFuncName, (int)kernelMeta.mSM, (int)kernelMeta.mDataTypeIn, (int)kernelMeta.mDataTypeOut);
            }

            if (kernelMeta.mDataTypeIn != mDataType || kernelMeta.mDataTypeOut != mDataType
                || (int32_t)kernelMeta.mSM != (int32_t)mSMVersion || kernelMeta.mCubin == nullptr)
            {
                continue;
            }





            // load CUmodule. Each module can contain multiple kernel function.
            CUmodule hModule;
            auto findModuleIter = mModules.find(kernelMeta.mCubin);
            if (findModuleIter != mModules.end())
            {
                hModule = findModuleIter->second;
            }
            else
            {
                fprintf(stderr, "[DEBUG] loadFMHAKernels: Loading cubin for %s (SM=%d)\n", kernelMeta.mFuncName, kernelMeta.mSM);
                CUresult status = cuModuleLoadData(&hModule, kernelMeta.mCubin);

                if (status != CUDA_SUCCESS) {
                    const char* errStr;
                    cuGetErrorString(status, &errStr);
                    fprintf(stderr, "[FATAL] ContextFMHARunner::loadFMHAKernels: Failed to load cubin for %s (smVersion=%d, status=%d: %s)\n",
                            kernelMeta.mFuncName, (int)mSMVersion, (int)status, errStr);
                    throw std::runtime_error("CUDA driver API error in cuModuleLoadData");
                }
                fprintf(stderr, "[DEBUG] loadFMHAKernels: Successfully loaded module for %s\n", kernelMeta.mFuncName);
                mModules.insert(std::make_pair(kernelMeta.mCubin, hModule));
            }

            FMHAKernelFuncInfo funcInfo{};
            CUDA_DRIVER_CHECK(cuModuleGetFunction(&funcInfo.mDeviceFunction, hModule, kernelMeta.mFuncName));
            funcInfo.mSharedMemBytes = kernelMeta.mSharedMemBytes;
            funcInfo.mThreadsPerCTA = kernelMeta.mThreadsPerCTA;
            funcInfo.mUnrollStep = kernelMeta.mUnrollStep;
            funcInfo.mStepQ = kernelMeta.mStepQ;
            funcInfo.mStepKV = kernelMeta.mStepKV;
            funcInfo.mFuncName = std::string(kernelMeta.mFuncName);

            if (funcInfo.mSharedMemBytes >= 48 * 1024)
            {
                CUDA_DRIVER_CHECK(cuFuncSetAttribute(funcInfo.mDeviceFunction,
                    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, funcInfo.mSharedMemBytes));
            }
            FMHAKernelHashKey hashKey{kernelMeta.mDataTypeIn, static_cast<int32_t>(kernelMeta.mS),
                static_cast<int32_t>(kernelMeta.mD), static_cast<int32_t>(kernelMeta.mStepQ),
                static_cast<int32_t>(kernelMeta.mStepKV), kernelMeta.mUnrollStep != 0, kernelMeta.mFP32Accumulation,
                kernelMeta.mFlashAttention, kernelMeta.mAttentionMaskType, kernelMeta.mTiled,
                kernelMeta.mAttentionInputLayout, kernelMeta.mInterleaved, kernelMeta.mWarpSpecialization,
                kernelMeta.mAlibiSupported};
            
            if (mSMVersion == 101 && kernelMeta.mD == 128) {
                fprintf(stderr, "[DEBUG] loadFMHAKernels: Registered HashKey: "
                        "DataType=%d, S=%d, D=%d, stepQ=%d, stepKV=%d, Unroll=%d, FP32Acc=%d, Flash=%d, Mask=%d, Tiled=%d, Layout=%d, Inter=%d, Warp=%d, Alibi=%d, hash=%zu\n",
                        (int)kernelMeta.mDataTypeIn, (int)kernelMeta.mS, (int)kernelMeta.mD, (int)kernelMeta.mStepQ, (int)kernelMeta.mStepKV, (int)(kernelMeta.mUnrollStep != 0),
                        (int)kernelMeta.mFP32Accumulation, (int)kernelMeta.mFlashAttention, (int)kernelMeta.mAttentionMaskType, 
                        (int)kernelMeta.mTiled, (int)kernelMeta.mAttentionInputLayout, (int)kernelMeta.mInterleaved, 
                        (int)kernelMeta.mWarpSpecialization, (int)kernelMeta.mAlibiSupported, FMHAKernelHasher()(hashKey));
            }

            fprintf(stderr, "[DEBUG] loadFMHAKernels: Inserting function %s into mFunctions (hash=%zu)\n", 
                kernelMeta.mFuncName, FMHAKernelHasher()(hashKey));
            mFunctions.insert(std::make_pair(hashKey, funcInfo));

        }
    }

    FMHAKernelFuncInfo findKernelFunction(FMHAKernelHashKey const& key) const noexcept
    {
        fprintf(stderr, "[DEBUG] findKernelFunction: searching for Key: DataType=%d, sequenceLen=%d, HeadSize=%d, stepQ=%d, stepKV=%d, Unroll=%d, FP32Acc=%d, Flash=%d, Mask=%d, Tiling=%d, Layout=%d, Inter=%d, Warp=%d, Alibi=%d, hash=%zu\n",
            (int)key.data_type, (int)key.sequenceLen, (int)key.headSize, (int)key.stepQ, (int)key.stepKV, (int)key.unroll, (int)key.force_fp32_acc, (int)key.flash_attention,
            (int)key.attention_mask_type, (int)key.tiled, (int)key.attention_input_layout, (int)key.interleaved, (int)key.warp_specialization, (int)key.alibi_supported,
            FMHAKernelHasher()(key));
        auto const findIter = mFunctions.find(key);
        if (findIter == mFunctions.end())
        {
            // Return empty function info.
            return FMHAKernelFuncInfo{};
        }

        return findIter->second;
    }

    std::unordered_map<FMHAKernelHashKey, FMHAKernelFuncInfo, FMHAKernelHasher> const& getFunctions() const {
        return mFunctions;
    }

protected:
    TKernelMetaInfo const* mKernelMeta;
    int32_t mKernelMetaCount;
    FMHADataType mDataType;
    uint32_t mSMVersion;
    std::unordered_map<unsigned char const*, CUmodule> mModules;

    std::unordered_map<FMHAKernelHashKey, FMHAKernelFuncInfo, FMHAKernelHasher> mFunctions;
};

class FMHAKernelLoader
{

public:
    //! @throws std::runtime_error if a CUDA driver error occurs
    FMHAKernelList* getFMHAKernelList(FMHADataType type, int32_t sm)
    {
        static std::mutex s_mutex;
        std::lock_guard<std::mutex> lg(s_mutex);

        FMHAKernelLoadHashKey hash_key{type, sm};

        auto findIter = mKernels.find(hash_key);
        if (findIter == mKernels.end())
        {
            std::unique_ptr<FMHAKernelList> newKernel = std::make_unique<FMHAKernelList>(type, sm);
            newKernel->loadFMHAKernels();
            mKernels.insert(std::make_pair(hash_key, std::move(newKernel)));
            findIter = mKernels.find(hash_key);
        }
        return findIter->second.get();
    }

    static FMHAKernelLoader& Get()
    {
        static std::unique_ptr<FMHAKernelLoader> kernelLoader = nullptr;
        if (kernelLoader == nullptr)
        {
            kernelLoader = std::make_unique<FMHAKernelLoader>(FMHAKernelLoader());
        }

        return *kernelLoader;
    }

private:
    FMHAKernelLoader() = default;

    std::unordered_map<FMHAKernelLoadHashKey, std::unique_ptr<FMHAKernelList> const, FMHAKernelLoadHasher> mKernels;
};

//! @throws std::runtime_error if a CUDA driver error occurs
inline FMHAKernelList* getFMHAKernels(FMHADataType type, int32_t sm)
{
    return FMHAKernelLoader::Get().getFMHAKernelList(type, sm);
}

}; // namespace

ContextFMHARunner::ContextFMHARunner(nvinfer1::DataType const dataType, int32_t batchSize, int32_t paddedSeqLen,
    int32_t numQHeads, int32_t numKvHeads, int32_t headSize, int32_t smVersion, AttentionInputLayout inputLayout,
    ContextAttentionMaskType maskType, bool isSPadded)
    : mDataType(dataType)
    , mBatchSize(batchSize)
    , mPaddedSequenceLen(paddedSeqLen)
    , mOriginalPaddedSequenceLen(paddedSeqLen)
    , mNumHeads(numQHeads)
    , mNumKVHeads(numKvHeads)
    , mHeadSize(headSize)
    , mSmVersion(smVersion)
    , mIsSPadded(isSPadded)
{
    // The context FMHA-v2 kernels taken by the project only support ampere/ada for
    // reference on x86 machine, Orin/Thor for production on auto platforms.
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    mLaunchParams.multi_processor_count = props.multiProcessorCount;
    mLaunchParams.device_l2_cache_size = props.l2CacheSize;
    mLaunchParams.attention_mask_type = maskType;
    mLaunchParams.attention_input_layout = inputLayout;

    bool const isSm8x = (smVersion == fmha_v2::kSM_80 || smVersion == fmha_v2::kSM_86 || smVersion == fmha_v2::kSM_87
        || smVersion == fmha_v2::kSM_89);
    bool const isSm10x = (smVersion == fmha_v2::kSM_100 || smVersion == fmha_v2::kSM_101 || smVersion == 110);
    bool const isSm12x = (smVersion == fmha_v2::kSM_120 || smVersion == fmha_v2::kSM_121);

    check::check((isSm8x || isSm10x || isSm12x), "Other SMs are not supported by context FMHA-v2 kernels");
    // Handle kernel selection under different context.
    if (isSm8x || isSm10x || isSm12x)
    {
        // always use flash attention kernels for Ampere/Ada
        mLaunchParams.flash_attention = true;
        // flash attention kernles s = 0 (support any seq length)
        mLaunchParams.force_unroll = true;

        // TODO: Check if still proper for contiguous q-kv input layout
        if (paddedSeqLen <= 128 || mHeadSize < 256)

        {
            // flash attention tiled kernels allows larger free dim tile size (M, N) with flexibility
            // in unroll dimension tile size (K). for short sequence length (s<=128), tiled kernels
            // can suffer from tile quantization loss.
            // Also flash attention tiled kernel is generally faster when head_size>=256
            mLaunchParams.use_granular_tiling = false;
        }
        else
        {
            // otherwise, choose tiled FMHA-v2 flash-attention kernel.
            mLaunchParams.use_granular_tiling = true;
        }
    }
}

void ContextFMHARunner::setupParams(FusedMultiheadAttentionParamsV2& params)
{
    float const invSqrtScale = (1.f / sqrtf(mHeadSize));

    float const scale_bmm1 = invSqrtScale;
    float const scale_softmax = 1.f; // Seems to be only required for int8
    float const scale_bmm2 = 1.f;

    FMHADataType scale_type = mLaunchParams.force_fp32_acc ? fmha_v2::DATA_TYPE_FP32 : trtToFMHADataType(mDataType);
    set_alpha(params.scale_bmm1, scale_bmm1, scale_type);
    set_alpha(params.scale_softmax, scale_softmax, scale_type);
    set_alpha(params.scale_bmm2, scale_bmm2, scale_type);

    params.b = mBatchSize;
    params.h = mNumHeads;
    params.h_kv = mNumKVHeads;
    params.h_q_per_kv = mNumHeads / mNumKVHeads;
    // is_s_padded=true means Q/K/V use normal [B, S, H, D] layout and s is used for indexing.
    // Otherwise tensors are ragged (B x S compacted), cu_seqlens drives indexing, and s is not used.
    params.s = mOriginalPaddedSequenceLen;
    params.s_kv = mOriginalPaddedSequenceLen;
    params.d = mHeadSize;
    params.dv = mHeadSize;
    params.is_s_padded = mIsSPadded;

    params.o_stride_in_bytes = mNumHeads * mHeadSize * sizeof(half);

    check::check(mLaunchParams.attention_input_layout == AttentionInputLayout::SEPARATE_Q_K_V
            || mLaunchParams.attention_input_layout == AttentionInputLayout::CONTIGUOUS_Q_KV,
        "Unsupported input layout");
    if (mLaunchParams.attention_input_layout == AttentionInputLayout::SEPARATE_Q_K_V)
    {
        int64_t q_stride_in_bytes = mNumHeads * mHeadSize * sizeof(half);
        int64_t kv_stride_in_bytes = mNumKVHeads * mHeadSize * sizeof(half);
        params.q_stride_in_bytes = q_stride_in_bytes;
        params.k_stride_in_bytes = kv_stride_in_bytes;
        params.v_stride_in_bytes = kv_stride_in_bytes;
    }
    else
    {
        int64_t q_stride_in_bytes = mNumHeads * mHeadSize * sizeof(half);
        int64_t kv_stride_in_bytes = (2 * mNumKVHeads) * mHeadSize * sizeof(half);
        params.q_stride_in_bytes = q_stride_in_bytes;
        params.k_stride_in_bytes = kv_stride_in_bytes;
        params.v_stride_in_bytes = kv_stride_in_bytes;
    }
}

bool ContextFMHARunner::canImplement(int32_t headSize, [[maybe_unused]] int32_t sm, nvinfer1::DataType dataType,
    AttentionInputLayout inputLayout, ContextAttentionMaskType maskType) noexcept
{
    if (dataType != DataType::kHALF)
    {
        LOG_ERROR(
            "ContextFMHARunner::canImplement() only supports FP16. Got dataType=%d.", static_cast<int32_t>(dataType));
        return false;
    }

    if (headSize == 64 || headSize == 128)
    {
        return true;
    }

    // Head sizes 72/80 are only available for PADDING mask with SEPARATE_Q_K_V layout, used for VIT.
    if (headSize == 72 || headSize == 80)
    {
        if (inputLayout != AttentionInputLayout::SEPARATE_Q_K_V || maskType != ContextAttentionMaskType::PADDING)
        {
            LOG_ERROR(
                "ContextFMHARunner::canImplement() headSize=%d requires inputLayout=SEPARATE_Q_K_V and "
                "maskType=PADDING. Got inputLayout=%d, maskType=%d.",
                headSize, static_cast<int32_t>(inputLayout), static_cast<int32_t>(maskType));
            return false;
        }
        return true;
    }

    LOG_ERROR("ContextFMHARunner::canImplement() unsupported headSize=%d. Supported head sizes are 64, 72, 80, 128.",
        headSize);
    return false;
}

bool ContextFMHARunner::loadContextFMHAKernels(int32_t smVersion, nvinfer1::DataType dataType)
{
    int32_t searchSmVersion = smVersion;
    if (searchSmVersion == 110) searchSmVersion = 101;
    else if (searchSmVersion > 101 && searchSmVersion < 120) searchSmVersion = 101;
    else if (searchSmVersion >= 120) searchSmVersion = 120;
    else if (searchSmVersion >= 100 && searchSmVersion <= 101) searchSmVersion = searchSmVersion; // Keep native
    else if (searchSmVersion > 89 && searchSmVersion < 100) searchSmVersion = 89;


    fprintf(stderr, "[DEBUG] ContextFMHARunner::loadContextFMHAKernels: smVersion=%d, dataType=%d (searched as %d)\n", smVersion, (int)dataType, searchSmVersion);
    FMHAKernelList* fmhaKernelList = getFMHAKernels(trtToFMHADataType(dataType), searchSmVersion);

    return fmhaKernelList != nullptr;
}




void ContextFMHARunner::dispatchFMHAKernel(FusedMultiheadAttentionParamsV2& params, cudaStream_t const& stream)
{
    if (mLaunchParams.attention_input_layout == AttentionInputLayout::SEPARATE_Q_K_V)
    {
        check::check(params.q_ptr != nullptr && params.k_ptr != nullptr && params.v_ptr != nullptr
                && params.o_ptr != nullptr && params.cu_q_seqlens != nullptr && params.cu_kv_seqlens != nullptr,
            "Device pointers are supposed to be valid");
    }
    else // CONTIGUOUS_Q_KV
    {
        check::check(params.q_ptr != nullptr && params.kv_ptr != nullptr && params.o_ptr != nullptr
                && params.cu_q_seqlens != nullptr && params.cu_kv_seqlens != nullptr,
            "Device pointers are supposed to be valid");
    }
    int32_t layoutKey = attentionInputLayoutToInt(mLaunchParams.attention_input_layout);
    bool forceFp32Acc = mLaunchParams.force_fp32_acc;
    bool unrollKey = mLaunchParams.force_unroll;
    bool tiledKey = mLaunchParams.use_granular_tiling;

    // Q
    uint64_t total_s = static_cast<uint64_t>(params.s * mBatchSize);

    if (mLaunchParams.flash_attention) {
        unrollKey = true; // Flash kernels have mUnrollStep != 0

        // TMA alignment: Blackwell hardware swizzling REQUIRES tensor_size to be multiple of 8.
        // If the actual buffer (total_s) is not aligned, we MUST fall back to non-TMA kernels
        // (Tiled=0) which use standard global memory access to avoid IMA.
        if ((mSmVersion == 100 || mSmVersion == 101 || mSmVersion == 110) && (total_s % 8 != 0)) {
            fprintf(stderr, "[DEBUG] total_s=%lu is not multiple of 8, forcing non-TMA (Tiling=0) for Blackwell safety\n", total_s);
            fflush(stderr);
            tiledKey = false;
        }

        // Tiled kernels (64_128 or 128_128) use FP32 accumulation
        // Non-tiled kernels (64_32 or 64_64) use FP16 accumulation
        if (tiledKey) {
            forceFp32Acc = true;
        } else {
            forceFp32Acc = false;
        }

        // Blackwell kernels in fmha_cubin.h strictly use FP32Acc=1
        if (mSmVersion == 100 || mSmVersion == 101 || mSmVersion == 110) {
            forceFp32Acc = true;
        }
    }
        int32_t stepQ = 64;
        int32_t stepKV = 32;
        if (tiledKey) {
        stepQ = 128;
        stepKV = 128;
        }

        bool alibiKey = false;
        if (mSmVersion == 100 || mSmVersion == 101 || mSmVersion == 110) {
        alibiKey = true; // The log shows all registered sm101 have Alibi=1
        }

        FMHAKernelHashKey hashKey{trtToFMHADataType(mDataType), 
        mLaunchParams.flash_attention ? 0 : mPaddedSequenceLen, 
        mHeadSize, stepQ, stepKV, unrollKey,
        forceFp32Acc, mLaunchParams.flash_attention,
        attentionMaskTypeToInt(mLaunchParams.attention_mask_type), tiledKey,
        layoutKey, false, false, alibiKey};
    int32_t searchSmVersion = mSmVersion;


    FMHAKernelList* fmhaKernelList = getFMHAKernels(trtToFMHADataType(mDataType), searchSmVersion);





    fprintf(stderr, "[DEBUG] dispatchFMHAKernel: Searching for Key: DataType=%d, sequenceLen=%d, HeadSize=%d, stepQ=%d, stepKV=%d, Unroll=%d, FP32Acc=%d, Flash=%d, Mask=%d, Tiling=%d, Layout=%d, Inter=0, Warp=0, Alibi=%d, hash=%zu, searchedSm=%d\n",
        (int)trtToFMHADataType(mDataType), (int)hashKey.sequenceLen, mHeadSize, (int)stepQ, (int)stepKV, (int)unrollKey, (int)forceFp32Acc,
        (int)mLaunchParams.flash_attention, (int)mLaunchParams.attention_mask_type, (int)tiledKey,
        (int)layoutKey, (int)alibiKey, FMHAKernelHasher()(hashKey), (int)searchSmVersion);

    FMHAKernelFuncInfo kernelInfo = fmhaKernelList->findKernelFunction(hashKey);

    if (kernelInfo.mSharedMemBytes != 0) {
        fprintf(stderr, "[DEBUG] dispatchFMHAKernel: Found kernel %s (sharedMem=%d, threads=%d, unrollStep=%d, mSmVersion=%d)\n",
            kernelInfo.mFuncName.c_str(), kernelInfo.mSharedMemBytes, kernelInfo.mThreadsPerCTA, kernelInfo.mUnrollStep, (int)mSmVersion);
        fflush(stderr);
    }
    
    if (kernelInfo.mSharedMemBytes == 0) {
        fprintf(stderr, "[FATAL] ContextFMHARunner::dispatchFMHAKernel: Implementation NOT FOUND for HashKey: DataType=%d, sequenceLen=%d, HeadSize=%d, stepQ=%d, stepKV=%d, Unroll=%d, FP32Acc=%d, Flash=%d, Mask=%d, Tiling=%d, Layout=%d, Inter=0, Warp=0, Alibi=0, hash=%zu, smVersion=%d (searched as %d)\n",
                (int)trtToFMHADataType(mDataType), (int)hashKey.sequenceLen, mHeadSize, (int)hashKey.stepQ, (int)hashKey.stepKV, (int)unrollKey, (int)forceFp32Acc,
                (int)mLaunchParams.flash_attention, (int)mLaunchParams.attention_mask_type, (int)tiledKey, (int)layoutKey, FMHAKernelHasher()(hashKey), (int)mSmVersion, (int)searchSmVersion);
        
        fprintf(stderr, "[DEBUG] dispatchFMHAKernel: Available keys in this FMHAKernelList:\n");
        auto const& functions = fmhaKernelList->getFunctions();
        for (auto const& [key, info] : functions) {
            fprintf(stderr, "  - Key: DataType=%d, S=%d, D=%d, stepQ=%d, stepKV=%d, Unroll=%d, FP32Acc=%d, Flash=%d, Mask=%d, Tiled=%d, Layout=%d, Inter=%d, Warp=%d, Alibi=%d, hash=%zu -> %s\n",
                (int)key.data_type, (int)key.sequenceLen, (int)key.headSize, (int)key.stepQ, (int)key.stepKV, (int)key.unroll, (int)key.force_fp32_acc, (int)key.flash_attention,
                (int)key.attention_mask_type, (int)key.tiled, (int)key.attention_input_layout, (int)key.interleaved, (int)key.warp_specialization, (int)key.alibi_supported,
                FMHAKernelHasher()(key), info.mFuncName.c_str());
        }
        fflush(stderr);
        throw std::runtime_error("There must be one kernel to implement the MHA");
    }
    
    if ((mSmVersion == 100 || mSmVersion == 101 || mSmVersion == 110) && tiledKey) {
        int d = mHeadSize;
        int h = mNumHeads;
        int h_kv = mNumKVHeads;
        int d_bytes = d * sizeof(half);
        
        // TRT-LLM logic for splitting D into multiple groups to match TMA swizzle mode (128B)
        uint32_t const d_groups = d_bytes > 128 ? d_bytes / 128 : 1;
        uint32_t const d_bytes_per_group = d_bytes / d_groups;
        uint32_t const d_per_group = d / d_groups;
        
        CUtensorMapSwizzle const swizzle_mode = (d_bytes_per_group > 64
            ? CU_TENSOR_MAP_SWIZZLE_128B
            : (d_bytes_per_group > 32 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_32B));
        
        uint32_t elem_stride[3] = {1, 1, 1};
        
        // TMA alignment: Blackwell hardware swizzling REQUIRES tensor_size to be multiple of 8.
        // We pad aligned_total_s to multiple of 8 and use OOB_FILL to safely handle the tail.
        uint64_t aligned_total_s = ((total_s + 7) / 8) * 8;
        uint64_t tensor_size_q[3] = {static_cast<uint64_t>(d), static_cast<uint64_t>(h), aligned_total_s};
        uint64_t tensor_stride_q[2] = {static_cast<uint64_t>(d_bytes), static_cast<uint64_t>(params.q_stride_in_bytes)};
        // Box must be multiple of 8 for Blackwell.
        uint32_t box_s_q = ((stepQ + 7) / 8) * 8; 
        // Ensure box_s_q <= aligned_total_s to satisfy cuTensorMapEncodeTiled
        if (box_s_q > aligned_total_s) box_s_q = aligned_total_s;
        uint32_t box_size_q[3] = {static_cast<uint32_t>(d_per_group), 1, box_s_q};
        
        fprintf(stderr, "[DEBUG] cuTensorMapEncodeTiled Q: ptr=%p, size=[%lu, %lu, %lu] (real_s=%lu), stride=[%lu, %lu], box=[%u, %u, %u], swizzle=%d\n",
            params.q_ptr, tensor_size_q[0], tensor_size_q[1], tensor_size_q[2], total_s,
            tensor_stride_q[0], tensor_stride_q[1], box_size_q[0], box_size_q[1], box_size_q[2], (int)swizzle_mode);
        fflush(stderr);
            
        CUDA_DRIVER_CHECK(cuTensorMapEncodeTiled(
            reinterpret_cast<CUtensorMap*>(&params.tma_desc_q),
            CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            3,
            const_cast<void*>(params.q_ptr),
            tensor_size_q,
            tensor_stride_q,
            box_size_q,
            elem_stride,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            swizzle_mode,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
        ));
            
        // K & V 
        uint64_t tensor_size_kv[3] = {static_cast<uint64_t>(d), static_cast<uint64_t>(h_kv), aligned_total_s};
        uint64_t tensor_stride_kv[2] = {static_cast<uint64_t>(d_bytes), static_cast<uint64_t>(params.k_stride_in_bytes)};
        uint32_t box_s_kv = ((stepKV + 7) / 8) * 8;
        if (box_s_kv > aligned_total_s) box_s_kv = aligned_total_s;
        uint32_t box_size_kv[3] = {static_cast<uint32_t>(d_per_group), 1, box_s_kv};
        
        fprintf(stderr, "[DEBUG] cuTensorMapEncodeTiled K: ptr=%p, size=[%lu, %lu, %lu] (real_s=%lu), stride=[%lu, %lu], box=[%u, %u, %u], swizzle=%d\n",
            params.k_ptr, tensor_size_kv[0], tensor_size_kv[1], tensor_size_kv[2], total_s,
            tensor_stride_kv[0], tensor_stride_kv[1], box_size_kv[0], box_size_kv[1], box_size_kv[2], (int)swizzle_mode);
        fflush(stderr);

        CUDA_DRIVER_CHECK(cuTensorMapEncodeTiled(
            reinterpret_cast<CUtensorMap*>(&params.tma_desc_k),
            CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            3,
            const_cast<void*>(params.k_ptr),
            tensor_size_kv,
            tensor_stride_kv,
            box_size_kv,
            elem_stride,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            swizzle_mode,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
        ));
        
        fprintf(stderr, "[DEBUG] cuTensorMapEncodeTiled V: ptr=%p, size=[%lu, %lu, %lu] (real_s=%lu), stride=[%lu, %lu], box=[%u, %u, %u], swizzle=%d\n",
            params.v_ptr, tensor_size_kv[0], tensor_size_kv[1], tensor_size_kv[2], total_s,
            tensor_stride_kv[0], tensor_stride_kv[1], box_size_kv[0], box_size_kv[1], box_size_kv[2], (int)swizzle_mode);
        fflush(stderr);
            
        CUDA_DRIVER_CHECK(cuTensorMapEncodeTiled(
            reinterpret_cast<CUtensorMap*>(&params.tma_desc_v),
            CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            3,
            const_cast<void*>(params.v_ptr),
            tensor_size_kv, 
            tensor_stride_kv,
            box_size_kv,
            elem_stride,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            swizzle_mode,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
        ));
    }
    
    void* kernelParams[] = {&params, nullptr};
    // Right now we onlu use flash attention kernel
    // flash attention supports any sequence length (0 in kernel meta)
    int32_t unroll = (params.s + kernelInfo.mUnrollStep - 1) / kernelInfo.mUnrollStep;
    // on Ampere/Ada flash attention, we launch blocks (steps, h, b)
    // TODO: Generalize the logic for more architectures.
    
    fprintf(stderr, "[DEBUG] Pre cuLaunchKernel... \n");
    fflush(stderr);

    CUDA_DRIVER_CHECK(cuLaunchKernel(kernelInfo.mDeviceFunction, unroll, params.h, params.b, kernelInfo.mThreadsPerCTA,
        1, 1, kernelInfo.mSharedMemBytes, stream, kernelParams, nullptr));
        
    fprintf(stderr, "[DEBUG] cuLaunchKernel returned successfully, syncing stream... \n");
    fflush(stderr);
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    fprintf(stderr, "[DEBUG] Kernel execution completed successfully.\n");
    fflush(stderr);
}
