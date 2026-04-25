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

#include "attentionPlugin.h"

#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/tensor.h"
#include "kernels/contextAttentionKernels/contextFMHARunner.h"
#include "kernels/contextAttentionKernels/utilKernels.h"
#include "kernels/decodeAttentionKernels/decoderXQARunner.h"
#include "kernels/posEncoding/applyRopeWriteKV.h"
#include "plugins/utils/pluginUtils.h"

// CuTe DSL FMHA kernel (Blackwell SM100+)
#ifdef CUTE_DSL_FMHA_ENABLED
#include "kernels/contextAttentionKernels/cuteDslFMHARunner.h"
#endif

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{

namespace
{
constexpr char const* kATTENTION_PLUGIN_VERSION{"1"};
constexpr char const* kATTENTION_PLUGIN_NAME{"AttentionPlugin"};
constexpr char const* kATTENTION_PLUGIN_NAMESPACE{""};

// Select KV cache storage datatype based on FP8 enablement
static inline DataType selectKvCacheDataType(bool enableFp8KVCache)
{
    return enableFp8KVCache ? DataType::kFP8 : DataType::kHALF;
}

// Define the mapping of input and output indices of the AttentionPlugin.
constexpr int32_t kIN_Q_IDX{0};
constexpr int32_t kIN_K_IDX{1};
constexpr int32_t kIN_V_IDX{2};
constexpr int32_t kIN_KV_CACHE_IDX{3};
constexpr int32_t kIN_CONTEXT_LENGTH_IDX{4};
constexpr int32_t kIN_ROPE_COS_SIN_IDX{5};
constexpr int32_t kIN_KV_CACHE_START_IDX{6};
constexpr int32_t kIN_OPTIONAL_ATTN_MASK_IDX{7};
constexpr int32_t kIN_OPTIONAL_ATTN_POS_ID_IDX{8};
constexpr int32_t kOUT_ATTENTION_IDX{0};
constexpr int32_t kOUT_KV_CACHE_IDX{1};

// Reflect the count of Inputs and Outputs of the AttentionPlugin,
// these definitions shall be consistent.
constexpr int32_t kNUM_REQUIRED_INPUTS{7};
constexpr int32_t kNUM_TREE_ATTN_OPTIONAL_INPUTS{2};
constexpr int32_t kNUM_FP8_KVCACHE_OPTIONAL_INPUTS{1};
constexpr int32_t kNUM_REQUIRED_OUTPUTS{2};

// Support Tree Attention decoding schema up to 128 tokens in the draft tree per batch.
// We are unable to check this property during shape checking since prefill length is much larger than this value.
constexpr int64_t kMAX_EAGLE_DECODING_TOKENS = 128;

enum class AttentionExecutionMode
{
    kINVALID,
    kNORMAL_PREFILL,
    kCHUNKED_PREFILL,
    kVANILLA_DECODING,
    kTREE_DECODING
};

AttentionExecutionMode deduceModeVanilla(rt::Tensor const& qInputTensor, rt::Tensor const& kvCacheStartIdxTensor)
{
    fprintf(stderr, "  Inside deduceModeVanilla: qInputTensor shape=%s\n", qInputTensor.getShape().formatString().c_str());
    // Empty KVCache Start indices means normal prefill without previous KVCache. Notice single token is also a valid
    // prefill length.
    if (kvCacheStartIdxTensor.getShape()[0] == 0)
    {
        return AttentionExecutionMode::kNORMAL_PREFILL;
    }

    // Otherwise, distinguish between chunked prefill and vanilla decoding based on the runtime Sequence Length.
    // Vanilla decoding should always have runtime sequence length of 1.
    int64_t const runtimeSeqLen = qInputTensor.getShape()[1];
    if (runtimeSeqLen > 1)
    {
        return AttentionExecutionMode::kCHUNKED_PREFILL;
    }
    return AttentionExecutionMode::kVANILLA_DECODING;
}

AttentionExecutionMode deduceModeTreeAttention(
    rt::Tensor const& qInputTensor, rt::Tensor const& kvCacheStartIdxTensor, rt::Tensor const& attentionPosIdTensor)
{
    // Normal prefill if there is no previous KVCache.
    if (kvCacheStartIdxTensor.getShape()[0] == 0)
    {
        return AttentionExecutionMode::kNORMAL_PREFILL;
    }

    // Under tree attention, each token will be associated with a position id (within the sequence) to perform correct
    // positional encoding. Even for casual decoding with multiple tokens, the position id is still required to be
    // supplied.

    // Note, chunked prefill is very similar to tree decoding, the difference is chunked prefill will have contiguous
    // tokens in the sequence while tree decoding has a "tree" structure described by attention mask and position ids.
    // By convention, we will supply 1 shape for position id tensor under prefill execution.
    int64_t const runtimeSeqLen = qInputTensor.getShape()[1];
    int64_t const positionIdLen = attentionPosIdTensor.getShape()[1];

    if (runtimeSeqLen == 1)
    {
        // Also supports single token decoding mode when tree attention is enabled.
        return AttentionExecutionMode::kVANILLA_DECODING;
    }
    else if (positionIdLen == runtimeSeqLen)
    {
        return AttentionExecutionMode::kTREE_DECODING;
    }
    else if (positionIdLen == 1)
    {
        return AttentionExecutionMode::kCHUNKED_PREFILL;
    }

    return AttentionExecutionMode::kINVALID;
}

} // namespace

// Static class fields initialization
PluginFieldCollection AttentionPluginCreator::mFieldCollection{};
std::vector<PluginField> AttentionPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(AttentionPluginCreator);

AttentionPlugin::AttentionPlugin(std::string const& name, int32_t numQHeads, int32_t numKVHeads, int32_t headSize,
    int32_t enableTreeAttention, int32_t enableFp8KVCache, int32_t slidingWindowSize)
    : mLayerName(name)
    , mNumQHeads(numQHeads)
    , mNumKVHeads(numKVHeads)
    , mHeadSize(headSize)
    , mEnableTreeAttention(enableTreeAttention)
    , mEnableFp8KVCache(enableFp8KVCache)
    , mSlidingWindowSize(slidingWindowSize)
{
    mDataType = nvinfer1::DataType::kHALF;
    mSMVersion = getSMVersion();
    applyThorSMRenumberWAR(mSMVersion);




    LOG_DEBUG("AttentionPlugin FMHA path: %s, sliding_window: %s", mUseCuteDslFMHA ? "CuTe DSL FMHA" : "FMHA_v2",
        mSlidingWindowSize > 0 ? std::to_string(mSlidingWindowSize).c_str() : "disabled");

    // Check FMHA implementation support and load the corresponding kernel module.
    bool canImplementFMHA = false;
#ifdef CUTE_DSL_FMHA_ENABLED
    if (mUseCuteDslFMHA && CuteDslFMHARunner::canImplement(mHeadSize, mSMVersion))
    {
        if (CuteDslFMHARunner::loadLLMKernelModule())
        {
            canImplementFMHA = true;
            LOG_DEBUG("CuTe DSL FMHA kernel loaded for SM%d", mSMVersion);
        }
        else
        {
            LOG_WARNING("CuTe DSL FMHA kernel failed to load, falling back to FMHA_v2");
            mUseCuteDslFMHA = false;
        }
    }
    if (!canImplementFMHA)
#endif
    {
        // Fallback to FMHA_v2 cubins.
        canImplementFMHA = ContextFMHARunner::canImplement(
            mHeadSize, mSMVersion, mDataType, AttentionInputLayout::SEPARATE_Q_K_V, ContextAttentionMaskType::CAUSAL);
        if (canImplementFMHA)
        {
            if (!ContextFMHARunner::loadContextFMHAKernels(mSMVersion, mDataType))
            {
                LOG_ERROR("Failed to load FMHA_v2 cubins for SM%d", mSMVersion);
                canImplementFMHA = false;
            }
        }
    }

    // XQA decode kernels are always needed regardless of FMHA path.
    bool const useSpecDecode = static_cast<bool>(mEnableTreeAttention);
    bool canImplementXQA = DecoderXQARunner::canImplement(
        mNumQHeads, mNumKVHeads, mSMVersion, mDataType, selectKvCacheDataType(mEnableFp8KVCache));
    if (canImplementXQA)
    {
        DecoderXQARunner::loadDecodeXQAKernels(
            mSMVersion, mDataType, selectKvCacheDataType(mEnableFp8KVCache), useSpecDecode);
    }

    if (!canImplementFMHA || !canImplementXQA)
    {
        LOG_ERROR(
            "Cannot implement AttentionPlugin configuration. FMHA: %s, XQA: %s, SM: %d, HeadSize: %d, NumQHeads: %d, "
            "NumKVHeads: %d",
            canImplementFMHA ? "supported" : "NOT supported", canImplementXQA ? "supported" : "NOT supported",
            mSMVersion, mHeadSize, mNumQHeads, mNumKVHeads);
        throw std::runtime_error("Cannot implement the AttentionPlugin configuration.");
    }
}

AttentionPlugin::AttentionPlugin(std::string const& name, std::byte const* data, size_t length)
    : mLayerName(name)
{
    deserializeValue(&data, &length, &mNumQHeads);
    deserializeValue(&data, &length, &mNumKVHeads);
    deserializeValue(&data, &length, &mHeadSize);
    deserializeValue(&data, &length, &mEnableTreeAttention);
    deserializeValue(&data, &length, &mEnableFp8KVCache);
    deserializeValue(&data, &length, &mSlidingWindowSize);
    
    // mDataType is initialized to kHALF in the header or should be set here
    mDataType = nvinfer1::DataType::kHALF;

#ifdef CUTE_DSL_FMHA_ENABLED
    mUseCuteDslFMHA = !std::getenv("DISABLE_CUTE_DSL_FMHA");
#else
    mUseCuteDslFMHA = false;
#endif

    mSMVersion = getSMVersion();
    applyThorSMRenumberWAR(mSMVersion);




    LOG_DEBUG("AttentionPlugin FMHA path: %s", mUseCuteDslFMHA ? "CuTe DSL FMHA" : "FMHA_v2");

    // Load FMHA kernel module based on implementation support.
#ifdef CUTE_DSL_FMHA_ENABLED
    if (mUseCuteDslFMHA && CuteDslFMHARunner::canImplement(mHeadSize, mSMVersion))
    {
        if (!CuteDslFMHARunner::loadLLMKernelModule())
        {
            LOG_WARNING("CuTe DSL FMHA kernel failed to load, falling back to FMHA_v2");
            mUseCuteDslFMHA = false;
        }
    }
    if (!mUseCuteDslFMHA)
#endif
    {
        if (!ContextFMHARunner::loadContextFMHAKernels(mSMVersion, mDataType))
        {
            LOG_ERROR("Failed to load FMHA_v2 cubins for SM%d", mSMVersion);
        }
    }

    // XQA decode kernels are always needed regardless of FMHA path.
    bool const useSpecDecode = static_cast<bool>(mEnableTreeAttention);
    DecoderXQARunner::loadDecodeXQAKernels(
        mSMVersion, mDataType, selectKvCacheDataType(mEnableFp8KVCache), useSpecDecode);
}

AttentionPlugin::~AttentionPlugin() {}

IPluginV2DynamicExt* AttentionPlugin::clone() const noexcept
{
    AttentionPlugin* plugin = new AttentionPlugin(
        mLayerName, mNumQHeads, mNumKVHeads, mHeadSize, mEnableTreeAttention, mEnableFp8KVCache, mSlidingWindowSize);
    plugin->mDataType = mDataType;
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

char const* AttentionPlugin::getPluginType() const noexcept
{
    return kATTENTION_PLUGIN_NAME;
}

char const* AttentionPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void AttentionPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = std::string(pluginNamespace);
}

char const* AttentionPlugin::getPluginVersion() const noexcept
{
    return kATTENTION_PLUGIN_VERSION;
}

int32_t AttentionPlugin::getNbOutputs() const noexcept
{
    // At both context and generation phase, output attention result and kv-cache.
    return 2;
}

bool AttentionPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    // Maximally permissive format checking to bypass Myelin Error Code 10 crashes
    // The Python ONNX trace enforces strict type safety and dimension validity
    auto type = inOut[pos].type;
    return inOut[pos].format == TensorFormat::kLINEAR && 
           (type == DataType::kHALF || type == DataType::kFLOAT || type == DataType::kINT32 || type == DataType::kBF16 || type == DataType::kFP8 || type == DataType::kINT8);
}

// IPluginV2Ext Methods
DataType AttentionPlugin::getOutputDataType([[maybe_unused]] int32_t index,
    [[maybe_unused]] nvinfer1::DataType const* inputTypes, [[maybe_unused]] int32_t nbInputs) const noexcept
{
    // Output[0] (attention) follows Q input dtype (HALF). Output[1] (KV cache) follows KV input dtype (HALF or FP8)
    if (index == kOUT_ATTENTION_IDX)
    {
        return inputTypes[kIN_Q_IDX];
    }
    return inputTypes[kIN_KV_CACHE_IDX];
}

DimsExprs AttentionPlugin::getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs,
    [[maybe_unused]] int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // Output[0] is attention result, has shape [B, S, Hq, D]. Refers to Q shape [B, S, Hq, D]
    DimsExprs output;
    if (outputIndex == kOUT_ATTENTION_IDX)
    {
        output.nbDims = 4;
        output.d[0] = inputs[kIN_Q_IDX].d[0];
        output.d[1] = inputs[kIN_Q_IDX].d[1];
        output.d[2] = exprBuilder.constant(mNumQHeads);
        output.d[3] = exprBuilder.constant(mHeadSize);
    }
    else if (outputIndex == kOUT_KV_CACHE_IDX)
    {
        // Output[1] is KVCache, same shape as input KV cache
        output.nbDims = 5;
        output.d[0] = inputs[kIN_KV_CACHE_IDX].d[0];
        output.d[1] = inputs[kIN_KV_CACHE_IDX].d[1];
        output.d[2] = inputs[kIN_KV_CACHE_IDX].d[2];
        output.d[3] = inputs[kIN_KV_CACHE_IDX].d[3];
        output.d[4] = inputs[kIN_KV_CACHE_IDX].d[4];
    }
    return output;
}

void AttentionPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in,
    int32_t nbInputs, nvinfer1::DynamicPluginTensorDesc const* out,
    int32_t nbOutputs) noexcept
{
    mDataType = in[kIN_Q_IDX].desc.type;
    
    // Update head counts and dimensions from input descriptors to ensure consistency with the engine.
    // Use the KV cache dimensions to determine the number of KV heads in the cache.
    // KV cache shape: [B, 2, Hkv, Capacity, D]
    mNumKVHeads = static_cast<int32_t>(in[kIN_KV_CACHE_IDX].desc.dims.d[2]);
    mHeadSize = static_cast<int32_t>(in[kIN_KV_CACHE_IDX].desc.dims.d[4]);
    
    // Q input shape: [B, S, Hq * D]
    int32_t const totalQDim = static_cast<int32_t>(in[kIN_Q_IDX].desc.dims.d[2]);
    mNumQHeads = totalQDim / mHeadSize;
}

// TODO: extend the workspace calculation to a more generalized form.
size_t AttentionPlugin::getWorkspaceSize([[maybe_unused]] nvinfer1::PluginTensorDesc const* inputs,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] nvinfer1::PluginTensorDesc const* outputs,
    [[maybe_unused]] int32_t nbOutputs) const noexcept
{
    // TensorRT will supply max profile shape for each input/output tensor across all optimization profiles.
    // We will request workspace to keep intermediate tensors under prefill/decode phase executions.
    // Obtain max supported batch size from the input tensor shapes. The Q input tensor will be in shape
    // [B, S, Hq, D] where S is padded the max length of the input sequence within this batch.
    PluginTensorDesc const& qInputDesc = inputs[kIN_Q_IDX];
    int64_t const maxBatchSize = qInputDesc.dims.d[0];

    // Obtain max KV cache capacity from the KV cache tensor shape.
    // The KV cache tensor has shape [B, 2, num_kv_heads, capacity, head_dim]
    PluginTensorDesc const& kvCacheDesc = inputs[kIN_KV_CACHE_IDX];
    int64_t const maxKVCacheCapacity = kvCacheDesc.dims.d[3];

    size_t workspaceSize = 0;

    // CuQSeqLens for FMHA.
    workspaceSize = accumulateWorkspaceSize(workspaceSize, {maxBatchSize + 1}, DataType::kINT32);

    // Always reserve workspace memory to prepare for chunked prefill decoding. The implementation should be further
    // optimized to avoid the workspace size overhead.

    // CuTotalKvCacheLens to describe the cumulative length of KV tensors.
    workspaceSize = accumulateWorkspaceSize(workspaceSize, rt::Coords{maxBatchSize + 1}, DataType::kINT32);
    // KVCache ends that denote the end index of each KVCache lane after adding current contents.
    workspaceSize = accumulateWorkspaceSize(workspaceSize, rt::Coords{maxBatchSize}, DataType::kINT32);
    // KV Tensor to store concated KV that include pre-cached KV and current KV.
    workspaceSize = accumulateWorkspaceSize(
        workspaceSize, rt::Coords{maxBatchSize, 2, mNumKVHeads, maxKVCacheCapacity, mHeadSize}, DataType::kHALF);

    // Request another alignment size to align the workspace pointer.
    workspaceSize += kDEVICE_ALIGNMENT;

    LOG_DEBUG("AttentionPlugin workspace size: %zu bytes", workspaceSize);
    return workspaceSize;
}

int32_t AttentionPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    [[maybe_unused]] nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept
{

    // Construct non-owned tensor objects from I/O data pointers and shapes.
    // Q input in the graph will be in shape [B, S, Hq x D], for convenience,
    // we will use shape of [B, S, Hq, D] to represent the tensor.
    // K and V inputs are in shape [B, S, Hkv x D], represented as [B, S, Hkv, D].
    PluginTensorDesc const& qInputDesc = inputDesc[kIN_Q_IDX];
    
    LOG_DEBUG("AttentionPlugin::enqueue: Q nbDims=%d, d[0]=%d, d[1]=%d, d[2]=%d", 
              qInputDesc.dims.nbDims, qInputDesc.dims.d[0], qInputDesc.dims.d[1], qInputDesc.dims.d[2]);
    fprintf(stderr, "  AttentionPlugin: mNumQHeads=%d, mNumKVHeads=%d, mHeadSize=%d\n", mNumQHeads, mNumKVHeads, mHeadSize);

    PluginTensorDesc const& kInputDesc = inputDesc[kIN_K_IDX];
    PluginTensorDesc const& vInputDesc = inputDesc[kIN_V_IDX];
    fprintf(stderr, "  AttentionPlugin: K dims=[%ld, %ld, %ld], V dims=[%ld, %ld, %ld]\n", 
            kInputDesc.dims.d[0], kInputDesc.dims.d[1], kInputDesc.dims.d[2],
            vInputDesc.dims.d[0], vInputDesc.dims.d[1], vInputDesc.dims.d[2]);
    int32_t const runtimeBatchSize = static_cast<int32_t>(qInputDesc.dims.d[0]);
    int32_t const runtimeSeqLen = static_cast<int32_t>(qInputDesc.dims.d[1]);
    
    // Determine the number of KV heads in the input tensors (might be different from cache due to GQA)
    int32_t const numKVHeadsIn = static_cast<int32_t>(kInputDesc.dims.d[2]) / mHeadSize;

    check::check(kInputDesc.dims.d[0] == runtimeBatchSize && vInputDesc.dims.d[0] == runtimeBatchSize,
        "Batch size must be consistent across Q/K/V inputs.");
    check::check(kInputDesc.dims.d[1] == runtimeSeqLen && vInputDesc.dims.d[1] == runtimeSeqLen,
        "Sequence length must be consistent across Q/K/V inputs.");
    check::check(qInputDesc.dims.d[2] == mNumQHeads * mHeadSize, "Q input shape shall be consistent.");
    check::check(kInputDesc.dims.d[2] == numKVHeadsIn * mHeadSize, "K input shape shall be consistent.");
    check::check(vInputDesc.dims.d[2] == numKVHeadsIn * mHeadSize, "V input shape shall be consistent.");

    rt::Tensor qInputTensor(const_cast<void*>(inputs[kIN_Q_IDX]),
        rt::Coords{runtimeBatchSize, runtimeSeqLen, mNumQHeads, mHeadSize}, rt::DeviceType::kGPU, qInputDesc.type);
    rt::Tensor kInputTensor(const_cast<void*>(inputs[kIN_K_IDX]),
        rt::Coords{runtimeBatchSize, runtimeSeqLen, numKVHeadsIn, mHeadSize}, rt::DeviceType::kGPU, kInputDesc.type);
    rt::Tensor vInputTensor(const_cast<void*>(inputs[kIN_V_IDX]),
        rt::Coords{runtimeBatchSize, runtimeSeqLen, numKVHeadsIn, mHeadSize}, rt::DeviceType::kGPU, vInputDesc.type);

    fprintf(stderr, "  AttentionPlugin Enqueue: q=%p, k=%p, v=%p, cache=%p, context=%p, rope=%p, start=%p\n",
            inputs[kIN_Q_IDX], inputs[kIN_K_IDX], inputs[kIN_V_IDX], inputs[kIN_KV_CACHE_IDX],
            inputs[kIN_CONTEXT_LENGTH_IDX], inputs[kIN_ROPE_COS_SIN_IDX], inputs[kIN_KV_CACHE_START_IDX]);
    PluginTensorDesc const& contextLengthInputDesc = inputDesc[kIN_CONTEXT_LENGTH_IDX];
    rt::Tensor const contextLengthTensor(const_cast<void*>(inputs[kIN_CONTEXT_LENGTH_IDX]),
        rt::Coords{contextLengthInputDesc.dims}, rt::DeviceType::kGPU, contextLengthInputDesc.type);

    fprintf(stderr, "  Creating ropeCosSinTensor\n");
    PluginTensorDesc const& posEncodingCosSinDesc = inputDesc[kIN_ROPE_COS_SIN_IDX];
    rt::Tensor const ropeCosSinTensor(const_cast<void*>(inputs[kIN_ROPE_COS_SIN_IDX]),
        rt::Coords{posEncodingCosSinDesc.dims}, rt::DeviceType::kGPU, posEncodingCosSinDesc.type);

    fprintf(stderr, "  Creating kvCacheStartIdxTensor\n");
    PluginTensorDesc const& kvCacheStartIdxInputDesc = inputDesc[kIN_KV_CACHE_START_IDX];
    rt::Tensor const kvCacheStartIdxTensor(const_cast<void*>(inputs[kIN_KV_CACHE_START_IDX]),
        rt::Coords{kvCacheStartIdxInputDesc.dims}, rt::DeviceType::kGPU, kvCacheStartIdxInputDesc.type);

    fprintf(stderr, "  Creating attentionOutputTensor\n");
    PluginTensorDesc const& attentionOutputDesc = outputDesc[kOUT_ATTENTION_IDX];
    rt::Tensor attentionOutputTensor(outputs[kOUT_ATTENTION_IDX], rt::Coords{attentionOutputDesc.dims},
        rt::DeviceType::kGPU, attentionOutputDesc.type);

    // Construct the KVCache tensor from the input KV cache descriptor.
    // This allows KV cache from 0 to maxSeqLen and helps adjust the profile at runtime.
    PluginTensorDesc const& kvCacheInputDesc = inputDesc[kIN_KV_CACHE_IDX];
    rt::Tensor kvCacheTensor(
        outputs[kOUT_KV_CACHE_IDX], rt::Coords{kvCacheInputDesc.dims}, rt::DeviceType::kGPU, kvCacheInputDesc.type);

    // Extract KV cache capacity from the runtime tensor shape.
    int32_t const kvCacheCapacity = static_cast<int32_t>(kvCacheInputDesc.dims.d[3]);

    // Optional Inputs that are not used with Tree Attention enabled.
    rt::Tensor attentionMaskTensor{};
    rt::Tensor attentionPosIdTensor{};
    if (mEnableTreeAttention)
    {
        PluginTensorDesc const& attentionMaskInputDesc = inputDesc[kIN_OPTIONAL_ATTN_MASK_IDX];
        PluginTensorDesc const& attentionPosIdInputDesc = inputDesc[kIN_OPTIONAL_ATTN_POS_ID_IDX];
        attentionMaskTensor = rt::Tensor(const_cast<void*>(inputs[kIN_OPTIONAL_ATTN_MASK_IDX]),
            rt::Coords{attentionMaskInputDesc.dims}, rt::DeviceType::kGPU, attentionMaskInputDesc.type);
        attentionPosIdTensor = rt::Tensor(const_cast<void*>(inputs[kIN_OPTIONAL_ATTN_POS_ID_IDX]),
            rt::Coords{attentionPosIdInputDesc.dims}, rt::DeviceType::kGPU, attentionPosIdInputDesc.type);
    }

    // Determine the attention execution mode based on the input tensors.
    AttentionExecutionMode executionMode{};
    if (!mEnableTreeAttention)
    {
        executionMode = deduceModeVanilla(qInputTensor, kvCacheStartIdxTensor);
    }
    else
    {
        executionMode = deduceModeTreeAttention(qInputTensor, kvCacheStartIdxTensor, attentionPosIdTensor);
    }

    // For invalid execution mode, log error and report error return value.
    if (executionMode == AttentionExecutionMode::kINVALID)
    {
        LOG_ERROR("Invalid attention execution mode detected. Abort the AttentionPlugin enqueue() call.");
        return 1;
    }

    // Optional packed FP8 KV cache scales: [k_scale_quant_orig, v_scale_quant_orig]
    // When FP8 KV cache is disabled, this tensor stays empty and downstream kernels ignore it.
    rt::Tensor kvScaleQuantOrigTensor{};
    if (mEnableFp8KVCache)
    {
        int32_t const kvScaleQuantOrigInputIdx
            = kIN_KV_CACHE_START_IDX + 1 + (mEnableTreeAttention ? kNUM_TREE_ATTN_OPTIONAL_INPUTS : 0);
        PluginTensorDesc const& kvScaleQuantOrigDesc = inputDesc[kvScaleQuantOrigInputIdx];
        kvScaleQuantOrigTensor = rt::Tensor(const_cast<void*>(inputs[kvScaleQuantOrigInputIdx]),
            rt::Coords{kvScaleQuantOrigDesc.dims}, rt::DeviceType::kGPU, kvScaleQuantOrigDesc.type);
        // Runtime validation (in addition to supportsFormatCombination)
        check::check(
            kvScaleQuantOrigTensor.getDataType() == nvinfer1::DataType::kFLOAT, "kvScaleQuantOrig must be FP32.");
        check::check(kvScaleQuantOrigTensor.getShape().getNumDims() == 1 && kvScaleQuantOrigTensor.getShape()[0] == 2,
            "kvScaleQuantOrig shall have shape [2] with layout [kScaleQuantOrig, vScaleQuantOrig].");
    }

    auto const nbInputs = kNUM_REQUIRED_INPUTS + (mEnableTreeAttention ? kNUM_TREE_ATTN_OPTIONAL_INPUTS : 0)
        + (mEnableFp8KVCache ? kNUM_FP8_KVCACHE_OPTIONAL_INPUTS : 0);
    auto const nbOutputs = kNUM_REQUIRED_OUTPUTS;
    size_t space = getWorkspaceSize(inputDesc, nbInputs, outputDesc, nbOutputs);
    // Align the workspace pointer so that each tensor assigned from the workspace will align to the device alignment
    // granularity.
    std::byte* alignedWorkspacePtr
        = static_cast<std::byte*>(std::align(kDEVICE_ALIGNMENT, space - kDEVICE_ALIGNMENT, workspace, space));
    if (alignedWorkspacePtr == nullptr)
    {
        LOG_ERROR("Workspace size is too small to hold all data structures with correct alignment");
        return 1;
    }

    if (executionMode == AttentionExecutionMode::kNORMAL_PREFILL
        || executionMode == AttentionExecutionMode::kCHUNKED_PREFILL)
    {
        fprintf(stderr, "  Creating cuQSeqLensTensor\n");
        rt::Tensor cuQSeqLensTensor
            = assignTensorFromWorkspace(alignedWorkspacePtr, {runtimeBatchSize + 1}, DataType::kINT32);

        fprintf(stderr, "  Creating cuKVSeqLensTensor\n");
        rt::Tensor cuKVSeqLensTensor
            = assignTensorFromWorkspace(alignedWorkspacePtr, {runtimeBatchSize + 1}, DataType::kINT32);

        fprintf(stderr, "  Creating kvCacheEndIdxsTensor\n");
        rt::Tensor kvCacheEndIdxsTensor
            = assignTensorFromWorkspace(alignedWorkspacePtr, {runtimeBatchSize}, DataType::kINT32);

        fprintf(stderr, "  Initializing cu_seqlens on host (runtimeSeqLen=%d)\n", runtimeSeqLen);
        
        // Host-side initialization
        int32_t* hostCuQ = new int32_t[runtimeBatchSize + 1];
        int32_t* hostCuKV = new int32_t[runtimeBatchSize + 1];
        int32_t* hostEndIdxs = new int32_t[runtimeBatchSize];
        
        hostCuQ[0] = 0;
        hostCuKV[0] = 0;
        for (int32_t i = 0; i < runtimeBatchSize; ++i) {
            hostCuQ[i+1] = hostCuQ[i] + runtimeSeqLen;
            hostCuKV[i+1] = hostCuKV[i] + runtimeSeqLen;
            hostEndIdxs[i] = runtimeSeqLen;
        }
        
        CUDA_CHECK(cudaMemcpy(cuQSeqLensTensor.rawPointer(), hostCuQ, 
                                  (runtimeBatchSize + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(cuKVSeqLensTensor.rawPointer(), hostCuKV, 
                                  (runtimeBatchSize + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(kvCacheEndIdxsTensor.rawPointer(), hostEndIdxs, 
                                  runtimeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice));
        
        delete[] hostCuQ;
        delete[] hostCuKV;
        delete[] hostEndIdxs;
        
        // Synchronize to ensure sequence lengths are ready for FMHA
        CUDA_CHECK(cudaStreamSynchronize(stream));

#ifdef CUTE_DSL_FMHA_ENABLED
        // Enable CuteDSL FMHA for single batch prefill usecase when FP8 KVCache is disabled.
        // TODO: Enable multi-batch prefill and FP8 KVCache after we improve the kernel implementation.
        bool const enableCuteDslFMHA = mUseCuteDslFMHA && !mEnableFp8KVCache && runtimeBatchSize == 1;
        if (enableCuteDslFMHA)
        {
            fprintf(stderr, "  Using CuteDSL FMHA\n");
            kernel::launchApplyRopeWriteKVSplitQKV(ropeCosSinTensor, kvCacheEndIdxsTensor, qInputTensor, kInputTensor,
                vInputTensor, kvCacheTensor, kvScaleQuantOrigTensor, stream);

            // Run CuTe DSL FMHA kernel with combined KV tensor
            // Expected layouts: Q [b, s_q, h_q, d], KV [b, 2, hkv, cap, d], O [b, s_q, h_q, d]
            CuteDslFMHARunner runner(
                mNumQHeads, mNumKVHeads, mHeadSize, runtimeBatchSize, runtimeSeqLen, kvCacheCapacity);
            runner.run(qInputTensor.dataPointer<half>(),   // qPtr [b, s_q, h_q, d]
                kvCacheTensor.dataPointer<half>(),         // kvPtr [b, 2, h_k, cap, d]
                attentionOutputTensor.dataPointer<half>(), // oPtr [b, s_q, h_q, d]
                cuKVSeqLensTensor.dataPointer<int32_t>(),  // cu_kv_seqlens [b+1]
                stream, mSlidingWindowSize > 0 ? mSlidingWindowSize : INT_MAX);
        }
        else
#endif
        {
            fprintf(stderr, "  Using FMHA_v2\n");
            // Force CONTIGUOUS_Q_KV layout (using KV cache) if GQA broadcasting is required,
            // as the FMHA raw input kernels don't handle broadcasting.
            bool const forceKvCachePath = (numKVHeadsIn != mNumKVHeads);
            
            AttentionInputLayout attentionInputLayout = (executionMode == AttentionExecutionMode::kCHUNKED_PREFILL || forceKvCachePath)
                ? AttentionInputLayout::CONTIGUOUS_Q_KV
                : AttentionInputLayout::SEPARATE_Q_K_V;
            
            fprintf(stderr, "  Creating ContextFMHARunner (forceKvCachePath=%d)\n", (int)forceKvCachePath);
            auto fmhaRunner = ContextFMHARunner(mDataType, runtimeBatchSize, runtimeSeqLen, mNumQHeads, mNumKVHeads,
                mHeadSize, mSMVersion, attentionInputLayout);

            fprintf(stderr, "  Setting up FMHA_v2 params\n");
            // Use heap-allocated aligned memory for params to ensure strict 64-byte alignment
            // as required by the CUDA driver for structs containing TMA descriptors.
            void* raw_params = nullptr;
            if (posix_memalign(&raw_params, 64, sizeof(FusedMultiheadAttentionParamsV2)) != 0) {
                LOG_ERROR("Failed to allocate aligned FMHA params");
                return 1;
            }
            std::unique_ptr<FusedMultiheadAttentionParamsV2, void(*)(void*)> params_guard(
                static_cast<FusedMultiheadAttentionParamsV2*>(raw_params), free);
            FusedMultiheadAttentionParamsV2& params = *params_guard;
            params = FusedMultiheadAttentionParamsV2{};
            
            fprintf(stderr, "  Params address=%p, alignment=%zu\n", (void*)&params, (size_t)(&params) % 64);
            fmhaRunner.setupParams(params);
            
            fprintf(stderr, "  Setting cu_q_seqlens\n");
            params.cu_q_seqlens = cuQSeqLensTensor.dataPointer<int32_t>();

            if (attentionInputLayout == AttentionInputLayout::CONTIGUOUS_Q_KV)
            {
                // FMHA kernel with CONTIGUOUS_Q_KV input layout currently only supports FP16 KV cache.
                // kvCache: [b, 2, hkv, s, d] -> [b, s, 2, hkv, d]
                kernel::launchApplyRopeWriteKV(ropeCosSinTensor, kvCacheEndIdxsTensor, qInputTensor, kInputTensor,
                    vInputTensor, kvCacheTensor, kvScaleQuantOrigTensor, stream, false);

                rt::Tensor transposedKVTensor = assignTensorFromWorkspace(alignedWorkspacePtr,
                    {runtimeBatchSize, kvCacheCapacity, 2, mNumKVHeads, mHeadSize}, DataType::kHALF);
                kernel::cvtKVLayoutBHSDToBSHD(kvCacheTensor, transposedKVTensor, kvScaleQuantOrigTensor, stream);

                // Set device ptr for FMHA kernel.
                params.s_kv = kvCacheCapacity;
                params.q_ptr = qInputTensor.dataPointer<half>();
                params.kv_ptr = transposedKVTensor.dataPointer<half>();
                params.cu_kv_seqlens = cuKVSeqLensTensor.dataPointer<int32_t>();
                params.o_ptr = attentionOutputTensor.dataPointer<half>();
            }
            else
            { // SEPARATE_Q_K_V
                kernel::launchApplyRopeWriteKV(ropeCosSinTensor, std::nullopt, qInputTensor, kInputTensor, vInputTensor,
                    kvCacheTensor, kvScaleQuantOrigTensor, stream, true);

                params.s_kv = runtimeSeqLen;
                params.q_ptr = qInputTensor.dataPointer<half>();
                params.k_ptr = kInputTensor.dataPointer<half>();
                params.v_ptr = vInputTensor.dataPointer<half>();
                params.cu_kv_seqlens = cuKVSeqLensTensor.dataPointer<int32_t>();
                params.o_ptr = attentionOutputTensor.dataPointer<half>();
                
                fprintf(stderr, "  FMHA Pointers: q=%p, k=%p, v=%p, cuQ=%p, cuKV=%p, o=%p\n",
                        params.q_ptr, params.k_ptr, params.v_ptr, params.cu_q_seqlens, params.cu_kv_seqlens, params.o_ptr);
            }

            // Dispatch FMHA kernel
            fmhaRunner.dispatchFMHAKernel(params, stream);
        }
    }
    else
    {
        // Prepare Decoding attention runner parameter to dispatch kernel
        if (executionMode == AttentionExecutionMode::kTREE_DECODING)
        {
            // Execute tree attention decoding.
            kernel::launchApplyRopeWriteKVTreeDecoding(ropeCosSinTensor, contextLengthTensor, attentionPosIdTensor,
                qInputTensor, kInputTensor, vInputTensor, kvCacheTensor, kvScaleQuantOrigTensor, stream);
        }
        else
        {
            // Execute vanilla decoding.
            kernel::launchApplyRopeWriteKV(ropeCosSinTensor, contextLengthTensor, qInputTensor, kInputTensor,
                vInputTensor, kvCacheTensor, kvScaleQuantOrigTensor, stream, false);
        }

        auto xqaRunner = DecoderXQARunner(mDataType, selectKvCacheDataType(mEnableFp8KVCache), runtimeBatchSize,
            mNumQHeads, mNumKVHeads, mHeadSize, mSMVersion);
        XQALaunchParams params = xqaRunner.initXQAParams();
        if (mEnableFp8KVCache)
        {
            float const* const kvScales = kvScaleQuantOrigTensor.dataPointer<float>();
            params.kScale = (kvScales + 0);
            params.vScale = (kvScales + 1);
        }
        params.output = attentionOutputTensor.dataPointer<half>();
        params.qInputPtr = qInputTensor.dataPointer<half>();
        params.kvCache.data = kvCacheTensor.rawPointer();
        params.kvCache.sequence_lengths = contextLengthTensor.dataPointer<int32_t>();
        params.kvCache.capacity = kvCacheCapacity;
        if (executionMode == AttentionExecutionMode::kTREE_DECODING)
        {
            // Execute tree attention decoding.
            params.treeAttnMask = attentionMaskTensor.dataPointer<int32_t>();
            params.qSeqLen = runtimeSeqLen;
            xqaRunner.dispatchSpecDecodeXQAKernel(params, stream);
        }
        else
        {
            // Execute vanilla decoding.
            xqaRunner.dispatchXQAKernel(params, stream);
        }
    }
    return 0;
}

size_t AttentionPlugin::getSerializationSize() const noexcept
{
    return sizeof(mNumQHeads) + sizeof(mNumKVHeads) + sizeof(mHeadSize) + sizeof(mEnableTreeAttention)
        + sizeof(mEnableFp8KVCache) + sizeof(int32_t) /* slidingWindowSize */;
}

void AttentionPlugin::serialize(void* buffer) const noexcept
{
    std::byte* byteBuffer = static_cast<std::byte*>(buffer);
    serializeValue(&byteBuffer, mNumQHeads);
    serializeValue(&byteBuffer, mNumKVHeads);
    serializeValue(&byteBuffer, mHeadSize);
    serializeValue(&byteBuffer, mEnableTreeAttention);
    serializeValue(&byteBuffer, mEnableFp8KVCache);
    serializeValue(&byteBuffer, mSlidingWindowSize);
}

int32_t AttentionPlugin::initialize() noexcept
{
    return 0;
}

void AttentionPlugin::terminate() noexcept {}

void AttentionPlugin::destroy() noexcept
{
    delete this;
}

AttentionPluginCreator::AttentionPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);

    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num_q_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_kv_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("head_size", nullptr, PluginFieldType::kINT32, 1));
    // Make enable_fp8_kv_cache optional with default value 0 (disable by default)
    mPluginAttributes.emplace_back(PluginField("enable_tree_attention", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("enable_fp8_kv_cache", nullptr, PluginFieldType::kINT32, 0));
    // Sliding window size (-1 = no sliding window, >0 = window size)
    mPluginAttributes.emplace_back(PluginField("sliding_window_size", nullptr, PluginFieldType::kINT32, 0));
    // Enforce Core parameters are specified.
    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* AttentionPluginCreator::getPluginName() const noexcept
{
    return kATTENTION_PLUGIN_NAME;
}

nvinfer1::PluginFieldCollection const* AttentionPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

void AttentionPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* AttentionPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* AttentionPluginCreator::getPluginVersion() const noexcept
{
    return kATTENTION_PLUGIN_VERSION;
}

nvinfer1::IPluginV2* AttentionPluginCreator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept
{
    try
    {
        std::optional<int32_t> numQHeads = parsePluginScalarField<int32_t>("num_q_heads", fc);
        std::optional<int32_t> numKVHeads = parsePluginScalarField<int32_t>("num_kv_heads", fc);
        std::optional<int32_t> headSize = parsePluginScalarField<int32_t>("head_size", fc);
        std::optional<int32_t> enableTreeAttention = parsePluginScalarField<int32_t>("enable_tree_attention", fc);
        std::optional<int32_t> enableFp8KVCache = parsePluginScalarField<int32_t>("enable_fp8_kv_cache", fc);
        // sliding_window_size: -1 = no sliding window (default), >0 = sliding window size
        int32_t slidingWindowSize = parsePluginScalarField<int32_t>("sliding_window_size", fc).value_or(-1);
        // Make enable_fp8_kv_cache optional with default value 0 (disable by default)
        int32_t enableFp8KVCacheValue = enableFp8KVCache.value_or(0);

        bool checkRequiredFields = numQHeads.has_value() && headSize.has_value() && numKVHeads.has_value()
            && enableTreeAttention.has_value();
        if (!checkRequiredFields)
        {
            LOG_ERROR("Missing required AttentionPlugin fields.");
            return nullptr;
        }

        AttentionPlugin* plugin = new AttentionPlugin(std::string(name), numQHeads.value(), numKVHeads.value(),
            headSize.value(), enableTreeAttention.value(), enableFp8KVCacheValue, slidingWindowSize);

        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to create AttentionPlugin: %s", e.what());
    }
    return nullptr;
}

nvinfer1::IPluginV2* AttentionPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        return new AttentionPlugin(name, static_cast<std::byte const*>(serialData), serialLength);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to deserialize AttentionPlugin: %s", e.what());
    }
    return nullptr;
}

} // namespace plugins
} // namespace trt_edgellm
