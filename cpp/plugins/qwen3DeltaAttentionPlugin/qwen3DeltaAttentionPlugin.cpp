/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "qwen3DeltaAttentionPlugin.h"
#include "qwen3DeltaAttention.h"
#include "common/logger.h"
#include "plugins/utils/pluginUtils.h"
#include <cstring>
#include <cmath>

namespace trt_edgellm
{
namespace plugins
{

namespace
{
constexpr char const* kQWEN3_DELTA_PLUGIN_VERSION{"1"};
constexpr char const* kQWEN3_DELTA_PLUGIN_NAME{"qwen3_delta_attention"};
} // namespace

nvinfer1::PluginFieldCollection Qwen3DeltaAttentionPluginCreator::mFieldCollection{};
std::vector<nvinfer1::PluginField> Qwen3DeltaAttentionPluginCreator::mPluginAttributes;

Qwen3DeltaAttentionPlugin::Qwen3DeltaAttentionPlugin(std::string name, int32_t numQHeads, int32_t numKVHeads, int32_t headSize, float eps)
    : mLayerName(std::move(name))
    , mNumQHeads(numQHeads)
    , mNumKVHeads(numKVHeads)
    , mHeadSize(headSize)
    , mEps(eps)
{
}

Qwen3DeltaAttentionPlugin::Qwen3DeltaAttentionPlugin(std::string const& name, void const* data, size_t length)
    : mLayerName(name)
{
    std::byte const* d = reinterpret_cast<std::byte const*>(data);
    deserializeValue(&d, &length, &mNumQHeads);
    deserializeValue(&d, &length, &mNumKVHeads);
    deserializeValue(&d, &length, &mHeadSize);
    deserializeValue(&d, &length, &mEps);
}

nvinfer1::IPluginV2DynamicExt* Qwen3DeltaAttentionPlugin::clone() const noexcept
{
    auto* plugin = new Qwen3DeltaAttentionPlugin(mLayerName, mNumQHeads, mNumKVHeads, mHeadSize, mEps);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

int32_t Qwen3DeltaAttentionPlugin::getNbOutputs() const noexcept
{
    return kNUM_OUTPUTS;
}

nvinfer1::DataType Qwen3DeltaAttentionPlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    return inputTypes[kIN_Q_IDX];
}

nvinfer1::DimsExprs Qwen3DeltaAttentionPlugin::getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    if (outputIndex == kOUT_OUTPUT_IDX)
    {
        return inputs[kIN_Q_IDX];
    }
    
    // State is ALWAYS 4D [B, H, D, D]
    nvinfer1::DimsExprs state;
    state.nbDims = 4;
    state.d[0] = inputs[kIN_Q_IDX].d[0]; // Batch size
    state.d[1] = exprBuilder.constant(mNumQHeads);
    state.d[2] = exprBuilder.constant(mHeadSize);
    state.d[3] = exprBuilder.constant(mHeadSize);
    return state;
}

bool Qwen3DeltaAttentionPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    return (inOut[pos].type == nvinfer1::DataType::kHALF || inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kBF16)
        && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

void Qwen3DeltaAttentionPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
}

size_t Qwen3DeltaAttentionPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

int32_t Qwen3DeltaAttentionPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    int32_t batch_size = inputDesc[kIN_STATE_IDX].dims.d[0];
    int32_t total_q_elements = 1;
    for (int i = 0; i < inputDesc[kIN_Q_IDX].dims.nbDims; ++i) total_q_elements *= inputDesc[kIN_Q_IDX].dims.d[i];
    int32_t seq_len = total_q_elements / (batch_size * mNumQHeads * mHeadSize);

    trt_edgellm::kernels::Qwen3DeltaAttentionParams params;
    params.q = inputs[kIN_Q_IDX];
    params.k = inputs[kIN_K_IDX];
    params.v = inputs[kIN_V_IDX];
    params.g = inputs[kIN_G_IDX];
    params.beta = inputs[kIN_BETA_IDX];
    params.z = inputs[kIN_Z_IDX];
    params.norm_weight = inputs[kIN_NORM_WEIGHT_IDX];
    params.output = outputs[kOUT_OUTPUT_IDX];
    params.state = outputs[kOUT_STATE_IDX];
    
    // Initial state copy if not in-place
    if (outputs[kOUT_STATE_IDX] != inputs[kIN_STATE_IDX]) {
        size_t element_size = (inputDesc[kIN_STATE_IDX].type == nvinfer1::DataType::kFLOAT) ? sizeof(float) : sizeof(uint16_t);
        size_t state_size = (size_t)batch_size * mNumQHeads * mHeadSize * mHeadSize * element_size;
        cudaMemcpyAsync(outputs[kOUT_STATE_IDX], inputs[kIN_STATE_IDX], state_size, cudaMemcpyDeviceToDevice, stream);
    }

    params.batch_size = batch_size;
    params.seq_len = seq_len;
    params.num_q_heads = mNumQHeads;
    params.num_kv_heads = mNumKVHeads;
    params.head_size = mHeadSize;
    params.scale = 1.0f; // Scale is often 1.0 or handled in Q projection
    params.eps = mEps;
    params.is_prefill = (seq_len > 1);

    trt_edgellm::kernels::invokeQwen3DeltaAttention(params, stream);

    return 0;
}

size_t Qwen3DeltaAttentionPlugin::getSerializationSize() const noexcept
{
    return sizeof(mNumQHeads) + sizeof(mNumKVHeads) + sizeof(mHeadSize) + sizeof(mEps);
}

void Qwen3DeltaAttentionPlugin::serialize(void* buffer) const noexcept
{
    std::byte* d = static_cast<std::byte*>(buffer);
    serializeValue(&d, mNumQHeads);
    serializeValue(&d, mNumKVHeads);
    serializeValue(&d, mHeadSize);
    serializeValue(&d, mEps);
}

void Qwen3DeltaAttentionPlugin::destroy() noexcept
{
    delete this;
}

void Qwen3DeltaAttentionPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

char const* Qwen3DeltaAttentionPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* Qwen3DeltaAttentionPlugin::getPluginType() const noexcept
{
    return kQWEN3_DELTA_PLUGIN_NAME;
}

char const* Qwen3DeltaAttentionPlugin::getPluginVersion() const noexcept
{
    return kQWEN3_DELTA_PLUGIN_VERSION;
}

void Qwen3DeltaAttentionPlugin::terminate() noexcept
{
}

int32_t Qwen3DeltaAttentionPlugin::initialize() noexcept
{
    return 0;
}

// Creator
Qwen3DeltaAttentionPluginCreator::Qwen3DeltaAttentionPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(nvinfer1::PluginField("num_q_heads", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("num_kv_heads", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("head_size", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("eps", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* Qwen3DeltaAttentionPluginCreator::getPluginName() const noexcept
{
    return kQWEN3_DELTA_PLUGIN_NAME;
}

char const* Qwen3DeltaAttentionPluginCreator::getPluginVersion() const noexcept
{
    return kQWEN3_DELTA_PLUGIN_VERSION;
}

nvinfer1::PluginFieldCollection const* Qwen3DeltaAttentionPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

nvinfer1::IPluginV2* Qwen3DeltaAttentionPluginCreator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept
{
    int32_t numQHeads{0};
    int32_t numKVHeads{0};
    int32_t headSize{0};
    float eps{1e-6f};

    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fc->fields[i].name;
        if (!strcmp(attrName, "num_q_heads"))
        {
            numQHeads = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (!strcmp(attrName, "num_kv_heads"))
        {
            numKVHeads = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (!strcmp(attrName, "head_size"))
        {
            headSize = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (!strcmp(attrName, "eps"))
        {
            eps = *static_cast<float const*>(fc->fields[i].data);
        }
    }

    auto* plugin = new Qwen3DeltaAttentionPlugin(name, numQHeads, numKVHeads, headSize, eps);
    return plugin;
}

nvinfer1::IPluginV2* Qwen3DeltaAttentionPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    return new Qwen3DeltaAttentionPlugin(name, serialData, serialLength);
}

void Qwen3DeltaAttentionPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* Qwen3DeltaAttentionPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

} // namespace plugins
} // namespace trt_edgellm
