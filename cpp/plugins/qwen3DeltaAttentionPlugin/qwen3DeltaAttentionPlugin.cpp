/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "qwen3DeltaAttentionPlugin.h"

#include "common/logger.h"
#include "plugins/utils/pluginUtils.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <mutex>
#include <optional>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{

namespace
{
constexpr char const* kQWEN3_DELTA_PLUGIN_VERSION{"1"};
constexpr char const* kQWEN3_DELTA_PLUGIN_NAME{"qwen3_delta_attention"};

// Input indices for Gated Delta Attention
constexpr int32_t kIN_Q_IDX{0};       // [batch, (seq_len,) n_q_heads, head_size]
constexpr int32_t kIN_K_IDX{1};       // [batch, (seq_len,) n_kv_heads, head_size]
constexpr int32_t kIN_V_IDX{2};       // [batch, (seq_len,) n_kv_heads, head_size]
constexpr int32_t kIN_G_IDX{3};       // [batch, (seq_len,) n_q_heads]
constexpr int32_t kIN_BETA_IDX{4};    // [batch, (seq_len,) n_q_heads]
constexpr int32_t kIN_STATE_IDX{5};   // [batch, n_q_heads, head_size, head_size]

// Output indices
constexpr int32_t kOUT_OUTPUT_IDX{0}; // [batch, (seq_len,) n_q_heads, head_size]
constexpr int32_t kOUT_STATE_IDX{1};  // [batch, n_q_heads, head_size, head_size]

constexpr int32_t kNUM_INPUTS{6};
constexpr int32_t kNUM_OUTPUTS{2};

} // namespace

PluginFieldCollection Qwen3DeltaAttentionPluginCreator::mFieldCollection{};
std::vector<PluginField> Qwen3DeltaAttentionPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(Qwen3DeltaAttentionPluginCreator);

Qwen3DeltaAttentionPlugin::Qwen3DeltaAttentionPlugin(std::string const& name, int32_t numQHeads, int32_t numKVHeads, int32_t headSize)
    : mLayerName(name)
    , mNumQHeads(numQHeads)
    , mNumKVHeads(numKVHeads)
    , mHeadSize(headSize)
{
}

Qwen3DeltaAttentionPlugin::~Qwen3DeltaAttentionPlugin() {}

IPluginCapability* Qwen3DeltaAttentionPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    if (type == PluginCapabilityType::kBUILD)
    {
        return static_cast<IPluginV3OneBuild*>(this);
    }
    if (type == PluginCapabilityType::kRUNTIME)
    {
        return static_cast<IPluginV3OneRuntime*>(this);
    }
    return static_cast<IPluginV3OneCore*>(this);
}

IPluginV3* Qwen3DeltaAttentionPlugin::clone() noexcept
{
    Qwen3DeltaAttentionPlugin* plugin = new Qwen3DeltaAttentionPlugin(mLayerName, mNumQHeads, mNumKVHeads, mHeadSize);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

char const* Qwen3DeltaAttentionPlugin::getPluginName() const noexcept
{
    return kQWEN3_DELTA_PLUGIN_NAME;
}

char const* Qwen3DeltaAttentionPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Qwen3DeltaAttentionPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = std::string(pluginNamespace);
}

char const* Qwen3DeltaAttentionPlugin::getPluginVersion() const noexcept
{
    return kQWEN3_DELTA_PLUGIN_VERSION;
}

int32_t Qwen3DeltaAttentionPlugin::getNbOutputs() const noexcept
{
    return kNUM_OUTPUTS;
}

int32_t Qwen3DeltaAttentionPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    outputTypes[kOUT_OUTPUT_IDX] = inputTypes[kIN_Q_IDX];
    outputTypes[kOUT_STATE_IDX] = inputTypes[kIN_Q_IDX];
    return 0;
}

int32_t Qwen3DeltaAttentionPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* /* shapeInputs */,
    int32_t /* nbShapeInputs */, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& /* exprBuilder */) noexcept
{
    outputs[kOUT_OUTPUT_IDX] = inputs[kIN_Q_IDX];
    outputs[kOUT_STATE_IDX] = inputs[kIN_STATE_IDX];
    return 0;
}

bool Qwen3DeltaAttentionPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    // Be maximally permissive during Myelin convergence to avoid Error Code 10
    // The Python trace strictly enforces type safety via Cast nodes anyway.
    auto type = inOut[pos].desc.type;
    return inOut[pos].desc.format == TensorFormat::kLINEAR && 
           (type == DataType::kHALF || type == DataType::kFLOAT || type == DataType::kINT32 || type == DataType::kBF16);
}

int32_t Qwen3DeltaAttentionPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    return 0;
}

size_t Qwen3DeltaAttentionPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* /* inputs */, int32_t /* nbInputs */,
    DynamicPluginTensorDesc const* /* outputs */, int32_t /* nbOutputs */) const noexcept
{
    // Workspace will depend on Triton kernel requirements
    return 0;
}

int32_t Qwen3DeltaAttentionPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    // Placeholder: Will call Triton-compiled kernels here
    LOG_ERROR("Qwen3DeltaAttentionPlugin::enqueue not yet implemented");
    return -1;
}

int32_t Qwen3DeltaAttentionPlugin::onShapeChange(PluginTensorDesc const* /* in */, int32_t /* nbInputs */,
    PluginTensorDesc const* /* out */, int32_t /* nbOutputs */) noexcept
{
    return 0;
}

IPluginV3* Qwen3DeltaAttentionPlugin::attachToContext(IPluginResourceContext* /* context */) noexcept
{
    return clone();
}

PluginFieldCollection const* Qwen3DeltaAttentionPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("num_q_heads", &mNumQHeads, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("num_kv_heads", &mNumKVHeads, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("head_size", &mHeadSize, PluginFieldType::kINT32, 1);
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

Qwen3DeltaAttentionPluginCreator::Qwen3DeltaAttentionPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num_q_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_kv_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("head_size", nullptr, PluginFieldType::kINT32, 1));
    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* Qwen3DeltaAttentionPluginCreator::getPluginName() const noexcept
{
    return kQWEN3_DELTA_PLUGIN_NAME;
}

PluginFieldCollection const* Qwen3DeltaAttentionPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

void Qwen3DeltaAttentionPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* Qwen3DeltaAttentionPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* Qwen3DeltaAttentionPluginCreator::getPluginVersion() const noexcept
{
    return kQWEN3_DELTA_PLUGIN_VERSION;
}

IPluginV3* Qwen3DeltaAttentionPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    try
    {
        std::optional<int32_t> num_q_heads = parsePluginScalarField<int32_t>("num_q_heads", fc);
        std::optional<int32_t> num_kv_heads = parsePluginScalarField<int32_t>("num_kv_heads", fc);
        std::optional<int32_t> head_size = parsePluginScalarField<int32_t>("head_size", fc);

        auto* plugin = new Qwen3DeltaAttentionPlugin(std::string(name), num_q_heads.value_or(64), num_kv_heads.value_or(2), head_size.value_or(128));
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to create qwen3_delta_attention plugin: %s", e.what());
    }
    return nullptr;
}

} // namespace plugins
} // namespace trt_edgellm
