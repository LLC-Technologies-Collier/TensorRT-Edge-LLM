/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvInferRuntime.h>
#include <cstddef>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace plugins
{

class Qwen3DeltaAttentionPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    Qwen3DeltaAttentionPlugin(std::string name, int32_t numQHeads, int32_t numKVHeads, int32_t headSize, float eps = 1e-6f);

    Qwen3DeltaAttentionPlugin(std::string const& name, void const* data, size_t length);

    Qwen3DeltaAttentionPlugin() = delete;

    ~Qwen3DeltaAttentionPlugin() override = default;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    
    // IPluginV2 Methods
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;
    nvinfer1::DataType getOutputDataType(int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;

// Input/Output indices
static constexpr int32_t kIN_Q_IDX{0};
static constexpr int32_t kIN_K_IDX{1};
static constexpr int32_t kIN_V_IDX{2};
static constexpr int32_t kIN_G_IDX{3};
static constexpr int32_t kIN_BETA_IDX{4};
static constexpr int32_t kIN_STATE_IDX{5};
static constexpr int32_t kIN_Z_IDX{6};
static constexpr int32_t kIN_NORM_WEIGHT_IDX{7};

static constexpr int32_t kOUT_OUTPUT_IDX{0};
static constexpr int32_t kOUT_STATE_IDX{1};

static constexpr int32_t kNUM_INPUTS{8};
static constexpr int32_t kNUM_OUTPUTS{2};

private:
std::string mLayerName;
std::string mNamespace;
int32_t mNumQHeads;
int32_t mNumKVHeads;
int32_t mHeadSize;
float mEps{1e-6f};

};

class Qwen3DeltaAttentionPluginCreator : public nvinfer1::IPluginCreator
{
public:
    Qwen3DeltaAttentionPluginCreator();

    ~Qwen3DeltaAttentionPluginCreator() override = default;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFieldCollection;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugins
} // namespace trt_edgellm
