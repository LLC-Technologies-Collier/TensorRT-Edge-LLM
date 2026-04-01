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

//! \brief TensorRT plugin for Qwen 3.5 Gated Delta Attention
//!
//! Registered as "qwen3_delta_attention" under the "trt_edgellm" ONNX domain.
//!
//! Implements the gated delta rule linear attention from Qwen 3.5:
//!   WY representation based chunked linear attention.
//!
//! Inputs:
//!   [0] q          [batch, (seq_len,) n_q_heads, head_size]
//!   [1] k          [batch, (seq_len,) n_kv_heads, head_size]
//!   [2] v          [batch, (seq_len,) n_kv_heads, head_size]
//!   [3] g          [batch, (seq_len,) n_q_heads]           (forget gate)
//!   [4] beta       [batch, (seq_len,) n_q_heads]           (update rate)
//!   [5] state_in   [batch, n_q_heads, head_size, head_size] (recurrent state)
//!
//! Outputs:
//!   [0] output     [batch, (seq_len,) n_q_heads, head_size]
//!   [1] state_out  [batch, n_q_heads, head_size, head_size]
class Qwen3DeltaAttentionPlugin : public nvinfer1::IPluginV3,
                                  public nvinfer1::IPluginV3OneCore,
                                  public nvinfer1::IPluginV3OneBuild,
                                  public nvinfer1::IPluginV3OneRuntime
{
public:
    Qwen3DeltaAttentionPlugin(std::string const& name, int32_t numQHeads, int32_t numKVHeads, int32_t headSize);

    Qwen3DeltaAttentionPlugin() = delete;
    Qwen3DeltaAttentionPlugin(Qwen3DeltaAttentionPlugin const&) = delete;
    ~Qwen3DeltaAttentionPlugin() override;

    // IPluginV3OneCore
    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;

    // IPluginV3OneBuild
    int32_t getNbOutputs() const noexcept override;
    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs, nvinfer1::DataType const* inputTypes,
        int32_t nbInputs) const noexcept override;
    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override;
    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    // IPluginV3OneRuntime
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs, nvinfer1::PluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;

protected:
    std::string mLayerName;
    std::string mNamespace;

    int32_t mNumQHeads{};
    int32_t mNumKVHeads{};
    int32_t mHeadSize{};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
};

class Qwen3DeltaAttentionPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    Qwen3DeltaAttentionPluginCreator();
    ~Qwen3DeltaAttentionPluginCreator() override = default;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;
    char const* getPluginNamespace() const noexcept override;
    nvinfer1::IPluginV3* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFieldCollection;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugins
} // namespace trt_edgellm
