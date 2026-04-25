/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include "plugins/qwen3DeltaAttentionPlugin/qwen3DeltaAttentionPlugin.h"
#include <memory>

using namespace trt_edgellm::plugins;

class Qwen3DeltaAttentionPluginTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // 0.8B Config: 16 heads, 128 head_dim
        plugin = std::make_unique<Qwen3DeltaAttentionPlugin>("test_layer", 16, 16, 128);
    }

    std::unique_ptr<Qwen3DeltaAttentionPlugin> plugin;
};

TEST_F(Qwen3DeltaAttentionPluginTest, MetadataBasic)
{
    EXPECT_EQ(plugin->getNbOutputs(), 2);
    EXPECT_STREQ(plugin->getPluginType(), "qwen3_delta_attention");
    EXPECT_STREQ(plugin->getPluginVersion(), "1");
}

TEST_F(Qwen3DeltaAttentionPluginTest, SerializationRoundTrip)
{
    size_t size = plugin->getSerializationSize();
    std::vector<char> buffer(size);
    plugin->serialize(buffer.data());
    
    Qwen3DeltaAttentionPlugin plugin2("test_layer", buffer.data(), size);
    EXPECT_EQ(plugin2.getSerializationSize(), size);
    EXPECT_STREQ(plugin2.getPluginType(), "qwen3_delta_attention");
}

TEST_F(Qwen3DeltaAttentionPluginTest, FormatCombination)
{
    nvinfer1::PluginTensorDesc desc[10];
    for(int i=0; i<10; ++i) {
        desc[i].type = nvinfer1::DataType::kHALF;
        desc[i].format = nvinfer1::TensorFormat::kLINEAR;
    }
    
    // Pos 0: Q
    EXPECT_TRUE(plugin->supportsFormatCombination(0, desc, 8, 2));
    
    // Pos 5: State (Input)
    EXPECT_TRUE(plugin->supportsFormatCombination(5, desc, 8, 2));
    
    // Pos 8: Output
    EXPECT_TRUE(plugin->supportsFormatCombination(8, desc, 8, 2));
    
    // Test invalid format
    desc[0].format = nvinfer1::TensorFormat::kCHW32;
    EXPECT_FALSE(plugin->supportsFormatCombination(0, desc, 8, 2));
    
    // Test invalid type
    desc[0].format = nvinfer1::TensorFormat::kLINEAR;
    desc[0].type = nvinfer1::DataType::kINT8;
    EXPECT_FALSE(plugin->supportsFormatCombination(0, desc, 8, 2));
}

TEST_F(Qwen3DeltaAttentionPluginTest, Clone)
{
    auto cloned = plugin->clone();
    ASSERT_NE(cloned, nullptr);
    EXPECT_STREQ(cloned->getPluginType(), plugin->getPluginType());
    cloned->destroy();
}
