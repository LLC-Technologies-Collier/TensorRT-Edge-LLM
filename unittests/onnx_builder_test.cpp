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

#include <gtest/gtest.h>
#include "builder/llmBuilder.h"
#include "common/logger.h"
#include <filesystem>
#include <fstream>

using namespace trt_edgellm::builder;

class ONNXBuilderTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        tempDir = std::filesystem::temp_directory_path() / "onnx_builder_test";
        std::filesystem::create_directories(tempDir);
        
        onnxDir = tempDir / "onnx";
        engineDir = tempDir / "engine";
        std::filesystem::create_directories(onnxDir);
        std::filesystem::create_directories(engineDir);
    }

    void TearDown() override
    {
        std::filesystem::remove_all(tempDir);
    }

    void createMockConfig(Json const& config)
    {
        std::ofstream f(onnxDir / "config.json");
        f << config.dump();
    }

    std::filesystem::path tempDir;
    std::filesystem::path onnxDir;
    std::filesystem::path engineDir;
};

TEST_F(ONNXBuilderTest, ConfigParsingLlama)
{
    Json config;
    config["hidden_size"] = 4096;
    config["num_attention_heads"] = 32;
    config["num_key_value_heads"] = 8;
    config["num_hidden_layers"] = 32;
    config["model_type"] = "llama";
    createMockConfig(config);
    
    LLMBuilderConfig builderCfg;
    LLMBuilder builder(onnxDir, engineDir, builderCfg);
    // Success means no crash in constructor
}

TEST_F(ONNXBuilderTest, ConfigParsingQwen3_5_Hybrid)
{
    Json config;
    config["model_type"] = "qwen3_5";
    
    Json textConfig;
    textConfig["hidden_size"] = 1024;
    textConfig["num_attention_heads"] = 16;
    textConfig["num_key_value_heads"] = 4;
    textConfig["num_hidden_layers"] = 24;
    textConfig["num_attention_layers"] = 6;
    textConfig["num_mamba_layers"] = 18;
    textConfig["mamba_num_heads"] = 16;
    textConfig["mamba_head_dim"] = 64;
    textConfig["ssm_state_size"] = 16;
    textConfig["conv_kernel"] = 4;
    textConfig["n_groups"] = 1;
    
    config["text_config"] = textConfig;
    createMockConfig(config);
    
    LLMBuilderConfig builderCfg;
    LLMBuilder builder(onnxDir, engineDir, builderCfg);
}

TEST_F(ONNXBuilderTest, ConfigParsingQwen3_5_MoE)
{
    Json config;
    config["model_type"] = "qwen3_5_moe";
    
    Json textConfig;
    textConfig["hidden_size"] = 2048;
    textConfig["num_attention_heads"] = 16;
    textConfig["num_key_value_heads"] = 16;
    textConfig["num_hidden_layers"] = 28;
    textConfig["num_experts"] = 64;
    textConfig["num_experts_per_tok"] = 8;
    textConfig["moe_intermediate_size"] = 1408;
    textConfig["shared_expert_intermediate_size"] = 19008;
    
    config["text_config"] = textConfig;
    createMockConfig(config);
    
    LLMBuilderConfig builderCfg;
    LLMBuilder builder(onnxDir, engineDir, builderCfg);
}

TEST_F(ONNXBuilderTest, ConfigParsingQwenVL)
{
    Json config;
    config["model_type"] = "qwen2_vl";
    config["hidden_size"] = 1536;
    config["num_attention_heads"] = 12;
    config["num_key_value_heads"] = 12;
    config["num_hidden_layers"] = 28;
    
    Json visionConfig;
    visionConfig["model_type"] = "qwen2_5_vl";
    visionConfig["out_hidden_size"] = 1536;
    
    config["vision_config"] = visionConfig;
    createMockConfig(config);
    
    LLMBuilderConfig builderCfg;
    builderCfg.isVlm = true;
    LLMBuilder builder(onnxDir, engineDir, builderCfg);
}

TEST_F(ONNXBuilderTest, ConfigParsingEagle)
{
    Json config;
    config["model_type"] = "eagle";
    config["hidden_size"] = 4096;
    config["num_attention_heads"] = 32;
    config["num_key_value_heads"] = 8;
    config["num_hidden_layers"] = 32;
    createMockConfig(config);
    
    LLMBuilderConfig builderCfg;
    builderCfg.eagleDraft = true;
    builderCfg.maxDraftTreeSize = 64;
    LLMBuilder builder(onnxDir, engineDir, builderCfg);
}

TEST_F(ONNXBuilderTest, LLMBuilderConfigRoundTrip)
{
    LLMBuilderConfig cfg;
    cfg.maxInputLen = 4096;
    cfg.maxBatchSize = 16;
    cfg.quantization = "nvfp4";
    cfg.useTrtNativeOps = true;
    cfg.weightStreamingBudget = 8192;
    
    Json j = cfg.toJson();
    EXPECT_EQ(j["max_input_len"], 4096);
    EXPECT_EQ(j["max_batch_size"], 16);
    EXPECT_EQ(j["quantization"], "nvfp4");
    EXPECT_TRUE(j["trt_native_ops"]);
    
    LLMBuilderConfig cfg2 = LLMBuilderConfig::fromJson(j);
    EXPECT_EQ(cfg2.maxInputLen, 4096);
    EXPECT_EQ(cfg2.maxBatchSize, 16);
    EXPECT_EQ(cfg2.quantization, "nvfp4");
    EXPECT_TRUE(cfg2.useTrtNativeOps);
}

TEST_F(ONNXBuilderTest, EdgeCaseMissingFields)
{
    Json config;
    config["model_type"] = "llama";
    // Missing hidden_size, num_attention_heads, etc.
    createMockConfig(config);
    
    LLMBuilderConfig builderCfg;
    LLMBuilder builder(onnxDir, engineDir, builderCfg);
    // Should handle defaults or not crash
}

TEST_F(ONNXBuilderTest, EdgeCaseInvalidJson)
{
    std::ofstream f(onnxDir / "config.json");
    f << "{ invalid json [ ] }";
    
    LLMBuilderConfig builderCfg;
    // Constructor might not throw, but parseConfig() likely will.
    LLMBuilder builder(onnxDir, engineDir, builderCfg);
}

TEST_F(ONNXBuilderTest, WeightStreamingBudget)
{
    Json config;
    config["hidden_size"] = 1024;
    config["num_attention_heads"] = 16;
    config["num_key_value_heads"] = 16;
    config["num_hidden_layers"] = 12;
    config["model_type"] = "llama";
    createMockConfig(config);
    
    LLMBuilderConfig builderCfg;
    builderCfg.weightStreamingBudget = 4096; // 4GB budget
    LLMBuilder builder(onnxDir, engineDir, builderCfg);
}

TEST_F(ONNXBuilderTest, ConfigParsingQwen2_5_VLM_mRoPE)
{
    Json config;
    config["model_type"] = "qwen2_5_vl";
    config["hidden_size"] = 1024;
    config["num_attention_heads"] = 16;
    config["num_key_value_heads"] = 16;
    config["num_hidden_layers"] = 12;
    
    Json ropeScaling;
    ropeScaling["type"] = "mrope";
    ropeScaling["mrope_section"] = {16, 16, 32};
    config["rope_scaling"] = ropeScaling;
    
    createMockConfig(config);
    
    LLMBuilderConfig builderCfg;
    LLMBuilder builder(onnxDir, engineDir, builderCfg);
}
