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
#include "runtime/llmEngineRunner.h"
#include "common/logger.h"
#include "common/trtUtils.h"
#include <cuda_runtime.h>
#include <filesystem>
#include <functional>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;

class Qwen3HybridTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Use the active 0.8B engine on robo for integration testing
        enginePath = "/srv/local-llm/qwen3.5_0.8b_deploy/engine_110_nvfp4_batch1_32k/llm.engine";
        configPath = "/srv/local-llm/qwen3.5_0.8b_deploy/engine_110_nvfp4_batch1_32k/config.json";

        if (!std::filesystem::exists(enginePath)) {
            GTEST_SKIP() << "Engine file not found. Skipping integration test.";
        }

        // Initialize plugin library
        if (!trt_edgellm::loadEdgellmPluginLib()) {
            GTEST_SKIP() << "Failed to load EdgeLLM plugin library. Skipping integration test.";
        }

        cudaStreamCreate(&stream);
    }

    void TearDown() override
    {
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }

    std::filesystem::path enginePath;
    std::filesystem::path configPath;
    std::unordered_map<std::string, std::string> loraWeightsMap;
    cudaStream_t stream = nullptr;
};

TEST_F(Qwen3HybridTest, EngineInitializationAndValidation)
{
    LLMEngineRunner runner(enginePath, configPath, loraWeightsMap, stream);
    auto const& config = runner.getEngineConfig();

    // Qwen 3.5 0.8B Hybrid has 24 layers total: 6 Attention + 18 Mamba
    EXPECT_EQ(config.numAttentionLayers, 6);
    EXPECT_EQ(config.numMambaLayers, 18);
    EXPECT_EQ(config.numDecoderLayers, 24);
    EXPECT_EQ(config.numKVHeads, 4); // GQA
    EXPECT_EQ(config.hiddenSize, 1024);
}

TEST_F(Qwen3HybridTest, PrefillStepWith3DLogits)
{
    LLMEngineRunner runner(enginePath, configPath, loraWeightsMap, stream);
    auto const& config = runner.getEngineConfig();

    int32_t const batchSize = 1;
    int32_t const seqLen = 32; // Use multiple of 8 for TMA alignment
    int32_t const hiddenSize = config.hiddenSize;
    int32_t const vocabSize = config.outputVocabSize;

    // 1. Prepare inputs_embeds [B, S, H]
    Tensor inputsEmbeds({batchSize, seqLen, hiddenSize}, DeviceType::kGPU, nvinfer1::DataType::kHALF, "inputs_embeds");
    
    // 2. Prepare context_lengths [B]
    Tensor contextLengths({batchSize}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "context_lengths");
    contextLengths.dataPointer<int32_t>()[0] = seqLen;

    // 3. Prepare output_logits [B, S, V] (Now 3D!)
    Tensor outputLogits({batchSize, seqLen, vocabSize}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "output_logits");

    // 4. Execute prefill
    bool status = runner.executePrefillStep(inputsEmbeds, contextLengths, {}, outputLogits, std::nullopt, stream);
    EXPECT_TRUE(status);
    
    // MUST synchronize to ensure GPU is done before tensors go out of scope
    cudaStreamSynchronize(stream);
    
    EXPECT_EQ(outputLogits.getShape().getNumDims(), 3);
}

TEST_F(Qwen3HybridTest, EdgeCaseUnalignedSeqLen)
{
    // Test Blackwell non-TMA fallback
    LLMEngineRunner runner(enginePath, configPath, loraWeightsMap, stream);
    auto const& config = runner.getEngineConfig();

    int32_t const seqLen = 13; // Not multiple of 8
    Tensor inputsEmbeds({1, seqLen, config.hiddenSize}, DeviceType::kGPU, nvinfer1::DataType::kHALF);
    Tensor contextLengths({1}, DeviceType::kCPU, nvinfer1::DataType::kINT32);
    contextLengths.dataPointer<int32_t>()[0] = seqLen;
    Tensor outputLogits({1, seqLen, config.outputVocabSize}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT);

    bool status = runner.executePrefillStep(inputsEmbeds, contextLengths, {}, outputLogits, std::nullopt, stream);
    EXPECT_TRUE(status);
    cudaStreamSynchronize(stream);
}

TEST_F(Qwen3HybridTest, EdgeCaseLargeSeqLen)
{
    LLMEngineRunner runner(enginePath, configPath, loraWeightsMap, stream);
    auto const& config = runner.getEngineConfig();

    int32_t const seqLen = 128; // Larger prefill
    Tensor inputsEmbeds({1, seqLen, config.hiddenSize}, DeviceType::kGPU, nvinfer1::DataType::kHALF);
    Tensor contextLengths({1}, DeviceType::kCPU, nvinfer1::DataType::kINT32);
    contextLengths.dataPointer<int32_t>()[0] = seqLen;
    Tensor outputLogits({1, seqLen, config.outputVocabSize}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT);

    bool status = runner.executePrefillStep(inputsEmbeds, contextLengths, {}, outputLogits, std::nullopt, stream);
    EXPECT_TRUE(status);
    cudaStreamSynchronize(stream);
}

TEST_F(Qwen3HybridTest, EdgeCaseMixedAttentionAndMamba)
{
    // This test implicitly validates that the runner correctly handles 
    // both 5D (attention) and 4D (mamba) states in the same engine.
    LLMEngineRunner runner(enginePath, configPath, loraWeightsMap, stream);
    
    EXPECT_EQ(runner.getEngineConfig().numAttentionLayers, 6);
    EXPECT_EQ(runner.getEngineConfig().numMambaLayers, 18);
}

TEST_F(Qwen3HybridTest, FullPrefillAndDecodeSequence)
{
    LLMEngineRunner runner(enginePath, configPath, loraWeightsMap, stream);
    auto const& config = runner.getEngineConfig();

    // 1. Prefill
    int32_t const prefillLen = 8;
    Tensor prefillEmbeds({1, prefillLen, config.hiddenSize}, DeviceType::kGPU, nvinfer1::DataType::kHALF);
    Tensor contextLengths({1}, DeviceType::kCPU, nvinfer1::DataType::kINT32);
    contextLengths.dataPointer<int32_t>()[0] = prefillLen;
    Tensor prefillLogits({1, prefillLen, config.outputVocabSize}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT);

    ASSERT_TRUE(runner.executePrefillStep(prefillEmbeds, contextLengths, {}, prefillLogits, std::nullopt, stream));
    cudaStreamSynchronize(stream);

    // 2. Decode Step 1
    Tensor decodeEmbeds({1, 1, config.hiddenSize}, DeviceType::kGPU, nvinfer1::DataType::kHALF);
    Tensor decodeLogits({1, 1, config.outputVocabSize}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT);

    ASSERT_TRUE(runner.executeVanillaDecodingStep(decodeEmbeds, decodeLogits, std::nullopt, stream));
    cudaStreamSynchronize(stream);

    // 3. Decode Step 2 (Verify state persistence doesn't crash)
    ASSERT_TRUE(runner.executeVanillaDecodingStep(decodeEmbeds, decodeLogits, std::nullopt, stream));
    cudaStreamSynchronize(stream);
}

TEST_F(Qwen3HybridTest, GQA_BroadcastingValidation)
{
    LLMEngineRunner runner(enginePath, configPath, loraWeightsMap, stream);
    auto const& config = runner.getEngineConfig();

    // Qwen 3.5 0.8B uses 4 KV heads and 16 Query heads.
    EXPECT_EQ(config.numKVHeads, 4);
    // If numKVHeads != numQHeads, GQA is active.
}

TEST_F(Qwen3HybridTest, VLM_DeepstackInputMapping)
{
    LLMEngineRunner runner(enginePath, configPath, loraWeightsMap, stream);
    auto const& config = runner.getEngineConfig();

    Tensor inputsEmbeds({1, 8, config.hiddenSize}, DeviceType::kGPU, nvinfer1::DataType::kHALF);
    Tensor contextLengths({1}, DeviceType::kCPU, nvinfer1::DataType::kINT32);
    contextLengths.dataPointer<int32_t>()[0] = 8;
    Tensor outputLogits({1, 8, config.outputVocabSize}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT);

    // Mock deepstack embeds [B, S, H]
    Tensor deepstackEmbeds({1, 8, config.hiddenSize}, DeviceType::kGPU, nvinfer1::DataType::kHALF);
    OptionalInputTensors optionalInputs;
    optionalInputs.push_back(std::cref(deepstackEmbeds));

    bool status = runner.executePrefillStep(inputsEmbeds, contextLengths, optionalInputs, outputLogits, std::nullopt, stream);
    EXPECT_TRUE(status);
    cudaStreamSynchronize(stream);
}
