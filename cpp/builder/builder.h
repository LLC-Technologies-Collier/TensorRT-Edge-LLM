/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <iostream>
#include <fstream>
#include <malloc.h>
#include <cuda_runtime.h>
#include "multimodal/modelTypes.h"
#include <NvInfer.h>
#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>

using Json = nlohmann::json;

namespace trt_edgellm
{

namespace builder
{

/*!
 * @brief Custom Progress Monitor that periodically saves the timing cache and monitors system memory
 */
class BuildProgressMonitor : public nvinfer1::IProgressMonitor
{
public:
    BuildProgressMonitor(nvinfer1::ITimingCache* cache, std::filesystem::path const& cachePath,
        std::chrono::seconds saveInterval = std::chrono::seconds(10))
        : mCache(cache)
        , mCachePath(cachePath)
        , mSaveInterval(saveInterval)
        , mLastSaveTime(std::chrono::steady_clock::now())
        , mLastReleaseTime(std::chrono::steady_clock::now() - std::chrono::seconds(60))
    {
    }

    void phaseStart(char const* phaseName, char const* parentPhase, int32_t nbSteps) noexcept override
    {
        mCurrentPhaseName = phaseName ? phaseName : "Unknown";
        mCurrentPhaseSteps = nbSteps > 0 ? nbSteps : 1;
        std::cout << "[BuildProgressMonitor] Starting phase: " << mCurrentPhaseName << " (" << nbSteps << " steps)" << std::endl;
    }

    void phaseFinish(char const* phaseName) noexcept override
    {
        checkAndSave();
    }

    bool stepComplete(char const* phaseName, int32_t step) noexcept override
    {
        checkAndSave();
        
        float progress = (static_cast<float>(step + 1) / static_cast<float>(mCurrentPhaseSteps)) * 100.0f;
        printf("[BuildProgressMonitor] Phase: %s | Progress: %.2f%% (%d/%d)\n", mCurrentPhaseName.c_str(), progress, step + 1, mCurrentPhaseSteps);

        // Proactive Memory Management: Trim if RAM is low or swap starts being used
        if (isMemoryPressureHigh())
        {
            releaseSpareMemory();
        }

        // Active Memory Guard: Abort if system memory is critically low
        if (isMemoryCritical())
        {
            saveCache(); // Ensure we save before quitting
            return false; // Signal TensorRT to stop
        }
        
        return true;
    }

private:
    bool isMemoryPressureHigh() noexcept
    {
        try {
            std::ifstream meminfo("/proc/meminfo");
            std::string line;
            size_t available_kb = 0;
            size_t swap_free_kb = 0;
            size_t swap_total_kb = 0;
            
            while (std::getline(meminfo, line))
            {
                if (line.compare(0, 13, "MemAvailable:") == 0)
                {
                    std::stringstream ss(line.substr(13));
                    ss >> available_kb;
                }
                else if (line.compare(0, 10, "SwapTotal:") == 0)
                {
                    std::stringstream ss(line.substr(10));
                    ss >> swap_total_kb;
                }
                else if (line.compare(0, 9, "SwapFree:") == 0)
                {
                    std::stringstream ss(line.substr(9));
                    ss >> swap_free_kb;
                }
            }

            // Pressure threshold: < 10GB RAM OR > 5% Swap used
            bool ram_pressure = (available_kb < 10485760);
            bool swap_pressure = (swap_total_kb > 0 && swap_free_kb < (swap_total_kb * 95 / 100));
            
            return ram_pressure || swap_pressure;
        } catch (...) {
            return false;
        }
    }

    bool isMemoryCritical() noexcept
    {
        try {
            std::ifstream meminfo("/proc/meminfo");
            std::string line;
            size_t available_kb = 0;
            size_t swap_free_kb = 0;
            size_t swap_total_kb = 0;
            
            while (std::getline(meminfo, line))
            {
                if (line.compare(0, 13, "MemAvailable:") == 0)
                {
                    std::stringstream ss(line.substr(13));
                    ss >> available_kb;
                }
                else if (line.compare(0, 10, "SwapTotal:") == 0)
                {
                    std::stringstream ss(line.substr(10));
                    ss >> swap_total_kb;
                }
                else if (line.compare(0, 9, "SwapFree:") == 0)
                {
                    std::stringstream ss(line.substr(9));
                    ss >> swap_free_kb;
                }
            }

            // RAM is critically low (< 1GB)
            bool ram_critical = (available_kb < 1048576);

            // Swap is critically low (usage reached 20%, i.e. Free < 80% of Total)
            bool swap_critical = (swap_total_kb > 0 && swap_free_kb < (swap_total_kb * 8 / 10));

            if (ram_critical || swap_critical)
            {
                std::cout << "[BuildProgressMonitor] CRITICAL MEMORY DETECTED:" << std::endl;
                if (ram_critical) std::cout << "  - Available RAM: " << (available_kb / 1024) << " MB (threshold: 1024 MB)" << std::endl;
                if (swap_critical) std::cout << "  - Swap Free: " << (swap_free_kb / 1024) << " MB (threshold: " << (swap_total_kb * 8 / 1024 / 10) << " MB)" << std::endl;
                std::cout << "[BuildProgressMonitor] Aborting build to save progress and prevent crash." << std::endl;
                return true;
            }
        } catch (...) {
            return false;
        }
        return false;
    }

    void checkAndSave() noexcept
    {
        auto now = std::chrono::steady_clock::now();
        if (now - mLastSaveTime >= mSaveInterval)
        {
            saveCache();
            mLastSaveTime = now;
        }
    }

    void releaseSpareMemory() noexcept
    {
        auto now = std::chrono::steady_clock::now();
        if (now - mLastReleaseTime < std::chrono::seconds(30))
        {
            return;
        }
        mLastReleaseTime = now;

        std::cout << "[BuildProgressMonitor] High memory pressure. Triggering proactive release." << std::endl;
        try {
            // 1. Synchronize to ensure all profiling/tactics are done
            cudaDeviceSynchronize();

            // 2. Trim CUDA memory pool (if using async allocator)
            int device = 0;
            if (cudaGetDevice(&device) == cudaSuccess)
            {
                cudaMemPool_t memPool;
                if (cudaDeviceGetDefaultMemPool(&memPool, device) == cudaSuccess)
                {
                    cudaMemPoolTrimTo(memPool, 0);
                }
            }

            // 3. Trim host heap
#ifdef __linux__
            malloc_trim(0);
#endif
        } catch (...) {
            // ignore
        }
    }

    void saveCache() noexcept
    {
        if (mCache && !mCachePath.empty())
        {
            try
            {
                auto blob = std::unique_ptr<nvinfer1::IHostMemory>(mCache->serialize());
                if (blob)
                {
                    auto tmpPath = mCachePath;
                    tmpPath += ".tmp";
                    {
                        std::ofstream cacheFile(tmpPath, std::ios::binary);
                        if (cacheFile)
                        {
                            cacheFile.write(static_cast<char*>(blob->data()), blob->size());
                            cacheFile.flush();
                            cacheFile.close();
                        }
                    }
                    std::filesystem::rename(tmpPath, mCachePath);
                    std::cout << "[BuildProgressMonitor] Timing cache flushed to " << mCachePath << std::endl;
                }

                // Release memory back to OS
                releaseSpareMemory();
            }
            catch (...)
            {
                // Suppress exceptions in noexcept
            }
        }
    }

    nvinfer1::ITimingCache* mCache;
    std::filesystem::path mCachePath;
    std::chrono::seconds mSaveInterval;
    std::chrono::steady_clock::time_point mLastSaveTime;
    std::chrono::steady_clock::time_point mLastReleaseTime;
    std::string mCurrentPhaseName;
    int32_t mCurrentPhaseSteps{1};
};

//! Configuration structure for LLM model building.
//! Contains all parameters needed to configure the TensorRT engine building process
//! for Large Language Models, including standard LLMs and Eagle models.
struct LLMBuilderConfig
{
    int64_t maxInputLen{1024};        //!< Maximum input sequence length for the model
    bool eagleDraft{false};           //!< Whether this is an Eagle draft model
    bool eagleBase{false};            //!< Whether this is an Eagle base model
    int64_t maxBatchSize{4};          //!< Maximum batch size for inference
    int64_t maxLoraRank{0};           //!< Maximum LoRA rank (0 = no LoRA support)
    int64_t maxKVCacheCapacity{4096}; //!< Maximum KV cache capacity (sequence length)
    int64_t maxVerifyTreeSize{60}; //!< Maximum length of input_ids passed into Eagle base model for tree verification
    int64_t maxDraftTreeSize{60};  //!< Maximum length of input_ids passed into Eagle draft model for draft generation

    //! Convert configuration to JSON format for serialization.
    //! @return JSON object containing all configuration parameters
    Json toJson() const
    {
        Json json;
        json["max_input_len"] = maxInputLen;
        json["eagle_draft"] = eagleDraft;
        json["eagle_base"] = eagleBase;
        json["max_batch_size"] = maxBatchSize;
        json["max_lora_rank"] = maxLoraRank;
        json["max_kv_cache_capacity"] = maxKVCacheCapacity;
        // Only include Eagle-specific fields when Eagle is enabled
        if (eagleBase)
        {
            json["max_verify_tree_size"] = maxVerifyTreeSize;
        }
        if (eagleDraft)
        {
            json["max_draft_tree_size"] = maxDraftTreeSize;
        }
        return json;
    }

    //! Create configuration from JSON format.
    //! @param json JSON object containing configuration parameters
    //! @return LLMBuilderConfig object with parsed parameters
    static LLMBuilderConfig fromJson(Json const& json)
    {
        LLMBuilderConfig config;
        if (json.contains("max_input_len"))
        {
            config.maxInputLen = json["max_input_len"];
        }
        if (json.contains("eagle_draft"))
        {
            config.eagleDraft = json["eagle_draft"];
        }
        if (json.contains("eagle_base"))
        {
            config.eagleBase = json["eagle_base"];
        }
        if (json.contains("max_batch_size"))
        {
            config.maxBatchSize = json["max_batch_size"];
        }
        if (json.contains("max_lora_rank"))
        {
            config.maxLoraRank = json["max_lora_rank"];
        }
        if (json.contains("max_kv_cache_capacity"))
        {
            config.maxKVCacheCapacity = json["max_kv_cache_capacity"];
        }
        if (json.contains("max_verify_tree_size"))
        {
            config.maxVerifyTreeSize = json["max_verify_tree_size"];
        }
        if (json.contains("max_draft_tree_size"))
        {
            config.maxDraftTreeSize = json["max_draft_tree_size"];
        }
        return config;
    }

    //! Convert configuration to human-readable string format.
    //! @return String representation of the configuration for debugging/logging
    std::string toString() const
    {
        std::ostringstream oss;
        oss << "LLMBuilderConfig:\n";
        oss << "  maxInputLen: " << maxInputLen << "\n";
        oss << "  eagleDraft: " << (eagleDraft ? "true" : "false") << "\n";
        oss << "  eagleBase: " << (eagleBase ? "true" : "false") << "\n";
        oss << "  maxBatchSize: " << maxBatchSize << "\n";
        oss << "  maxLoraRank: " << maxLoraRank << "\n";
        oss << "  maxKVCacheCapacity: " << maxKVCacheCapacity << "\n";
        // Only show Eagle-specific fields when Eagle is enabled
        if (eagleBase)
        {
            oss << "  maxVerifyTreeSize: " << maxVerifyTreeSize << "\n";
        }
        if (eagleDraft)
        {
            oss << "  maxDraftTreeSize: " << maxDraftTreeSize << "\n";
        }
        return oss.str();
    }
};

//! Configuration structure for visual model building.
//! Contains parameters needed to configure the TensorRT engine building process
//! for visual encoders used in Vision-Language Models.
struct VisualBuilderConfig
{
    int64_t minImageTokens{4};           //!< Minimum number of image tokens in a batch
    int64_t maxImageTokens{1024};        //!< Maximum number of image tokens in a batch
    int64_t maxImageTokensPerImage{512}; //!< Maximum number of image tokens per image, used for preprocessing

    //! Convert configuration to JSON format for serialization.
    //! @return JSON object containing all configuration parameters
    Json toJson() const
    {
        Json json;
        json["min_image_tokens"] = minImageTokens;
        json["max_image_tokens"] = maxImageTokens;
        json["max_image_tokens_per_image"] = maxImageTokensPerImage;
        return json;
    }

    //! Create configuration from JSON format.
    //! @param json JSON object containing configuration parameters
    //! @return VisualBuilderConfig object with parsed parameters
    static VisualBuilderConfig fromJson(Json const& json)
    {
        VisualBuilderConfig config;
        if (json.contains("min_image_tokens"))
        {
            config.minImageTokens = json["min_image_tokens"];
        }
        if (json.contains("max_image_tokens"))
        {
            config.maxImageTokens = json["max_image_tokens"];
        }
        if (json.contains("max_image_tokens_per_image"))
        {
            config.maxImageTokensPerImage = json["max_image_tokens_per_image"];
        }
        return config;
    }

    //! Convert configuration to human-readable string format.
    //! @return String representation of the configuration for debugging/logging
    std::string toString() const
    {
        std::ostringstream oss;
        oss << "VisualBuilderConfig:\n";
        oss << "  minImageTokens: " << minImageTokens << "\n";
        oss << "  maxImageTokens: " << maxImageTokens << "\n";
        oss << "  maxImageTokensPerImage: " << maxImageTokensPerImage << "\n";
        return oss.str();
    }
};

//! Builder class for Large Language Model TensorRT engines.
//! Handles the complete process of building TensorRT engines from ONNX models
//! for various types of LLMs including standard models, Eagle models, and VLMs.
class LLMBuilder
{
public:
    //! Constructor for LLMBuilder.
    //! @param onnxDir Directory containing the ONNX model and configuration files
    //! @param engineDir Directory where the built engine and related files will be saved
    //! @param config Configuration object specifying build parameters
    LLMBuilder(
        std::filesystem::path const& onnxDir, std::filesystem::path const& engineDir, LLMBuilderConfig const& config);

    //! Destructor.
    ~LLMBuilder() = default;

    //! Build the TensorRT engine from the ONNX model.
    //! This method performs the complete build process including:
    //! - Loading and parsing the ONNX model
    //! - Setting up optimization profiles
    //! - Building the TensorRT engine
    //! - Copying necessary files to the engine directory
    //! @return true if build was successful, false otherwise
    bool build();

private:
    std::filesystem::path mOnnxDir;   //!< Directory containing ONNX model files
    std::filesystem::path mEngineDir; //!< Directory for saving built engine
    LLMBuilderConfig mBuilderConfig;  //!< Build configuration

    //! Parse the model configuration from config.json.
    //! Extracts model dimensions and parameters needed for optimization profile setup.
    //! @return true if parsing was successful, false otherwise
    bool parseConfig();

    //! Set up optimization profiles for LLM models.
    //! Creates context and generation profiles with appropriate dynamic shapes.
    //! @param builder TensorRT builder object
    //! @param config TensorRT builder config object
    //! @param network TensorRT network definition
    //! @return true if setup was successful, false otherwise
    bool setupLLMOptimizationProfiles(
        nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, nvinfer1::INetworkDefinition const* network);

    //! Set up common optimization profiles shared by all LLM types.
    //! Configures context lengths, rotary embeddings, and KV cache profiles.
    //! @param contextProfile Optimization profile for context processing
    //! @param generationProfile Optimization profile for generation processing
    //! @return true if setup was successful, false otherwise
    bool setupCommonProfiles(
        nvinfer1::IOptimizationProfile* contextProfile, nvinfer1::IOptimizationProfile* generationProfile);

    //! Set up optimization profiles for vanilla (non-Eagle) LLM models.
    //! Configures input IDs and last token IDs for standard transformer models.
    //! @param contextProfile Optimization profile for context processing
    //! @param generationProfile Optimization profile for generation processing
    //! @return true if setup was successful, false otherwise
    bool setupVanillaProfiles(
        nvinfer1::IOptimizationProfile* contextProfile, nvinfer1::IOptimizationProfile* generationProfile);

    //! Set up optimization profiles for Eagle models.
    //! Configures Eagle-specific inputs like hidden states and attention masks.
    //! @param contextProfile Optimization profile for context processing
    //! @param generationProfile Optimization profile for generation processing
    //! @return true if setup was successful, false otherwise
    bool setupEagleProfiles(
        nvinfer1::IOptimizationProfile* contextProfile, nvinfer1::IOptimizationProfile* generationProfile);

    //! Set up optimization profiles for Deepstack embeddings (Qwen3VL).
    //! Configures deepstack embedding inputs with the same profile as inputs_embeds.
    //! @param contextProfile Optimization profile for context processing
    //! @param generationProfile Optimization profile for generation processing
    //! @param network TensorRT network definition for input analysis
    //! @return true if setup was successful, false otherwise
    bool setupDeepstackProfiles(nvinfer1::IOptimizationProfile* contextProfile,
        nvinfer1::IOptimizationProfile* generationProfile, nvinfer1::INetworkDefinition const* network);

    //! Set up optimization profiles for LoRA-enabled models.
    //! Configures LoRA weight matrices with dynamic rank support.
    //! @param contextProfile Optimization profile for context processing
    //! @param generationProfile Optimization profile for generation processing
    //! @param network TensorRT network definition for LoRA input analysis
    //! @return true if setup was successful, false otherwise
    bool setupLoraProfiles(nvinfer1::IOptimizationProfile* contextProfile,
        nvinfer1::IOptimizationProfile* generationProfile, nvinfer1::INetworkDefinition const* network);

    //! Set up optimization profiles for KV cache tensors.
    //! Configures dynamic shapes for key-value cache inputs across all layers.
    //! @param contextProfile Optimization profile for context processing
    //! @param generationProfile Optimization profile for generation processing
    //! @return true if setup was successful, false otherwise
    bool setupKVCacheProfiles(
        nvinfer1::IOptimizationProfile* contextProfile, nvinfer1::IOptimizationProfile* generationProfile);

    //! Copy and save the model configuration with builder config.
    //! Creates a config.json file in the engine directory with both original model config
    //! and builder configuration parameters.
    //! @return true if copying was successful, false otherwise
    bool copyConfig();

    //! Copy tokenizer files to the engine directory.
    //! Copies tokenizer_config.json and tokenizer.json files needed for inference.
    //! @return true if copying was successful, false otherwise
    bool copyTokenizerFiles();

    //! Copy Eagle-specific files to the engine directory.
    //! Copies d2t.safetensors file for Eagle3 draft models.
    //! @return true if copying was successful, false otherwise
    bool copyEagleFiles();

    //! Copy vocabulary mapping files to the engine directory.
    //! Copies vocab_map.safetensors file if reduced vocabulary is used.
    //! @return true if copying was successful, false otherwise
    bool copyVocabMappingFiles();

    //! Copy embedding table file to the engine directory.
    //! Copies embedding.safetensors file for eagleBase and vanilla LLM models.
    //! @return true if copying was successful, false otherwise
    bool copyEmbeddingFile();

    // Model dimensions extracted from config.json
    int64_t mHiddenSize{0};                 //!< Hidden size of the model
    int64_t mNumKVHeads{0};                 //!< Number of key-value heads
    int64_t mHeadSize{0};                   //!< Size of each attention head
    int64_t mRotaryDim{0};                  //!< Dimension for rotary position embeddings
    int32_t mNbKVCacheInputs{0};            //!< Number of KV cache inputs (layers)
    int32_t mTargetModelOutputHiddenDim{0}; //!< Target output hidden dimension
    int32_t mNumDeepstackFeatures{0};       //!< Number of deepstack features (for Qwen3VL)
    Json mModelConfig;                      //!< Parsed model configuration
};

//! Builder class for visual encoder TensorRT engines.
//! Handles the complete process of building TensorRT engines from ONNX models
//! for visual encoders used in Vision-Language Models.
class VisualBuilder
{
public:
    //! Constructor for VisualBuilder.
    //! @param onnxDir Directory containing the ONNX model and configuration files
    //! @param engineDir Directory where the built engine and related files will be saved
    //! @param config Configuration object specifying build parameters
    VisualBuilder(std::filesystem::path const& onnxDir, std::filesystem::path const& engineDir,
        VisualBuilderConfig const& config);

    //! Destructor.
    ~VisualBuilder() = default;

    //! Build the TensorRT engine from the ONNX model.
    //! This method performs the complete build process including:
    //! - Loading and parsing the ONNX model
    //! - Setting up optimization profiles
    //! - Building the TensorRT engine
    //! - Copying necessary files to the engine directory
    //! @return true if build was successful, false otherwise
    bool build();

private:
    std::filesystem::path mOnnxDir;     //!< Directory containing ONNX model files
    std::filesystem::path mEngineDir;   //!< Directory for saving built engine
    VisualBuilderConfig mBuilderConfig; //!< Build configuration
    multimodal::ModelType mModelType;   //!< Model type inferred from config.json

    //! Parse the model configuration from config.json.
    //! Extracts model type and dimensions needed for optimization profile setup.
    //! @return true if parsing was successful, false otherwise
    bool parseConfig();

    //! Set up optimization profile for visual models.
    //! Creates a single optimization profile with appropriate dynamic shapes.
    //! @param builder TensorRT builder object
    //! @param config TensorRT builder config object
    //! @param network TensorRT network definition
    //! @return true if setup was successful, false otherwise
    bool setupVisualOptimizationProfile(
        nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, nvinfer1::INetworkDefinition const* network);

    //! Set up optimization profile for Qwen ViT models.
    //! Configures inputs for Qwen2-VL and Qwen2.5-VL visual encoders.
    //! @param profile Optimization profile to configure
    //! @param network TensorRT network definition for input analysis
    //! @return true if setup was successful, false otherwise
    bool setupQwenViTProfile(nvinfer1::IOptimizationProfile* profile, nvinfer1::INetworkDefinition const* network);

    //! Set up optimization profile for InternVL or Phi4-MM ViT models.
    //! Configures inputs for InternVL or Phi4-MM visual encoders.
    //! @param profile Optimization profile to configure
    //! @return true if setup was successful, false otherwise
    bool setupInternPhi4ViTProfile(nvinfer1::IOptimizationProfile* profile);

    //! Copy and save the model configuration with builder config.
    //! Creates a config.json file in the engine directory with both original model config
    //! and builder configuration parameters.
    //! @return true if copying was successful, false otherwise
    bool copyConfig();

    // Model dimensions extracted from config.json
    int64_t mNumChannels{0};   //!< Number of input channels
    int64_t mImageSizeH{0};    //!< Image height
    int64_t mImageSizeW{0};    //!< Image width
    int64_t mInputDim{0};      //!< Input dimension for Qwen models
    int64_t mRopeEmbedSize{0}; //!< Rotary position embedding size
    Json mModelConfig;         //!< Parsed model configuration
};

} // namespace builder
} // namespace trt_edgellm
