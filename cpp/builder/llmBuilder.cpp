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

#include "llmBuilder.h"
#include "builderUtils.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/fileUtils.h"
#include "common/logger.h"
#include "common/mathUtils.h"
#include "common/trtUtils.h"
#include "common/version.h"
#include <NvOnnxParser.h>
#include <filesystem>
#include <fstream>

namespace trt_edgellm
{
namespace builder
{

LLMBuilder::LLMBuilder(
    std::filesystem::path const& onnxDir, std::filesystem::path const& engineDir, LLMBuilderConfig const& config)
    : mOnnxDir(onnxDir)
    , mEngineDir(engineDir)
    , mBuilderConfig(config)
{
    printf("[Builder] LLMBuilder constructor entered\n");
}

LLMBuilder::~LLMBuilder() noexcept
{
    printf("[Builder] LLMBuilder destructor entered\n");
}

bool LLMBuilder::build()
{
    printf("================================================================================\n");
    printf("[Builder] LLMBuilder::build() ENTERED (VERSION_MARKER_V105)\n");
    printf("================================================================================\n");
    // Load plugin library
    void* pluginHandles = loadEdgellmPluginLib();
    if (!pluginHandles)
    {
        printf("[Builder] FAILED to load plugin library\n");
        LOG_ERROR("LLMBuilder::build(): FAILED to load plugin library. Engine build will likely fail.");
        return false;
    }

    // Parse model config
    if (!parseConfig())
    {
        printf("[Builder] FAILED to parse model config\n");
        return false;
    }

    // Create builder and network
    auto [builder, network] = createBuilderAndNetwork();
    if (!builder || !network)
    {
        printf("[Builder] FAILED to create builder/network\n");
        return false;
    }

    // List all registered plugins for debugging
    auto& registry = *getPluginRegistry();
    int32_t nbCreators = 0;
    nvinfer1::IPluginCreator* const* list = registry.getPluginCreatorList(&nbCreators);
    printf("[Builder] Registered TensorRT Plugins (%d found):\n", nbCreators);
    for (int32_t i = 0; i < nbCreators; ++i)
    {
        if (list[i])
        {
            printf("  - %s (Namespace: '%s', Version: %s)\n", 
                   list[i]->getPluginName(), 
                   list[i]->getPluginNamespace(), 
                   list[i]->getPluginVersion());
        }
    }
    // Determine ONNX file path
    std::string onnxFilePath;
    if (mBuilderConfig.maxLoraRank > 0)
    {
        onnxFilePath = (mOnnxDir / "lora_model.onnx").string();
        printf("[Builder] Parsing LoRA-enabled ONNX model: %s\n", onnxFilePath.c_str());
    }
    else
    {
        onnxFilePath = (mOnnxDir / "model.onnx").string();
        printf("[Builder] Parsing ONNX model: %s\n", onnxFilePath.c_str());
    }

    // Set up parser
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, trt_edgellm::gLogger));
    if (!parser)
    {
        printf("[Builder] FAILED to create ONNX parser\n");
        return false;
    }

    printf("[Builder] ONNX parser created. Starting parseFromFile for %s\n", onnxFilePath.c_str());
    
    // Attempt to catch parser internal issues before SEGV
    if (!parser->parseFromFile(onnxFilePath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kVERBOSE)))
    {
        for (int32_t i = 0; i < parser->getNbErrors(); ++i)
        {
            auto* error = parser->getError(i);
            printf("[Builder]   Parser Error [%d]: (code %d) %s at %s:%d\n", 
                   i, static_cast<int>(error->code()), error->desc(), error->file(), error->line());
        }
        return false;
    }
    printf("[Builder] ONNX model parsed successfully\n");

    // Print network information
    LOG_DEBUG("%s", printNetworkInfo(network.get(), "LLM").c_str());

    LOG_DEBUG("ONNX parsing complete. mNbKVCacheInputs=%d, mNumMambaLayers=%d", mNbKVCacheInputs, mNumMambaLayers);

    // Create builder config with weight streaming support
    printf("[Builder] Creating builder config...\n");
    auto config = createBuilderConfig(builder.get(), mBuilderConfig);
    if (!config)
    {
        printf("[Builder] FAILED to create builder config\n");
        return false;
    }

    printf("[Builder] Setting up optimization profiles...\n");
    // Setup optimization profiles
    if (!setupLLMOptimizationProfiles(*builder.get(), *config.get(), *network.get()))
    {
        printf("[Builder] FAILED to setup optimization profiles\n");
        return false;
    }
    printf("[Builder] Optimization profiles setup successfully\n");

    // Network diagnostic dump
    printf("[Builder] Network Diagnostic Dump:\n");
    for (int32_t i = 0; i < network->getNbLayers(); ++i)
    {
        auto* layer = network->getLayer(i);
        printf("  Layer %d: name=%s, type=%d\n", i, layer->getName(), static_cast<int>(layer->getType()));
        for (int32_t j = 0; j < layer->getNbInputs(); ++j)
        {
            auto* input = layer->getInput(j);
            if (input == nullptr) {
                printf("    Input %d: NULL!\n", j);
            } else {
                printf("    Input %d: name=%s, dtype=%d, dims=%s\n", 
                       j, input->getName(), static_cast<int>(input->getType()), dimsToString(input->getDimensions()).c_str());
            }
        }
        for (int32_t j = 0; j < layer->getNbOutputs(); ++j)
        {
            auto* output = layer->getOutput(j);
            printf("    Output %d: name=%s, dtype=%d, dims=%s\n", 
                   j, output->getName(), static_cast<int>(output->getType()), dimsToString(output->getDimensions()).c_str());
        }
    }

    // Create engine directory
    if (!std::filesystem::exists(mEngineDir))
    {
        if (!std::filesystem::create_directories(mEngineDir))
        {
            printf("[Builder] FAILED to create directory %s\n", mEngineDir.string().c_str());
            LOG_ERROR("Failed to create directory %s", mEngineDir.string().c_str());
            return false;
        }
        LOG_INFO("Created directory %s for saving LLM engine.", mEngineDir.string().c_str());
    }

    // Determine engine file name
    std::string engineFileName;
    if (mBuilderConfig.eagleDraft)
    {
        engineFileName = "eagle_draft.engine";
    }
    else if (mBuilderConfig.eagleBase)
    {
        engineFileName = "eagle_base.engine";
    }
    else
    {
        engineFileName = "llm.engine";
    }

    // Build and save engine
    std::string const engineFilePath = (mEngineDir / engineFileName).string();
    printf("[Builder] Building and serializing engine to %s (this may take a while)...\n", engineFilePath.c_str());
    if (!buildAndSerializeEngine(builder.get(), network.get(), config.get(), engineFilePath))
    {
        printf("[Builder] FAILED to build/serialize engine\n");
        return false;
    }
    printf("[Builder] Engine built and saved successfully\n");

    // Detect number of deepstack embeds from network (for Qwen3VL models)
    mNumDeepstackFeatures = 0;
    for (int32_t idx = 0; idx < network->getNbInputs(); idx++)
    {
        std::string const inputName = network->getInput(idx)->getName();
        if (inputName.find(binding_names::kDeepstackEmbedsTemplate) != std::string::npos)
        {
            mNumDeepstackFeatures++;
        }
    }
    if (mNumDeepstackFeatures > 0)
    {
        LOG_INFO("Detected %d deepstack embedding inputs in network (Qwen3VL model)", mNumDeepstackFeatures);
    }

    // Copy files and save builder config
    printf("[Builder] Copying configuration and tokenizer files...\n");
    if (!copyConfig())
    {
        printf("[Builder] FAILED to copy config\n");
        return false;
    }

    if (!copyTokenizerFiles())
    {
        printf("[Builder] FAILED to copy tokenizer files\n");
        return false;
    }

    if (!copyEagleFiles())
    {
        printf("[Builder] FAILED to copy Eagle files\n");
        return false;
    }

    if (!copyVocabMappingFiles())
    {
        printf("[Builder] FAILED to copy vocab mapping files\n");
        return false;
    }

    if (!copyEmbeddingFile())
    {
        printf("[Builder] FAILED to copy embedding file\n");
        return false;
    }

    printf("[Builder] LLMBuilder::build() completed successfully\n");
    return true;
}

bool LLMBuilder::parseConfig()
{
    std::string const jsonPath = (mOnnxDir / "config.json").string();
    if (!loadJsonConfig(jsonPath, mModelConfig))
    {
        return false;
    }

    // Check model version
    std::string modelVersion = mModelConfig.value(binding_names::kEdgellmVersion, "");
    version::checkVersion(modelVersion);

    mHiddenSize = mModelConfig["hidden_size"].get<int32_t>();
    mTargetModelOutputHiddenDim = mHiddenSize * 3;
    mNumKVHeads = mModelConfig["num_key_value_heads"].get<int32_t>();
    auto numAttentionHeads = mModelConfig["num_attention_heads"].get<int32_t>();

    if (mModelConfig.contains("head_dim"))
    {
        mHeadSize = mModelConfig["head_dim"].get<int32_t>();
    }
    else
    {
        mHeadSize = mHiddenSize / numAttentionHeads;
    }

    if (mModelConfig.contains("partial_rotary_factor"))
    {
        mRotaryDim = static_cast<int64_t>(mModelConfig["partial_rotary_factor"].get<float>() * mHeadSize);
    }
    else
    {
        mRotaryDim = mHeadSize;
    }
    
    // Ensure mModelConfig has the final values
    mModelConfig["head_dim"] = static_cast<int32_t>(mHeadSize);
    mModelConfig["num_key_value_heads"] = static_cast<int32_t>(mNumKVHeads);

    mNumMambaLayers = mModelConfig.value("num_mamba_layers", 0);
    mMambaNumHeads = mModelConfig.value("mamba_num_heads", 0);
    mMambaHeadDim = mModelConfig.value("mamba_head_dim", 0);
    mSSMStateSize = mModelConfig.value("ssm_state_size", 0);
    mConvDim = mModelConfig.value("conv_dim", 0);
    mConvKernel = mModelConfig.value("conv_kernel", 0);

    // For hybrid models, only attention layers have KV caches
    if (mNumMambaLayers > 0)
    {
        mNbKVCacheInputs = mModelConfig.value("num_attention_layers", mModelConfig["num_hidden_layers"].get<int32_t>());
    }
    else
    {
        mNbKVCacheInputs = mModelConfig["num_hidden_layers"].get<int32_t>();
    }

    // Read trt_native_ops flag from config if present
    if (mModelConfig.contains("trt_native_ops"))
    {
        mBuilderConfig.useTrtNativeOps = mModelConfig["trt_native_ops"].get<bool>();
    }

    return true;
}

bool LLMBuilder::setupLLMOptimizationProfiles(
    nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config, nvinfer1::INetworkDefinition const& network)
{
    printf("[Builder] setupLLMOptimizationProfiles: Using DUAL profiles (Context=0, Generation=1)...\n");
    auto* contextProfile = builder.createOptimizationProfile();
    auto* generationProfile = builder.createOptimizationProfile();

    bool result = true;

    // Setup common profiles
    result &= setupCommonProfiles(*contextProfile, *generationProfile, network);

    // Setup model-specific profiles
    if (mBuilderConfig.eagleBase || mBuilderConfig.eagleDraft)
    {
        result &= setupEagleProfiles(*contextProfile, *generationProfile, network);
    }
    else
    {
        result &= setupVanillaProfiles(*contextProfile, *generationProfile, network);
    }

    // New catch-all pass: Ensure EVERY input has a profile
    for (int32_t i = 0; i < network.getNbInputs(); ++i)
    {
        auto* input = network.getInput(i);
        const char* name = input->getName();
        nvinfer1::Dims dims = input->getDimensions();
        printf("[Builder] setupLLMOptimizationProfiles: Input %d: %s, dims=%s\n", (int)i, name, dimsToString(dims).c_str());
        
        // Check both profiles
        bool inContext = (contextProfile->getDimensions(name, nvinfer1::OptProfileSelector::kMIN).nbDims >= 0);
        bool inGeneration = (generationProfile->getDimensions(name, nvinfer1::OptProfileSelector::kMIN).nbDims >= 0);

        if (!inContext || !inGeneration)
        {
            LOG_DEBUG("Adding catch-all static profile for input: %s (Context=%d, Gen=%d)", name, inContext, inGeneration);
            nvinfer1::Dims dims = input->getDimensions();
            nvinfer1::Dims minDims = dims;
            nvinfer1::Dims optDims = dims;
            nvinfer1::Dims maxDims = dims;

            for (int32_t j = 0; j < dims.nbDims; ++j)
            {
                if (dims.d[j] < 0)
                {
                    minDims.d[j] = 1;
                    optDims.d[j] = (j == 1 || j == 3) ? mHeadSize : 1;
                    maxDims.d[j] = (j == 1 || j == 3) ? mBuilderConfig.maxKVCacheCapacity : 1;
                }
            }

            if (!inContext) {
                contextProfile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN, minDims);
                contextProfile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT, optDims);
                contextProfile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX, maxDims);
            }
            if (!inGeneration) {
                generationProfile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN, minDims);
                generationProfile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT, optDims);
                generationProfile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX, maxDims);
            }
        }
    }


    // Setup Deepstack profiles for Qwen3VL models
    result &= setupDeepstackProfiles(*contextProfile, *generationProfile, network);

    // Setup lm_head_weight profile for CodePredictor (Qwen3-Omni)
    result &= setupLmHeadWeightProfiles(*contextProfile, *generationProfile, network);

    if (mBuilderConfig.maxLoraRank > 0)
    {
        result &= setupLoraProfiles(*contextProfile, *generationProfile, network);
    }

    if (!result)
    {
        LOG_ERROR("Failed to setup optimization profiles");
        return false;
    }

    // CRITICAL: Order matters! Runtime expects Context at index 0 and Generation at index 1.
    int32_t ctxIdx = config.addOptimizationProfile(contextProfile);
    int32_t genIdx = config.addOptimizationProfile(generationProfile);
    
    printf("[Builder] setupLLMOptimizationProfiles: Added Context profile at index %d, Generation profile at index %d\n", ctxIdx, genIdx);
    LOG_INFO("Added Context profile at index %d, Generation profile at index %d", ctxIdx, genIdx);

    return true;
}

bool LLMBuilder::setupCommonProfiles(nvinfer1::IOptimizationProfile& contextProfile,
    nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network)
{
    bool result = true;

    // Context lengths
    result &= setOptimizationProfile(&contextProfile, network, binding_names::kContextLengths, createDims({1}),
        createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));
    result &= setOptimizationProfile(&generationProfile, network, binding_names::kContextLengths, createDims({1}),
        createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));

    // Rope rotary cos sin
    result &= setOptimizationProfile(&contextProfile, network, binding_names::kRopeCosSin,
        createDims({1, mBuilderConfig.maxKVCacheCapacity, mRotaryDim}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxKVCacheCapacity, mRotaryDim}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxKVCacheCapacity, mRotaryDim}));
    result &= setOptimizationProfile(&generationProfile, network, binding_names::kRopeCosSin,
        createDims({1, mBuilderConfig.maxKVCacheCapacity, mRotaryDim}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxKVCacheCapacity, mRotaryDim}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxKVCacheCapacity, mRotaryDim}));

    // For KVCacheStartIndex, we use zero shape to indicate the kvcache is empty for all sequences in the batch.
    // This can help distinguish the normal prefill and chunked prefill execution.
    result &= setOptimizationProfile(&contextProfile, network, binding_names::kKVCacheStartIndex, createDims({0}),
        createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));
    result &= setOptimizationProfile(&generationProfile, network, binding_names::kKVCacheStartIndex, createDims({1}),
        createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));

    // KV cache profiles
    LOG_DEBUG("Setting up KV cache profiles for %d layers...", mNbKVCacheInputs);
    result &= setupKVCacheProfiles(contextProfile, generationProfile, network);
    LOG_DEBUG("KV cache profiles done. Setting up SSM state profiles for %d Mamba layers...", mNumMambaLayers);

    // SSM state profiles for Mamba layers
    result &= setupSSMStateProfiles(contextProfile, generationProfile, network);

    LOG_DEBUG("SSM state profiles done. Setting up Conv state profiles...");
    // Conv state profiles for Mamba causal conv1d layers
    result &= setupConvStateProfiles(contextProfile, generationProfile, network);
    LOG_DEBUG("Conv state profiles done.");

    return result;
}

bool LLMBuilder::setupVanillaProfiles(nvinfer1::IOptimizationProfile& contextProfile,
    nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network)
{
    bool result = true;

    // Input embeddings - match fixed hidden dimension for strongly typed networks
    int32_t embedsIdx = -1;
    for (int32_t i = 0; i < network.getNbInputs(); ++i) {
        if (std::string(network.getInput(i)->getName()) == binding_names::kInputsEmbeds) {
            embedsIdx = i;
            break;
        }
    }
    
    int32_t hiddenDim = mHiddenSize;
    if (embedsIdx != -1) {
        auto dims = network.getInput(embedsIdx)->getDimensions();
        if (dims.d[2] >= 0) hiddenDim = dims.d[2];
    }

    nvinfer1::Dims minEmbeds = createDims({1, 1, hiddenDim});
    nvinfer1::Dims optEmbeds = createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen / 2, hiddenDim});
    nvinfer1::Dims maxEmbeds = createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen, hiddenDim});

    result &= setOptimizationProfile(&contextProfile, network, binding_names::kInputsEmbeds, minEmbeds, optEmbeds, maxEmbeds);
    result &= setOptimizationProfile(&generationProfile, network, binding_names::kInputsEmbeds, minEmbeds, optEmbeds, maxEmbeds);

    // Last token IDs
    result &= setOptimizationProfile(&contextProfile, network, binding_names::kLastTokenIds, createDims({1, 1}),
        createDims({mBuilderConfig.maxBatchSize, 1}), createDims({mBuilderConfig.maxBatchSize, 1}));
    result &= setOptimizationProfile(&generationProfile, network, binding_names::kLastTokenIds, createDims({1, 1}),
        createDims({mBuilderConfig.maxBatchSize, 1}), createDims({mBuilderConfig.maxBatchSize, 1}));

    return result;
}

bool LLMBuilder::setupEagleProfiles(nvinfer1::IOptimizationProfile& contextProfile,
    nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network)
{
    bool result = true;

    int const maxTokens
        = mBuilderConfig.eagleDraft ? mBuilderConfig.maxDraftTreeSize : mBuilderConfig.maxVerifyTreeSize;

    // Input embeddings - match fixed hidden dimension for strongly typed networks
    int32_t embedsIdx = -1;
    for (int32_t i = 0; i < network.getNbInputs(); ++i) {
        if (std::string(network.getInput(i)->getName()) == binding_names::kInputsEmbeds) {
            embedsIdx = i;
            break;
        }
    }
    
    int32_t hiddenDim = mHiddenSize;
    if (embedsIdx != -1) {
        auto dims = network.getInput(embedsIdx)->getDimensions();
        if (dims.d[2] >= 0) hiddenDim = dims.d[2];
    }

    result &= setOptimizationProfile(&contextProfile, network, binding_names::kInputsEmbeds, createDims({1, 1, hiddenDim}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen / 2, hiddenDim}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen, hiddenDim}));
    result &= setOptimizationProfile(&generationProfile, network, binding_names::kInputsEmbeds, createDims({1, 1, hiddenDim}),
        createDims({mBuilderConfig.maxBatchSize, maxTokens / 2, hiddenDim}),
        createDims({mBuilderConfig.maxBatchSize, maxTokens, hiddenDim}));

    // Last token IDs - 2D shape [batch_size, num_selected_tokens]
    result &= setOptimizationProfile(&contextProfile, network, binding_names::kLastTokenIds, createDims({1, 1}),
        createDims({mBuilderConfig.maxBatchSize, 1}), createDims({mBuilderConfig.maxBatchSize, 1}));
    result &= setOptimizationProfile(&generationProfile, network, binding_names::kLastTokenIds, createDims({1, 1}),
        createDims({mBuilderConfig.maxBatchSize, maxTokens / 2}), createDims({mBuilderConfig.maxBatchSize, maxTokens}));

    if (mBuilderConfig.eagleDraft)
    {
        // Hidden states from draft
        result &= setOptimizationProfile(&contextProfile, network, binding_names::kDraftModelHiddenStates,
            createDims({1, 1, mHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen / 2, mHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen, mHiddenSize}));
        result &= setOptimizationProfile(&generationProfile, network, binding_names::kDraftModelHiddenStates,
            createDims({1, 1, mHiddenSize}), createDims({mBuilderConfig.maxBatchSize, maxTokens / 2, mHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens, mHiddenSize}));

        // Hidden states input
        result &= setOptimizationProfile(&contextProfile, network, binding_names::kBaseModelHiddenStates,
            createDims({1, 1, mTargetModelOutputHiddenDim}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen / 2, mTargetModelOutputHiddenDim}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen, mTargetModelOutputHiddenDim}));
        result &= setOptimizationProfile(&generationProfile, network, binding_names::kBaseModelHiddenStates,
            createDims({1, 1, mTargetModelOutputHiddenDim}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens / 2, mTargetModelOutputHiddenDim}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens, mTargetModelOutputHiddenDim}));
    }

    // Attention mask and position ID
    if (mBuilderConfig.eagleDraft || mBuilderConfig.eagleBase)
    {
        int32_t const attnMaskAlignSize = 32;
        result &= setOptimizationProfile(&contextProfile, network, binding_names::kAttentionMask, createDims({1, 1, 1}),
            createDims({mBuilderConfig.maxBatchSize, 1, 1}), createDims({mBuilderConfig.maxBatchSize, 1, 1}));
            result &= setOptimizationProfile(&generationProfile, network, binding_names::kAttentionMask, createDims({1, 1, 1}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens / 2,
                static_cast<int64_t>(trt_edgellm::divUp(maxTokens / 2, attnMaskAlignSize) * attnMaskAlignSize)}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens,
                static_cast<int64_t>(trt_edgellm::divUp(maxTokens, attnMaskAlignSize) * attnMaskAlignSize)}));

        result &= setOptimizationProfile(&contextProfile, network, binding_names::kAttentionPosId, createDims({1, 1}),
            createDims({mBuilderConfig.maxBatchSize, 1}), createDims({mBuilderConfig.maxBatchSize, 1}));
        result &= setOptimizationProfile(&generationProfile, network, binding_names::kAttentionPosId, createDims({1, 1}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens / 2}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens}));
    }

    return result;
}

bool LLMBuilder::setupDeepstackProfiles(nvinfer1::IOptimizationProfile& contextProfile,
    nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network)
{
    bool result = true;

    // Dynamically detect all deepstack_embeds inputs in the network
    std::vector<std::string> deepstackInputs;
    for (int32_t idx = 0; idx < network.getNbInputs(); idx++)
    {
        std::string const inputName = network.getInput(idx)->getName();
        if (inputName.find(binding_names::kDeepstackEmbedsTemplate) != std::string::npos)
        {
            deepstackInputs.push_back(inputName);
        }
    }

    // If no deepstack embeds found, return early (not a Qwen3VL model)
    if (deepstackInputs.empty())
    {
        return true;
    }

    LOG_INFO("Detected %zu deepstack embedding inputs", deepstackInputs.size());

    // Setup profiles for all detected deepstack_embeds inputs
    // These have the same shape as inputs_embeds: [batch_size, seq_len, hidden_size]
    for (auto const& deepstackInputName : deepstackInputs)
    {
        LOG_INFO("Setting up optimization profile for %s", deepstackInputName.c_str());

        // Same profile as inputs_embeds
        result &= setOptimizationProfile(&contextProfile, network, deepstackInputName.c_str(), createDims({1, 1, mHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen / 2, mHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen, mHiddenSize}));

        if (mBuilderConfig.eagleBase || mBuilderConfig.eagleDraft)
        {
            int const maxTokens
                = mBuilderConfig.eagleDraft ? mBuilderConfig.maxDraftTreeSize : mBuilderConfig.maxVerifyTreeSize;
            result &= setOptimizationProfile(&generationProfile, network, deepstackInputName.c_str(),
                createDims({1, 1, mHiddenSize}), createDims({mBuilderConfig.maxBatchSize, maxTokens / 2, mHiddenSize}),
                createDims({mBuilderConfig.maxBatchSize, maxTokens, mHiddenSize}));
        }
        else
        {
            result &= setOptimizationProfile(&generationProfile, network, deepstackInputName.c_str(),
                createDims({1, 1, mHiddenSize}), createDims({mBuilderConfig.maxBatchSize, 1, mHiddenSize}),
                createDims({mBuilderConfig.maxBatchSize, 1, mHiddenSize}));
        }
    }

    if (!result)
    {
        LOG_ERROR("Failed to setup optimization profiles at setupDeepstackProfiles().");
    }

    return result;
}

bool LLMBuilder::setupLmHeadWeightProfiles(nvinfer1::IOptimizationProfile& contextProfile,
    nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network)
{
    bool result = true;

    // Detect if lm_head_weight input exists (CodePredictor model)
    bool hasLmHeadWeight = false;
    for (int32_t idx = 0; idx < network.getNbInputs(); idx++)
    {
        std::string const inputName = network.getInput(idx)->getName();
        if (inputName == binding_names::kLmHeadWeight)
        {
            hasLmHeadWeight = true;
            break;
        }
    }

    // If no lm_head_weight input found, return early (not a CodePredictor model)
    if (!hasLmHeadWeight)
    {
        return true;
    }

    LOG_INFO("Detected lm_head_weight input (CodePredictor model)");

    // lm_head_weight shape: [vocab_size, hidden_size]
    // For CodePredictor: vocab_size=2048 (codebook size), hidden_size=1024
    // This is a fixed-size weight tensor that gets bound at runtime
    int64_t const vocabSize = mModelConfig["vocab_size"].get<int64_t>();
    int64_t const hiddenSize = mHiddenSize;

    // Both context and generation profiles use the same shape since this is a weight tensor
    result &= setOptimizationProfile(&contextProfile, network, binding_names::kLmHeadWeight, createDims({vocabSize, hiddenSize}),
        createDims({vocabSize, hiddenSize}), createDims({vocabSize, hiddenSize}));
    result &= setOptimizationProfile(&generationProfile, network, binding_names::kLmHeadWeight,
        createDims({vocabSize, hiddenSize}), createDims({vocabSize, hiddenSize}), createDims({vocabSize, hiddenSize}));

    if (!result)
    {
        LOG_ERROR("Failed to setup optimization profiles for lm_head_weight.");
    }

    return result;
}

bool LLMBuilder::setupLoraProfiles(nvinfer1::IOptimizationProfile& contextProfile,
    nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network)
{
    bool result = true;
    if (mBuilderConfig.maxLoraRank == 0)
    {
        LOG_WARNING(
            "Your model has dynamic LoRA, but max LoRA rank is 0. This is equivalent to no LoRA. Please set "
            "--maxLoraRank to a positive value if you want to use LoRA.");
        return true;
    }

    bool findLoraWeights = false;

    for (int i = 0; i < network.getNbInputs(); ++i)
    {
        auto* input = network.getInput(i);
        std::string inputName = input->getName();

        if (inputName.find(binding_names::kLoraAPrefix) != std::string::npos)
        {
            if (!findLoraWeights)
            {
                findLoraWeights = true;
            }
            // For lora_A, the shape is [gemm_k, lora_rank]
            auto dims = input->getDimensions();
            if (dims.nbDims == 2)
            {
                int64_t gemm_k = dims.d[0];
                result
                    &= setOptimizationProfile(&contextProfile, network, inputName.c_str(), createDims({gemm_k, 0}), // min shape
                        createDims({gemm_k, mBuilderConfig.maxLoraRank / 2}),                              // opt shape
                        createDims({gemm_k, mBuilderConfig.maxLoraRank}));                                 // max shape
                result &= setOptimizationProfile(&generationProfile, network, inputName.c_str(),
                    createDims({gemm_k, 0}),                              // min shape
                    createDims({gemm_k, mBuilderConfig.maxLoraRank / 2}), // opt shape
                    createDims({gemm_k, mBuilderConfig.maxLoraRank}));    // max shape
            }
        }
        else if (inputName.find(binding_names::kLoraBPrefix) != std::string::npos)
        {
            if (!findLoraWeights)
            {
                findLoraWeights = true;
            }
            // For lora_B, the shape is [lora_rank, gemm_n]
            auto dims = input->getDimensions();
            if (dims.nbDims == 2)
            {
                int64_t gemm_n = dims.d[1];
                result
                    &= setOptimizationProfile(&contextProfile, network, inputName.c_str(), createDims({0, gemm_n}), // min shape
                        createDims({mBuilderConfig.maxLoraRank / 2, gemm_n}),                              // opt shape
                        createDims({mBuilderConfig.maxLoraRank, gemm_n}));                                 // max shape
                result &= setOptimizationProfile(&generationProfile, network, inputName.c_str(),
                    createDims({0, gemm_n}),                              // min shape
                    createDims({mBuilderConfig.maxLoraRank / 2, gemm_n}), // opt shape
                    createDims({mBuilderConfig.maxLoraRank, gemm_n}));    // max shape
            }
        }
    }

    if (!findLoraWeights)
    {
        LOG_ERROR(
            "Failed to find any LoRA weights inputs in the ONNX model. Have you inserted LoRA weights using "
            "tensorrt-edgellm-insert-lora command?");
        return false;
    }

    if (!result)
    {
        LOG_ERROR("Failed to setup optimization profiles at setupLoraProfiles().");
    }

    return result;
}

bool LLMBuilder::setupKVCacheProfiles(nvinfer1::IOptimizationProfile& contextProfile,
    nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network)
{
    bool result = true;
    for (int i = 0; i < mNbKVCacheInputs; ++i)
    {
        std::string kName = binding_names::formatKCacheName(i, true);
        std::string vName = binding_names::formatVCacheName(i, true);

        // Standard alternate: check for dot separator
        std::string kNameDot = "past_key_values." + std::to_string(i);
        std::string vNameDot = "present_key_values." + std::to_string(i);

        // Find input in network to get its actual dimensions
        int32_t kInputIdx = -1;
        int32_t vInputIdx = -1;
        bool isCombinedKV = false;
        std::string kvNameCombined = "kv_cache_" + std::to_string(i);
        
        for (int32_t j = 0; j < network.getNbInputs(); ++j)
        {
            std::string const inputName = network.getInput(j)->getName();
            if (inputName == kName || inputName == kNameDot)
            {
                kInputIdx = j;
                kName = inputName; // Use the found name
            }
            if (inputName == vName || inputName == vNameDot)
            {
                vInputIdx = j;
                vName = inputName; // Use the found name
            }
            if (inputName == kvNameCombined)
            {
                kInputIdx = j;
                isCombinedKV = true;
                kName = inputName; // Use combined name for the profile
            }
        }

        if (mBuilderConfig.useTrtNativeOps)
        {
            if (isCombinedKV) {
                // Determine if it's Standard 5D [B, 2, H, S, D] or Recurrent 4D [B, H, D, D]
                auto dims = network.getInput(kInputIdx)->getDimensions();
                if (dims.nbDims == 4) {
                    // Recurrent state for Gated Delta Rule
                    int32_t heads = dims.d[1];
                    int32_t headSize = dims.d[2];
                    nvinfer1::Dims kvShape = createDims({1, heads, headSize, headSize});
                    nvinfer1::Dims maxKvShape = createDims({mBuilderConfig.maxBatchSize, heads, headSize, headSize});
                    
                    printf("[Builder] Setting 4D profile for %s: heads=%d, headSize=%d\n", kName.c_str(), heads, headSize);
                    
                    result &= setOptimizationProfile(&contextProfile, network, kName.c_str(),
                        kvShape, maxKvShape, maxKvShape);
                    result &= setOptimizationProfile(&generationProfile, network, kName.c_str(),
                        kvShape, maxKvShape, maxKvShape);
                } else {
                    // Standard TRT attention with combined KV cache (exported via tuple)
                    // Shape: [batch, 2, num_kv_heads, seq_len, head_dim]
                    int32_t heads = dims.d[2];
                    int32_t headSize = dims.d[4];
                    int32_t seqLen = dims.d[3];
                    
                    // Strongly typed networks require profile bounds to match fixed dimensions exactly.
                    // If seqLen is not dynamic (-1), we must use its actual value for min/opt/max.
                    int32_t minSeq = (seqLen < 0) ? 1 : seqLen;
                    int32_t optSeq = (seqLen < 0) ? mBuilderConfig.maxInputLen / 2 : seqLen;
                    int32_t maxSeq = (seqLen < 0) ? mBuilderConfig.maxKVCacheCapacity : seqLen;
                    
                    if (seqLen >= 0) {
                        minSeq = optSeq = maxSeq = seqLen;
                    }

                    nvinfer1::Dims minKVCacheShape = createDims({1, 2, heads, minSeq, headSize});
                    nvinfer1::Dims optKVCacheShape = createDims({mBuilderConfig.maxBatchSize, 2, heads, optSeq, headSize});
                    nvinfer1::Dims maxKVCacheShape = createDims({mBuilderConfig.maxBatchSize, 2, heads, maxSeq, headSize});

                    printf("[Builder] Setting 5D profile for %s: heads=%d, headSize=%d, seq=[%d,%d,%d]\n", 
                           kName.c_str(), heads, headSize, minSeq, optSeq, maxSeq);

                    result &= setOptimizationProfile(&contextProfile, network, kName.c_str(),
                        minKVCacheShape, optKVCacheShape, maxKVCacheShape);
                    
                    result &= setOptimizationProfile(&generationProfile, network, kName.c_str(),
                        minKVCacheShape, optKVCacheShape, maxKVCacheShape);
                }
            } else {
                // TRT attention: separate K and V caches
                // Shape: [batch, num_kv_heads, seq_len, head_dim]
                int32_t kHeads = mNumKVHeads;
                int32_t kHeadSize = mHeadSize;
                int32_t kSeqLen = mBuilderConfig.maxKVCacheCapacity;
                if (kInputIdx != -1)
                {
                    auto dims = network.getInput(kInputIdx)->getDimensions();
                    kHeads = dims.d[1];
                    kHeadSize = dims.d[3];
                    kSeqLen = dims.d[2];
                }

                int32_t vHeads = mNumKVHeads;
                int32_t vHeadSize = mHeadSize;
                int32_t vSeqLen = mBuilderConfig.maxKVCacheCapacity;
                if (vInputIdx != -1)
                {
                    auto dims = network.getInput(vInputIdx)->getDimensions();
                    vHeads = dims.d[1];
                    vHeadSize = dims.d[3];
                    vSeqLen = dims.d[2];
                }

                // Match fixed dimensions for strongly typed networks
                int32_t minKSeq = (kSeqLen < 0) ? 1 : kSeqLen;
                int32_t optKSeq = (kSeqLen < 0) ? mBuilderConfig.maxKVCacheCapacity / 2 : kSeqLen;
                int32_t maxKSeq = (kSeqLen < 0) ? mBuilderConfig.maxKVCacheCapacity : kSeqLen;

                int32_t minVSeq = (vSeqLen < 0) ? 1 : vSeqLen;
                int32_t optVSeq = (vSeqLen < 0) ? mBuilderConfig.maxKVCacheCapacity / 2 : vSeqLen;
                int32_t maxVSeq = (vSeqLen < 0) ? mBuilderConfig.maxKVCacheCapacity : vSeqLen;

                if (kSeqLen >= 0) {
                    minKSeq = optKSeq = maxKSeq = kSeqLen;
                }
                if (vSeqLen >= 0) {
                    minVSeq = optVSeq = maxVSeq = vSeqLen;
                }

                nvinfer1::Dims minKCacheShape = createDims({1, kHeads, minKSeq, kHeadSize});
                nvinfer1::Dims optKCacheShape = createDims({mBuilderConfig.maxBatchSize, kHeads, optKSeq, kHeadSize});
                nvinfer1::Dims maxKCacheShape = createDims({mBuilderConfig.maxBatchSize, kHeads, maxKSeq, kHeadSize});

                printf("[Builder] Setting separate K profile for %s: heads=%d, headSize=%d, seq=[%d,%d,%d]\n", 
                       kName.c_str(), kHeads, kHeadSize, minKSeq, optKSeq, maxKSeq);

                result &= setOptimizationProfile(&contextProfile, network, kName.c_str(),
                    minKCacheShape, optKCacheShape, maxKCacheShape);
                result &= setOptimizationProfile(&generationProfile, network, kName.c_str(),
                    minKCacheShape, optKCacheShape, maxKCacheShape);

                nvinfer1::Dims minVCacheShape = createDims({1, vHeads, minVSeq, vHeadSize});
                nvinfer1::Dims optVCacheShape = createDims({mBuilderConfig.maxBatchSize, vHeads, optVSeq, vHeadSize});
                nvinfer1::Dims maxVCacheShape = createDims({mBuilderConfig.maxBatchSize, vHeads, maxVSeq, vHeadSize});

                printf("[Builder] Setting separate V profile for %s: heads=%d, headSize=%d, seq=[%d,%d,%d]\n", 
                       vName.c_str(), vHeads, vHeadSize, minVSeq, optVSeq, maxVSeq);

                result &= setOptimizationProfile(&contextProfile, network, vName.c_str(),
                    minVCacheShape, optVCacheShape, maxVCacheShape);
                result &= setOptimizationProfile(&generationProfile, network, vName.c_str(),
                    minVCacheShape, optVCacheShape, maxVCacheShape);
            }
        }

        else
        {
            // Plugin path: combined KV cache with "2" dimension
            // KV cache shape is [B, 2, num_kv_heads, 0 to max_kv_cache_capacity, head_dim]
            
            // Use the consistent kv_cache_ naming convention from ONNX export
            std::string kvName = "kv_cache_" + std::to_string(i);

            bool inputExists = false;
            int32_t inputIdx = -1;
            for (int32_t j = 0; j < network.getNbInputs(); ++j)
            {
                if (std::string(network.getInput(j)->getName()) == kvName)
                {
                    inputExists = true;
                    inputIdx = j;
                    break;
                }
            }

            if (!inputExists)
            {
                LOG_DEBUG("Skipping KV cache profile for %s (not found in network)", kvName.c_str());
                continue;
            }

            // Extract actual heads and headSize from the network input to be safe
            auto dims = network.getInput(inputIdx)->getDimensions();

            // Determine if it's Recurrent 4D [B, H, D, D] or Standard 5D [B, 2, H, S, D]
            if (dims.nbDims == 4) {
                int32_t heads = dims.d[1];
                int32_t headSize = dims.d[2];
                nvinfer1::Dims kvShape = createDims({1, heads, headSize, headSize});
                nvinfer1::Dims maxKvShape = createDims({mBuilderConfig.maxBatchSize, heads, headSize, headSize});

                printf("[Builder] Setting 4D plugin profile for %s: heads=%d, headSize=%d\n", kvName.c_str(), heads, headSize);

                result &= setOptimizationProfile(&contextProfile, network, kvName.c_str(),
                    kvShape, maxKvShape, maxKvShape);
                result &= setOptimizationProfile(&generationProfile, network, kvName.c_str(),
                    kvShape, maxKvShape, maxKvShape);
            } else {
                int32_t heads = dims.d[2];
                int32_t headSize = dims.d[4];
                int32_t seqLen = dims.d[3];

                int32_t minSeq = (seqLen < 0) ? 1 : seqLen;
                int32_t optSeq = (seqLen < 0) ? mBuilderConfig.maxInputLen / 2 : seqLen;
                int32_t maxSeq = (seqLen < 0) ? mBuilderConfig.maxKVCacheCapacity : seqLen;

                if (seqLen >= 0) minSeq = optSeq = maxSeq = seqLen;

                nvinfer1::Dims minKVCacheShape = createDims({1, 2, heads, minSeq, headSize});
                nvinfer1::Dims optKVCacheShape = createDims({mBuilderConfig.maxBatchSize, 2, heads, optSeq, headSize});
                nvinfer1::Dims maxKVCacheShape = createDims({mBuilderConfig.maxBatchSize, 2, heads, maxSeq, headSize});

                printf("[Builder] Setting 5D plugin profile for %s: heads=%d, headSize=%d, seq=[%d,%d,%d]\n", 
                       kvName.c_str(), heads, headSize, minSeq, optSeq, maxSeq);

                result &= setOptimizationProfile(&contextProfile, network, kvName.c_str(),
                    minKVCacheShape, optKVCacheShape, maxKVCacheShape);
                result &= setOptimizationProfile(&generationProfile, network, kvName.c_str(),
                    minKVCacheShape, optKVCacheShape, maxKVCacheShape);
            }

        }
    }

    return result;
}

bool LLMBuilder::setupSSMStateProfiles(nvinfer1::IOptimizationProfile& contextProfile,
    nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network)
{
    if (mNumMambaLayers == 0)
    {
        return true;
    }

    bool result = true;

    // SSM state shape: [batch, mamba_num_heads, mamba_head_dim, ssm_state_size]
    nvinfer1::Dims minSSMShape = createDims({1, mMambaNumHeads, mMambaHeadDim, mSSMStateSize});
    nvinfer1::Dims optSSMShape
        = createDims({mBuilderConfig.maxBatchSize, mMambaNumHeads, mMambaHeadDim, mSSMStateSize});
    nvinfer1::Dims maxSSMShape
        = createDims({mBuilderConfig.maxBatchSize, mMambaNumHeads, mMambaHeadDim, mSSMStateSize});

    for (int32_t i = 0; i < mNumMambaLayers; ++i)
    {
        std::string const ssmStateName = binding_names::formatSSMStateName(i, /*isPast=*/true);
        result &= setOptimizationProfile(&contextProfile, network, ssmStateName.c_str(), minSSMShape, optSSMShape, maxSSMShape);
        result
            &= setOptimizationProfile(&generationProfile, network, ssmStateName.c_str(), minSSMShape, optSSMShape, maxSSMShape);
    }

    LOG_DEBUG("Set up SSM state optimization profiles for %d Mamba layers", mNumMambaLayers);
    return result;
}

bool LLMBuilder::setupConvStateProfiles(nvinfer1::IOptimizationProfile& contextProfile,
    nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network)
{
    if (mNumMambaLayers == 0 || mConvDim == 0 || mConvKernel == 0)
    {
        return true;
    }

    bool result = true;

    // Conv state shape: [batch, conv_dim, conv_kernel]
    nvinfer1::Dims minConvShape = createDims({1, mConvDim, mConvKernel});
    nvinfer1::Dims optConvShape = createDims({mBuilderConfig.maxBatchSize, mConvDim, mConvKernel});
    nvinfer1::Dims maxConvShape = createDims({mBuilderConfig.maxBatchSize, mConvDim, mConvKernel});

    for (int32_t i = 0; i < mNumMambaLayers; ++i)
    {
        std::string const convStateName = binding_names::formatConvStateName(i, /*isPast=*/true);
        result
            &= setOptimizationProfile(&contextProfile, network, convStateName.c_str(), minConvShape, optConvShape, maxConvShape);
        result &= setOptimizationProfile(
            &generationProfile, network, convStateName.c_str(), minConvShape, optConvShape, maxConvShape);
    }

    LOG_DEBUG("Set up conv state optimization profiles for %d Mamba layers", mNumMambaLayers);
    return result;
}

bool LLMBuilder::copyConfig()
{
    // Determine config file name based on model type
    std::string configFileName;
    if (mBuilderConfig.eagleDraft)
    {
        configFileName = "draft_config.json";
    }
    else if (mBuilderConfig.eagleBase)
    {
        configFileName = "base_config.json";
    }
    else
    {
        configFileName = "config.json";
    }

    std::string const targetConfigPath = (mEngineDir / configFileName).string();

    // Create a copy of mModelConfig and add builder config
    Json configWithBuilder = mModelConfig;
    configWithBuilder["builder_config"] = mBuilderConfig.toJson();
    
    // Ensure maxSupportedInputLength is consistent for runtime
    configWithBuilder["max_supported_input_length"] = static_cast<int32_t>(mBuilderConfig.maxKVCacheCapacity);

    // Add detected num_deepstack_features if present (Qwen3VL models)
    configWithBuilder["num_deepstack_features"] = mNumDeepstackFeatures;

    // Write updated config
    std::ofstream targetConfigFile(targetConfigPath);
    if (!targetConfigFile.is_open())
    {
        LOG_ERROR("Failed to open target config file: %s", targetConfigPath.c_str());
        return false;
    }
    
    printf("[Builder] copyConfig: Exporting head_dim=%d, numKVHeads=%d, maxInputLen=%d to %s\n",
           configWithBuilder["head_dim"].get<int32_t>(),
           configWithBuilder["num_key_value_heads"].get<int32_t>(),
           configWithBuilder["max_supported_input_length"].get<int32_t>(),
           targetConfigPath.c_str());

    targetConfigFile << configWithBuilder.dump(2);
    targetConfigFile.flush();
    targetConfigFile.close();

    LOG_INFO("Copied config.json with builder config to %s", targetConfigPath.c_str());
    return true;
}

bool LLMBuilder::copyTokenizerFiles()
{
    // Eagle3 draft model does not need tokenizer files
    if (mBuilderConfig.eagleDraft)
    {
        return true;
    }

    // Models that use embeddings as input (e.g., Talker, CodePredictor) don't need tokenizer
    bool useEmbeddingsInput = mModelConfig.value("use_embeddings_input", false);
    if (useEmbeddingsInput)
    {
        LOG_INFO("Skipping tokenizer files (model uses embeddings input)");
        return true;
    }

    std::vector<std::string> tokenizerFiles
        = {"tokenizer_config.json", "tokenizer.json", "processed_chat_template.json"};
    bool allSuccess = true;

    for (auto const& filename : tokenizerFiles)
    {
        std::string const srcPath = (mOnnxDir / filename).string();
        std::string const dstPath = (mEngineDir / filename).string();

        if (file_io::copyFile(srcPath, dstPath))
        {
            LOG_INFO("Copied tokenizer file: %s", filename.c_str());
        }
        else
        {
            LOG_WARNING("Failed to copy tokenizer file %s", filename.c_str());
            allSuccess = false;
        }
    }

    return allSuccess;
}

bool LLMBuilder::copyEagleFiles()
{
    // Copy d2t.safetensors for Eagle3 draft models
    if (mBuilderConfig.eagleDraft)
    {
        std::string const d2tPath = (mOnnxDir / "d2t.safetensors").string();
        std::string const targetD2tPath = (mEngineDir / "d2t.safetensors").string();

        if (file_io::copyFile(d2tPath, targetD2tPath))
        {
            LOG_INFO("Copied d2t.safetensors to %s", targetD2tPath.c_str());
        }
        else
        {
            LOG_WARNING("Failed to copy d2t.safetensors to %s", targetD2tPath.c_str());
            return false;
        }
    }

    return true;
}

bool LLMBuilder::copyVocabMappingFiles()
{
    // Copy vocab_map.safetensors if reduced vocabulary is used
    if (mModelConfig.contains(binding_names::kReducedVocabSizeKey)
        && mModelConfig[binding_names::kReducedVocabSizeKey].get<int32_t>() > 0)
    {
        std::string const vocabMapPath = (mOnnxDir / binding_names::kVocabMapFileName).string();
        std::string const targetVocabMapPath = (mEngineDir / binding_names::kVocabMapFileName).string();

        if (file_io::copyFile(vocabMapPath, targetVocabMapPath))
        {
            LOG_INFO("Copied %s to %s", binding_names::kVocabMapFileName, targetVocabMapPath.c_str());
        }
        else
        {
            LOG_WARNING("%s not found in %s. This is expected if reduced vocabulary is not used.",
                binding_names::kVocabMapFileName, mOnnxDir.string().c_str());
        }
    }

    return true;
}

bool LLMBuilder::copyEmbeddingFile()
{
    // Eagle draft model uses shared embedding table from base model, so skip copying
    if (mBuilderConfig.eagleDraft)
    {
        return true;
    }

    // Check if this is a Talker model (has text_projection.safetensors)
    std::filesystem::path const textProjectionPath = mOnnxDir / "text_projection.safetensors";
    if (std::filesystem::exists(textProjectionPath))
    {
        // Talker: copy embedding + text_projection + hidden_projection (optional, text-only TTS omits it)
        LOG_INFO("Detected Talker model, copying projection files...");

        std::vector<std::string> requiredFiles
            = {"embedding.safetensors", "text_projection.safetensors", "text_embedding.safetensors"};
        std::vector<std::string> optionalFiles = {"hidden_projection.safetensors"};

        bool allSuccess = true;
        for (auto const& filename : requiredFiles)
        {
            std::string const srcPath = (mOnnxDir / filename).string();
            std::string const dstPath = (mEngineDir / filename).string();

            if (file_io::copyFile(srcPath, dstPath))
            {
                LOG_INFO("Copied %s", filename.c_str());
            }
            else
            {
                LOG_ERROR("Failed to copy %s", filename.c_str());
                allSuccess = false;
            }
        }
        for (auto const& filename : optionalFiles)
        {
            std::string const srcPath = (mOnnxDir / filename).string();
            std::string const dstPath = (mEngineDir / filename).string();

            if (file_io::copyFile(srcPath, dstPath))
            {
                LOG_INFO("Copied %s", filename.c_str());
            }
            else
            {
                LOG_INFO("Optional %s not found, skipping", filename.c_str());
            }
        }

        return allSuccess;
    }

    // Check if this is a CodePredictor model (has codec_embeddings.safetensors)
    std::filesystem::path const codecEmbedPath = mOnnxDir / "codec_embeddings.safetensors";
    if (std::filesystem::exists(codecEmbedPath))
    {
        LOG_INFO("Detected CodePredictor model, copying codec files...");
        std::vector<std::string> cpFiles
            = {"codec_embeddings.safetensors", "lm_heads.safetensors", "small_to_mtp_projection.safetensors"};
        bool allSuccess = true;
        for (auto const& filename : cpFiles)
        {
            std::string const srcPath = (mOnnxDir / filename).string();
            std::string const dstPath = (mEngineDir / filename).string();
            if (file_io::copyFile(srcPath, dstPath))
            {
                LOG_INFO("Copied %s", filename.c_str());
            }
            else
            {
                LOG_ERROR("Failed to copy required CodePredictor file: %s", filename.c_str());
                allSuccess = false;
            }
        }
        return allSuccess;
    }

    // Copy embedding.safetensors for vanilla LLM models
    std::string const embeddingPath = (mOnnxDir / "embedding.safetensors").string();
    std::string const targetEmbeddingPath = (mEngineDir / "embedding.safetensors").string();

    if (file_io::copyFile(embeddingPath, targetEmbeddingPath))
    {
        LOG_INFO("Copied embedding.safetensors to %s", targetEmbeddingPath.c_str());
    }
    else
    {
        LOG_ERROR(
            "Failed to copy embedding.safetensors from %s to %s", embeddingPath.c_str(), targetEmbeddingPath.c_str());
        return false;
    }

    return true;
}

} // namespace builder
} // namespace trt_edgellm
