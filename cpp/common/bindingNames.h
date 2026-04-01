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

#pragma once

#include <string>
#include <cstring>
#include <NvInferRuntime.h>

namespace trt_edgellm
{
/*!
 * @namespace binding_names
 * @brief Unified tensor binding names for TensorRT engines
 */
namespace binding_names
{

/*! @name Core LLM Input/Output Bindings
 * @{
 */
inline constexpr char const* kInputsEmbeds = "inputs_embeds";
inline constexpr char const* kContextLengths = "context_lengths";
inline constexpr char const* kLastTokenIds = "last_token_ids";
inline constexpr char const* kLogits = "logits";
inline constexpr char const* kOutputHiddenStates = "hidden_states";
/*! @} */

/*! @name Positional Encoding Bindings
 * @{
 */
inline constexpr char const* kRopeCosSin = "rope_rotary_cos_sin";
/*! @} */

/*! @name KV Cache Bindings
 * @{
 */
inline constexpr char const* kKVCacheStartIndex = "kvcache_start_index";
inline constexpr char const* kPastKeyValuesTemplate = "past_key_values";
inline constexpr char const* kPresentKeyValuesTemplate = "present_key_values";
inline constexpr char const* kKCacheTemplate = "k_cache";
inline constexpr char const* kVCacheTemplate = "v_cache";
inline constexpr char const* kPresentKCacheTemplate = "present_k_cache";
inline constexpr char const* kPresentVCacheTemplate = "present_v_cache";
/*! @} */

/*! @name SSM (Mamba) State Bindings
 * @{
 */
inline constexpr char const* kSSMStateTemplate = "ssm_state";
inline constexpr char const* kPresentSSMStateTemplate = "present_ssm_state";
inline constexpr char const* kConvStateTemplate = "conv_state";
inline constexpr char const* kPresentConvStateTemplate = "present_conv_state";
/*! @} */

/*! @name Eagle Speculative Decoding Bindings
 * @{
 */
inline constexpr char const* kBaseModelHiddenStates = "hidden_states_input";
inline constexpr char const* kDraftModelHiddenStates = "hidden_states_from_draft";
inline constexpr char const* kAttentionMask = "attention_mask";
inline constexpr char const* kAttentionPosId = "attention_pos_id";
/*! @} */

/*! @name Visual Encoder Bindings (Qwen-VL, InternVL)
 * @{
 */
inline constexpr char const* kVisualInput = "input";
inline constexpr char const* kVisualOutput = "output";
inline constexpr char const* kRotaryPosEmb = "rotary_pos_emb";
inline constexpr char const* kCuSeqlens = "cu_seqlens";
inline constexpr char const* kMaxSeqLenCarrier = "max_seqlen_carrier";
inline constexpr char const* kCuWindowSeqlens = "cu_window_seqlens";
inline constexpr char const* kWindowIndex = "window_index";
inline constexpr char const* kReverseWindowIndex = "reverse_window_index";
inline constexpr char const* kFastPosEmbIdx = "fast_pos_embed_idx";
inline constexpr char const* kFastPosEmbWeight = "fast_pos_embed_weight";
inline constexpr char const* kDeepstackFeaturesTemplate = "deepstack_features";
inline constexpr char const* kDeepstackEmbedsTemplate = "deepstack_embeds";
/*! @} */

/*! @name Vocabulary Mapping Configuration
 * @{
 */
inline constexpr char const* kReducedVocabSizeKey = "reduced_vocab_size";
inline constexpr char const* kVocabMapFileName = "vocab_map.safetensors";
/*! @} */

/*! @name Audio Encoder Bindings (Qwen3-Omni)
 * @{
 */
inline constexpr char const* kAudioPaddedFeatures = "padded_feature";
inline constexpr char const* kAudioPaddedMaskIndices = "padded_mask_after_cnn_indices";
inline constexpr char const* kAudioAttentionMask = "attention_mask";
inline constexpr char const* kAudioOutput = "last_hidden_state";
/*! @} */

/*! @name CodePredictor Bindings (Qwen3-Omni)
 * @{
 */
inline constexpr char const* kLmHeadWeight = "lm_head_weight";
/*! @} */

/*! @name Code2Wav Vocoder Bindings (Qwen3-Omni)
 * @{
 */
inline constexpr char const* kCode2WavCodes = "codes";
inline constexpr char const* kCode2WavWaveform = "waveform";
/*! @} */

/*! @name LoRA (Low-Rank Adaptation) Bindings
 * @{
 */
inline constexpr char const* kLoraAPrefix = "lora_A";
inline constexpr char const* kLoraBPrefix = "lora_B";
inline constexpr char const* kEdgellmVersion = "edgellm_version";
/*! @} */

/*! @name Utility Functions
 * @{
 */

/**
 * @brief Check if engine has a tensor with the given name without triggering TRT internal error logs.
 */
inline bool hasTensor(nvinfer1::ICudaEngine const* engine, char const* name)
{
    if (!engine || !name || name[0] == '\0') return false;
    for (int i = 0; i < engine->getNbIOTensors(); ++i)
    {
        if (std::strcmp(engine->getIOTensorName(i), name) == 0)
        {
            return true;
        }
    }
    return false;
}

/*!
 * @brief Format KV cache binding name for a specific layer (Static version for building)
 */
inline std::string formatKVCacheName(int32_t layerIdx, bool isPast = true)
{
    return std::string(isPast ? kPastKeyValuesTemplate : kPresentKeyValuesTemplate) + "_" + std::to_string(layerIdx);
}

/*!
 * @brief Format KV cache binding name for a specific layer (Runtime version with auto-detection)
 */
inline std::string formatKVCacheName(nvinfer1::ICudaEngine const* engine, int32_t layerIdx, bool isPast = true)
{
    // 1. past_key_values_N (Standard TRT-LLM)
    std::string name = formatKVCacheName(layerIdx, isPast);
    if (hasTensor(engine, name.c_str())) return name;

    // 1b. past_key_values.N (Alternative dot separator)
    name = std::string(isPast ? kPastKeyValuesTemplate : kPresentKeyValuesTemplate) + "." + std::to_string(layerIdx);
    if (hasTensor(engine, name.c_str())) return name;

    // 2. kv_cache_N (Fragmented style)
    name = (isPast ? "kv_cache_" : "present_kv_cache_") + std::to_string(layerIdx);
    if (hasTensor(engine, name.c_str())) return name;

    return "";
}

/*!
 * @brief Format K cache binding name for a specific layer (Static version for building)
 */
inline std::string formatKCacheName(int32_t layerIdx, bool isPast = true)
{
    return std::string(isPast ? kKCacheTemplate : kPresentKCacheTemplate) + "_" + std::to_string(layerIdx);
}

/*!
 * @brief Format K cache binding name for a specific layer (Runtime version with auto-detection)
 */
inline std::string formatKCacheName(nvinfer1::ICudaEngine const* engine, int32_t layerIdx, bool isPast = true)
{
    // If unified KV exists, K name is the unified name.
    std::string name = formatKVCacheName(engine, layerIdx, isPast);
    if (!name.empty()) return name;

    // 3. k_cache_N (Separate style)
    name = formatKCacheName(layerIdx, isPast);
    if (hasTensor(engine, name.c_str())) return name;

    return "";
}

/*!
 * @brief Format V cache binding name for a specific layer (Static version for building)
 */
inline std::string formatVCacheName(int32_t layerIdx, bool isPast = true)
{
    return std::string(isPast ? kVCacheTemplate : kPresentVCacheTemplate) + "_" + std::to_string(layerIdx);
}

/*!
 * @brief Format V cache binding name for a specific layer (Runtime version with auto-detection)
 */
inline std::string formatVCacheName(nvinfer1::ICudaEngine const* engine, int32_t layerIdx, bool isPast = true)
{
    // If unified KV exists, there is no separate V binding.
    if (!formatKVCacheName(engine, layerIdx, isPast).empty()) return "";

    // 3. v_cache_N (Separate style)
    std::string name = formatVCacheName(layerIdx, isPast);
    if (hasTensor(engine, name.c_str())) return name;

    return "";
}

inline std::string formatSSMStateName(int32_t mambaLayerIdx, bool isPast = true)
{
    return std::string(isPast ? kSSMStateTemplate : kPresentSSMStateTemplate) + "_" + std::to_string(mambaLayerIdx);
}

inline bool isSSMStateBinding(std::string const& bindingName)
{
    return bindingName.find(kSSMStateTemplate) != std::string::npos
        || bindingName.find(kPresentSSMStateTemplate) != std::string::npos;
}

inline std::string formatConvStateName(int32_t mambaLayerIdx, bool isPast = true)
{
    return std::string(isPast ? kConvStateTemplate : kPresentConvStateTemplate) + "_" + std::to_string(mambaLayerIdx);
}

inline bool isConvStateBinding(std::string const& bindingName)
{
    return bindingName.find(kConvStateTemplate) != std::string::npos
        || bindingName.find(kPresentConvStateTemplate) != std::string::npos;
}

inline bool isLoraBinding(std::string const& bindingName) noexcept
{
    return bindingName.find(kLoraAPrefix) != std::string::npos || bindingName.find(kLoraBPrefix) != std::string::npos;
}

inline bool isKVCacheBinding(std::string const& bindingName) noexcept
{
    return bindingName.find(kPastKeyValuesTemplate) != std::string::npos
        || bindingName.find(kPresentKeyValuesTemplate) != std::string::npos
        || bindingName.find(kKCacheTemplate) != std::string::npos
        || bindingName.find(kVCacheTemplate) != std::string::npos
        || bindingName.find(kPresentKCacheTemplate) != std::string::npos
        || bindingName.find(kPresentVCacheTemplate) != std::string::npos;
}

inline std::string formatDeepstackFeaturesName(int32_t layerIdx)
{
    return std::string(kDeepstackFeaturesTemplate) + "_" + std::to_string(layerIdx);
}

inline std::string formatDeepstackEmbedsName(int32_t embedIdx)
{
    return std::string(kDeepstackEmbedsTemplate) + "_" + std::to_string(embedIdx);
}

/*! @} */

} // namespace binding_names
} // namespace trt_edgellm
