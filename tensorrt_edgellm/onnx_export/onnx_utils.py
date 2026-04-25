# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gc
import json
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import torch
import torch.nn as nn
from ..common import ONNX_OPSET_VERSION

@torch.library.custom_op("trt::kv_cache_update_onnx", mutates_args=())
def kv_cache_update_onnx(
    past_key_value: torch.Tensor,
    new_key_value: torch.Tensor,
    kvcache_start_index: torch.Tensor,
) -> torch.Tensor:
    return past_key_value.clone()

@kv_cache_update_onnx.register_fake
def kv_cache_update_onnx_fake(past_key_value, new_key_value, kvcache_start_index):
    return torch.empty_like(past_key_value)

@torch.library.custom_op("trt::rope_onnx", mutates_args=())
def rope_onnx(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return query.clone(), key.clone()

@rope_onnx.register_fake
def rope_onnx_fake(query, key, cos, sin):
    return torch.empty_like(query), torch.empty_like(key)

@torch.library.custom_op("trt::attention_onnx", mutates_args=())
def attention_onnx(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    context_lengths: torch.Tensor,
    rope_rotary_cos_sin: torch.Tensor,
    kvcache_start_index: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = True,
    scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return query.clone(), kv_cache.clone()

@attention_onnx.register_fake
def attention_onnx_fake(query, key, value, kv_cache, context_lengths, rope_rotary_cos_sin, kvcache_start_index, num_q_heads, num_kv_heads, head_size, attn_mask=None, is_causal=True, scale=1.0):
    return torch.empty_like(query), torch.empty_like(kv_cache)

def _is_nvfp4_quantized(model: nn.Module) -> bool:
    for m in model.modules():
        if hasattr(m, "input_quantizer") and hasattr(m.input_quantizer, "block_sizes"):
            if m.input_quantizer.block_sizes is not None and m.input_quantizer.block_sizes.get("scale_bits") == (4, 3):
                return True
    return False

def _is_mxfp8_quantized(model: nn.Module) -> bool:
    for m in model.modules():
        if hasattr(m, "input_quantizer") and hasattr(m.input_quantizer, "block_sizes"):
            if m.input_quantizer.block_sizes is not None and m.input_quantizer.block_sizes.get("scale_bits") == (8, 0):
                return True
    return False

def _is_int4_awq_quantized(model: nn.Module) -> bool:
    for m in model.modules():
        if hasattr(m, "input_quantizer") and hasattr(m.input_quantizer, "num_bits"):
            if m.input_quantizer.num_bits == 4:
                return True
    return False

def export_onnx(model,
                inputs,
                output_dir,
                input_names,
                output_names,
                dynamic_axes,
                custom_opsets=None,
                dynamo=False,
                qwen35_interleaving=False,
                config=None) -> None:

    '''
    Export the model to ONNX format.
    Args:
        model: The model to export
        inputs: The inputs to the model
        output_dir: The directory to save the ONNX model
        input_names: The names of the input tensors
        output_names: The names of the output tensors
        dynamic_axes: The dynamic axes of the model
        custom_opsets: Optional dict mapping custom domain names to opset versions
        dynamo: Ignored, always uses jit.trace due to MoE complexity
    '''
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = f'{output_dir}/model.onnx'
    print(f"DEBUG: Entering export_onnx (path={onnx_path})")

    print(f"DEBUG: Starting standard torch.onnx.export (jit.trace) to {onnx_path}...")
    
    # Check if model is on meta device (skeleton tracing)
    is_meta = False
    try:
        is_meta = next(model.parameters()).is_meta
    except StopIteration:
        pass
    
    if is_meta:
        print(f"DEBUG: Meta device detected, setting export_params=False for skeleton tracing. ONNX_OPSET_VERSION is {ONNX_OPSET_VERSION}")

    if custom_opsets is None:
        custom_opsets = {"trt_edgellm": 1}
    else:
        custom_opsets["trt_edgellm"] = 1

    with torch.inference_mode():


        # Force dynamo=False to use legacy jit.trace, which is more robust for complex MoE
        print(f"DEBUG: Calling torch.onnx.export with opset_version={ONNX_OPSET_VERSION}")
        torch.onnx.export(model,
                          inputs,
                          onnx_path,
                          export_params=not is_meta,
                          dynamic_axes=dynamic_axes,
                          input_names=input_names,
                          output_names=output_names,
                          opset_version=ONNX_OPSET_VERSION,
                          do_constant_folding=not is_meta, # Constant folding also fails on meta
                          custom_opsets=custom_opsets,
                          dynamo=False)

    t1 = time.time()
    print(f"ONNX export completed in {t1 - t0}s. Apply post-processing...")
    # Post-processing
    onnx.shape_inference.infer_shapes_path(onnx_path)
    onnx_model = onnx.load(onnx_path)
    
    # Pass 1: Initial GraphSurgeon import and attention fusion
    from .surgeon import fuse_attention_nodes, broadcast_heterogeneous_heads, fix_flatten_reshapes, barrier_plugins, cleanup_graph, enforce_plugin_domains
    graph = gs.import_onnx(onnx_model)
    del onnx_model
    gc.collect()

    if qwen35_interleaving:
        print("Applying Qwen 3.5 specific AttentionPlugin transformation and fixing reshapes...")
        graph = fuse_attention_nodes(graph, config=config)
        graph = fix_flatten_reshapes(graph)
    else:
        print("Applying AttentionPlugin transformation, broadcasting heads, and fixing reshapes...")
        graph = fuse_attention_nodes(graph, config=config)
        graph = broadcast_heterogeneous_heads(graph)
        graph = fix_flatten_reshapes(graph)

    graph = barrier_plugins(graph)

    # NVFP4 processing (Surgical modification on GraphSurgeon graph BEFORE final export)
    if _is_nvfp4_quantized(model):
        print("DEBUG: NVFP4 quantization detected. Resolving Identity aliases...")
        for node in graph.nodes:
            if node.op == "TRT_FP4QDQ":
                inp = node.inputs[0]
                while inp.inputs and inp.inputs[0].op == "Identity":
                    inp = inp.inputs[0].inputs[0]
                if isinstance(inp, gs.Constant):
                    unique_const = gs.Constant(name=node.name + "_unique_weight", values=inp.values)
                    node.inputs[0] = unique_const
                else:
                    node.inputs[0] = inp

    # Standard post-processing for custom plugins
    graph = cleanup_graph(graph)
    
    # FINAL EXPORT
    print("DEBUG: Exporting final graph from surgeon...")
    graph.cleanup().toposort()
    onnx_model = gs.export_onnx(graph, do_type_check=False)
    del graph
    gc.collect()

    # Final ModelOpt NVFP4 weight compression
    if _is_nvfp4_quantized(model):
        from modelopt.onnx.export import NVFP4QuantExporter
        print("NVFP4 quantization detected. Processing weights with modelopt...")
        onnx_model = NVFP4QuantExporter.process_model(onnx_model)
        gc.collect()

    # Pass 3: Final Domain Enforcement using raw ONNX API
    onnx_model = enforce_plugin_domains(onnx_model)

    # 3. Save the final ONNX model
    # Large models need to be saved with external data
    print(f"DEBUG: Saving final ONNX model to {onnx_path} (Fragmented External Data enabled)...")
    onnx.save_model(onnx_model,
                    onnx_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=False)

    print(f"Final ONNX model saved to {onnx_path}")

    print(f"Final ONNX model saved to {onnx_path}")
