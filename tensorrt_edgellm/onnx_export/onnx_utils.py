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

ONNX_OPSET_VERSION = 17

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
                dynamo=False):
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

    print(f"DEBUG: Starting standard torch.onnx.export (jit.trace) to {onnx_path}...")
    with torch.inference_mode():
        # Force dynamo=False to use legacy jit.trace, which is more robust for complex MoE
        torch.onnx.export(model,
                          inputs,
                          onnx_path,
                          export_params=True,
                          dynamic_axes=dynamic_axes,
                          input_names=input_names,
                          output_names=output_names,
                          opset_version=ONNX_OPSET_VERSION,
                          do_constant_folding=True,
                          custom_opsets=custom_opsets,
                          dynamo=False)

    t1 = time.time()
    print(f"ONNX export completed in {t1 - t0}s. Apply post-processing...")
    # Post-processing
    onnx.shape_inference.infer_shapes_path(onnx_path)
    onnx_model = onnx.load(onnx_path)
    graph = None

    if _is_int4_awq_quantized(model):
        from .int4_awq_utils import fix_model_int4_output_dtypes, int4_dq_gemm_to_plugin
        from .quant_utils import INT4QuantExporter
        print(
            "INT4 AWQ quantization detected in the model, compressing some weights to INT4 and inserting int4 gemm plugin"
        )
        onnx_model = INT4QuantExporter.compute_scales(onnx_model)
        onnx_model = INT4QuantExporter.compress_weights(onnx_model)
        onnx_model = INT4QuantExporter.post_process(onnx_model)
        # Fix the Cast nodes and hidden_states output types for INT4 models
        onnx_model = fix_model_int4_output_dtypes(onnx_model)
        graph = gs.import_onnx(onnx_model)
        graph = int4_dq_gemm_to_plugin(graph)

    if _is_nvfp4_quantized(model):
        from .nvfp4_utils import nvfp4_to_plugin
        from .quant_utils import NVFP4QuantExporter
        print(
            "NVFP4 quantization detected in the model, compressing some weights to NVFP4 and inserting nvfp4 gemm plugin"
        )
        onnx_model = NVFP4QuantExporter.compute_scales(onnx_model)
        onnx_model = NVFP4QuantExporter.compress_weights(onnx_model)
        onnx_model = NVFP4QuantExporter.post_process(onnx_model)
        graph = gs.import_onnx(onnx_model)
        graph = nvfp4_to_plugin(graph)

    if _is_mxfp8_quantized(model):
        from .mxfp8_utils import mxfp8_to_plugin
        from .quant_utils import MXFP8QuantExporter
        print(
            "MXFP8 quantization detected in the model, compressing some weights to MXFP8"
        )
        onnx_model = MXFP8QuantExporter.compute_scales(onnx_model)
        onnx_model = MXFP8QuantExporter.compress_weights(onnx_model)
        onnx_model = MXFP8QuantExporter.post_process(onnx_model)
        graph = gs.import_onnx(onnx_model)
        graph = mxfp8_to_plugin(graph)

    if graph is not None:
        from .int4_moe_utils import int4_moe_to_plugin
        # Check if model is MoE
        is_moe = "moe" in str(type(model)).lower()
        if is_moe:
            print("MoE detected, applying Int4MoePlugin transformation")
            graph = int4_moe_to_plugin(graph)

        # Standard post-processing for custom plugins
        graph.cleanup().toposort()
        onnx_model = gs.export_onnx(graph)
        
        # Large models need to be saved with external data
        onnx.save_model(onnx_model,
                        onnx_path,
                        save_as_external_data=True,
                        all_tensors_to_one_file=True,
                        location="model.onnx.data")

    print(f"Final ONNX model saved to {onnx_path}")
