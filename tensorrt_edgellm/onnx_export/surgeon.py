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

import onnx
import onnx_graphsurgeon as gs
import numpy as np
from typing import Dict, List, Optional

def fuse_attention_nodes(graph, config=None, is_hybrid=False):
    """
    Tags AttentionPlugin nodes with domain='trt' and prunes shuffle artifacts
    on k/v inputs to satisfy TensorRT 10.x parser requirements.
    """
    print(f"DEBUG Surgeon: fuse_attention_nodes called with {len(graph.nodes)} total nodes.")
    
    nodes_to_remove = []
    
    # Identify custom attention nodes
    attention_ops = ["AttentionPlugin", "qwen3_delta_attention"]
    attention_nodes = [n for n in graph.nodes if any(op in n.op for op in attention_ops)]
    print(f"DEBUG Surgeon: Found {len(attention_nodes)} custom attention nodes.")

    for node in attention_nodes:
        # Force full 'domain::op' name AND explicit domain property
        node.op = "AttentionPlugin"
        node.domain = "trt_edgellm"
        
        # Remove any confusing attributes
        if "plugin_version" in node.attrs:
            del node.attrs["plugin_version"]
        
        print(f"DEBUG Surgeon: Tagging {node.name} with op='{node.op}' domain='{node.domain}'")
        
        # For 1D-Neutralized trace, we expect inputs[1] (k) and inputs[2] (v) 
        # to be 1D. PyTorch often inserts chains of Reshape/Transpose ops
        # that cause IShuffleLayer errors in the TensorRT parser.
        # We trace back and bypass these chains entirely.
        for i in [1, 2]: # k and v
            if i < len(node.inputs):
                curr_var = node.inputs[i]
                depth = 0
                while len(curr_var.inputs) > 0 and depth < 10:
                    producer = curr_var.inputs[0]
                    if producer.op in ["Transpose", "Reshape", "Identity", "Unsqueeze", "Squeeze"]:
                        if producer not in nodes_to_remove:
                            nodes_to_remove.append(producer)
                        curr_var = producer.inputs[0]
                        depth += 1
                    else:
                        break
                # Final connection: connect plugin input directly to the source 1D tensor
                node.inputs[i] = curr_var

    graph.cleanup()
    return graph

def enforce_plugin_domains(onnx_model):
    """
    Final Domain Enforcement using raw ONNX API.
    GraphSurgeon sometimes strips domains during export if they aren't in a known opset.
    """
    print("DEBUG Surgeon: enforce_plugin_domains called.")
    fixed_domains = 0
    for node in onnx_model.graph.node:
        op = node.op_type
        # Match our custom ops even if they have prefixes like 'trt::'
        if "AttentionPlugin" in op or "qwen3_delta_attention" in op:
            node.domain = "trt_edgellm"
            fixed_domains += 1
            print(f"DEBUG Surgeon: [FINAL] Force domain='trt_edgellm' for node={node.name} (op={op})")
        elif "TRT_FP4" in op:
            node.domain = ""
            fixed_domains += 1
            print(f"DEBUG Surgeon: [FINAL] Force domain='' for node={node.name} (op={op})")
            
    print(f"DEBUG Surgeon: enforce_plugin_domains finished. Forced domain for {fixed_domains} nodes.")

    return onnx_model

def fix_flatten_reshapes(graph):
    """
    Placeholder for future flatten-specific fixes.
    """
    return graph

def cleanup_graph(graph):
    """
    Performs general graph cleanup, including NVFP4 signature fixes and 
    removing non-standard attributes that cause TensorRT parser SEGVs.
    """
    dummy_scale = gs.Constant(name="dummy_fp4_scale_tensor", values=np.array([1.0], dtype=np.float32))
    fixed_nvfp4_nodes = 0
    fixed_dq_nodes = 0
    
    for node in graph.nodes:
        # 1. Fix NVFP4 custom ops
        if node.op in ["TRT_FP4DynamicQuantize", "TRT_FP4QDQ"]:
            # Fix inputs: ensure scale tensor is present
            if len(node.inputs) == 1:
                node.inputs.append(dummy_scale)
            # Fix outputs: ensure dummy scale output is present
            if len(node.outputs) == 1:
                dummy_out = gs.Variable(name=node.outputs[0].name + "_dummy_scale_out", dtype=np.float32, shape=[1])
                node.outputs.append(dummy_out)
            fixed_nvfp4_nodes += 1
            node.domain = ""
            node.op = node.op.split("::")[-1] # Ensure no domain in op name
            
        # 2. Fix non-standard attributes in standard ops
        if node.op == "DequantizeLinear":
            if "block_size" in node.attrs:
                print(f"DEBUG Surgeon: Removing invalid 'block_size' attribute from {node.name}")
                del node.attrs["block_size"]
                fixed_dq_nodes += 1

    if fixed_nvfp4_nodes > 0:
        print(f"DEBUG Surgeon: Fixed signature for {fixed_nvfp4_nodes} NVFP4 nodes.")
    if fixed_dq_nodes > 0:
        print(f"DEBUG Surgeon: Cleaned {fixed_dq_nodes} DequantizeLinear nodes.")
        
    return graph

def barrier_plugins(graph):
    return graph

def broadcast_heterogeneous_heads(graph):
    return graph
