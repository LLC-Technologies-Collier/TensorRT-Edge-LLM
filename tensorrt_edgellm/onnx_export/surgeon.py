# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import onnx
import onnx_graphsurgeon as gs
import numpy as np
from typing import Dict, List, Optional

def fuse_attention_nodes(graph, config=None, is_hybrid=False):
    """
    Tags AttentionPlugin nodes with domain='' and prunes shuffle artifacts.
    """
    attention_ops = ["AttentionPlugin", "qwen3_delta_attention"]
    attention_nodes = [n for n in graph.nodes if any(op in n.op for op in attention_ops)]
    print(f"DEBUG Surgeon: Found {len(attention_nodes)} custom attention nodes.")

    for node in attention_nodes:
        # Standardize: empty domain and version '1' for standard TRT registry matching
        node.domain = ""
        node.attrs["plugin_version"] = "1"
        
        print(f"DEBUG Surgeon: Tagging {node.name} (op={node.op}) with domain='' and version='1'")
        
    graph.cleanup()
    return graph

def enforce_plugin_domains(onnx_model):
    """
    Final Domain Enforcement using raw ONNX API.
    """
    print("DEBUG Surgeon: enforce_plugin_domains called.")
    fixed_domains = 0
    for node in onnx_model.graph.node:
        op = node.op_type
        if any(x in op for x in ["AttentionPlugin", "qwen3_delta_attention"]):
            node.domain = ""
            fixed_domains += 1
        elif "TRT_FP4" in op:
            node.domain = ""
            fixed_domains += 1
            
    print(f"DEBUG Surgeon: enforce_plugin_domains finished. Forced domain for {fixed_domains} nodes.")
    return onnx_model

def cleanup_graph(graph):
    """
    General hygiene for NVFP4 and DequantizeLinear nodes.
    """
    fixed_nvfp4_nodes = 0
    fixed_dq_nodes = 0
    
    for node in graph.nodes:
        if node.op in ["TRT_FP4DynamicQuantize", "TRT_FP4QDQ"]:
            if len(node.outputs) == 1:
                dummy_out = gs.Variable(name=node.outputs[0].name + "_dummy_scale_out", dtype=np.float32, shape=[1])
                node.outputs.append(dummy_out)
            fixed_nvfp4_nodes += 1
            node.domain = ""
            
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

def fix_flatten_reshapes(graph):
    """
    Removes flattening Reshape nodes that PyTorch often inserts before plugin calls.
    Ensures qwen3_delta_attention nodes receive 4D tensors.
    """
    print("DEBUG Surgeon: fix_flatten_reshapes starting...")
    target_ops = ["qwen3_delta_attention"]
    removed_count = 0
    
    for node in graph.nodes:
        if node.op in target_ops:
            # Check the first 5 inputs (Q, K, V, g, beta)
            for i in range(5):
                if len(node.inputs) > i:
                    input_node = node.inputs[i].inputs[0] if node.inputs[i].inputs else None
                    if input_node and input_node.op == "Reshape":
                        print(f"DEBUG Surgeon: Pruning flattening Reshape {input_node.name} for {node.name} input {i}")
                        node.inputs[i] = input_node.inputs[0]
                        removed_count += 1
                        
    if removed_count > 0:
        graph.cleanup()
        print(f"DEBUG Surgeon: Removed {removed_count} redundant Reshape nodes.")
    return graph
