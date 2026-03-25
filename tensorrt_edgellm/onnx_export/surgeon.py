import onnx
import onnx_graphsurgeon as gs
import numpy as np

def fix_flatten_reshapes(graph):
    """
    Finds Reshape nodes that flatten [B, S, H] to [B*S, H] and ensures 
    the output shape is calculated dynamically to avoid volume mismatches.
    """
    for node in list(graph.nodes):
        if node.op == "Reshape":
            inp = node.inputs[0]
            out = node.outputs[0]
            
            # Target Reshapes: 3D input [B, S, H] -> 2D output [B*S, H]
            if inp.shape and len(inp.shape) == 3 and out.shape and len(out.shape) == 2:
                print(f"DEBUG Surgeon: Forcing dynamic flatten for Reshape {node.name}")
                
                # 1. Get input shape [B, S, H]
                inp_shape = gs.Variable(name=node.name + "_inp_shape_f", dtype=np.int64, shape=(3,))
                graph.nodes.append(gs.Node(op="Shape", inputs=[inp], outputs=[inp_shape]))
                
                # 2. Extract B (0) and S (1)
                b_val = gs.Variable(name=node.name + "_b_f", dtype=np.int64, shape=(1,))
                graph.nodes.append(gs.Node(op="Slice", inputs=[inp_shape, 
                                                              gs.Constant(name=node.name + "_s0_f", values=np.array([0], dtype=np.int64)),
                                                              gs.Constant(name=node.name + "_e0_f", values=np.array([1], dtype=np.int64))], 
                                           outputs=[b_val]))
                
                s_val = gs.Variable(name=node.name + "_s_f", dtype=np.int64, shape=(1,))
                graph.nodes.append(gs.Node(op="Slice", inputs=[inp_shape, 
                                                              gs.Constant(name=node.name + "_s1_f", values=np.array([1], dtype=np.int64)),
                                                              gs.Constant(name=node.name + "_e1_f", values=np.array([2], dtype=np.int64))], 
                                           outputs=[s_val]))
                
                # 3. Multiply B * S
                prod_val = gs.Variable(name=node.name + "_bs_prod", dtype=np.int64, shape=(1,))
                graph.nodes.append(gs.Node(op="Mul", inputs=[b_val, s_val], outputs=[prod_val]))
                
                # 4. Get H (2) from input shape
                h_val_v = gs.Variable(name=node.name + "_h_f_v", dtype=np.int64, shape=(1,))
                graph.nodes.append(gs.Node(op="Slice", inputs=[inp_shape, 
                                                              gs.Constant(name=node.name + "_s2_f", values=np.array([2], dtype=np.int64)),
                                                              gs.Constant(name=node.name + "_e2_f", values=np.array([3], dtype=np.int64))], 
                                           outputs=[h_val_v]))
                
                new_shape = gs.Variable(name=node.name + "_dyn_shape", dtype=np.int64, shape=(2,))
                graph.nodes.append(gs.Node(op="Concat", inputs=[prod_val, h_val_v], outputs=[new_shape], attrs={"axis": 0}))
                
                # Update node to use dynamic shape
                node.inputs[1] = new_shape

    return graph

def broadcast_heterogeneous_heads(graph):
    """
    Finds MatMul nodes where the head dimension (dim 1 of 4D tensor) doesn't match
    and injects Tile nodes to make them conformable for TensorRT.
    """
    for node in list(graph.nodes):
        if node.op == "MatMul":
            in0, in1 = node.inputs
            if (in0.shape and len(in0.shape) == 4 and isinstance(in0.shape[1], int) and
                in1.shape and len(in1.shape) == 4 and isinstance(in1.shape[1], int)):
                
                h0 = in0.shape[1]
                h1 = in1.shape[1]
                
                if h0 != h1:
                    print(f"DEBUG Surgeon: Found heterogeneous MatMul {node.name}: {h0} vs {h1} heads")
                    if h0 > h1 and h0 % h1 == 0:
                        ratio = h0 // h1
                        repeats_const = gs.Constant(name=node.name + "_h_repeats", values=np.array([1, ratio, 1, 1], dtype=np.int64))
                        tiled_var = gs.Variable(name=in1.name + "_h_tiled", dtype=in1.dtype)
                        graph.nodes.append(gs.Node(op="Tile", inputs=[in1, repeats_const], outputs=[tiled_var]))
                        node.inputs[1] = tiled_var
                    elif h1 > h0 and h1 % h0 == 0:
                        ratio = h1 // h0
                        repeats_const = gs.Constant(name=node.name + "_h_repeats", values=np.array([1, ratio, 1, 1], dtype=np.int64))
                        tiled_var = gs.Variable(name=in0.name + "_h_tiled", dtype=in0.dtype)
                        graph.nodes.append(gs.Node(op="Tile", inputs=[in0, repeats_const], outputs=[tiled_var]))
                        node.inputs[0] = tiled_var

    return graph

def fuse_attention_nodes(graph):
    """
    Decomposes AttentionPlugin with internal Slice barriers to force Myelin fragmentation.
    """
    for node in list(graph.nodes):
        if node.op == "AttentionPlugin":
            # Extract inputs
            q, k, v, kv_cache, context_len, rope, kv_start = node.inputs[:7]
            
            print(f"DEBUG Surgeon: Decomposing {node.name} with internal fragmentation barriers")
            
            # 0. Helper for barriers
            def add_barrier(name, inp):
                inp_sh = gs.Variable(name=name + "_sh", dtype=np.int64, shape=(len(inp.shape),))
                graph.nodes.append(gs.Node(op="Shape", inputs=[inp], outputs=[inp_sh]))
                out = gs.Variable(name=name + "_bar", dtype=inp.dtype)
                starts = gs.Constant(name=name + "_st", values=np.zeros(len(inp.shape), dtype=np.int64))
                axes = gs.Constant(name=name + "_ax", values=np.array(list(range(len(inp.shape))), dtype=np.int64))
                graph.nodes.append(gs.Node(op="Slice", name=name + "_node",
                                           inputs=[inp, starts, inp_sh, axes], outputs=[out]))
                return out

            # 1. Normalize
            q_h = node.attrs.get("num_q_heads", 64)
            kv_h = node.attrs.get("num_kv_heads", 2)
            h_dim = node.attrs.get("head_size", 128)
            
            # [B, H, S, D] - Simplified for validation
            q_step = q
            k_step = k
            v_step = v

            # 2. QK MatMul + Barrier
            qk_raw = gs.Variable(name=node.name + "_qk_raw", dtype=q.dtype)
            k_t = gs.Variable(name=node.name + "_k_t", dtype=k.dtype)
            graph.nodes.append(gs.Node(op="Transpose", inputs=[k_step], outputs=[k_t], attrs={"perm": [0, 1, 3, 2]}))
            graph.nodes.append(gs.Node(op="MatMul", inputs=[q_step, k_t], outputs=[qk_raw]))
            qk_out = add_barrier(node.name + "_qk", qk_raw)
            
            # 3. Softmax + Barrier
            probs_raw = gs.Variable(name=node.name + "_probs_raw", dtype=q.dtype)
            graph.nodes.append(gs.Node(op="Softmax", inputs=[qk_out], outputs=[probs_raw], attrs={"axis": -1}))
            probs_out = add_barrier(node.name + "_probs", probs_raw)
            
            # 4. PV MatMul + Barrier
            attn_raw = gs.Variable(name=node.name + "_attn_raw", dtype=q.dtype)
            graph.nodes.append(gs.Node(op="MatMul", inputs=[probs_out, v_step], outputs=[attn_raw]))
            attn_out = add_barrier(node.name + "_attn", attn_raw)
            
            # 5. Connect outputs
            graph.nodes.append(gs.Node(op="Identity", inputs=[attn_out], outputs=[node.outputs[0]]))
            graph.nodes.append(gs.Node(op="Identity", inputs=[kv_cache], outputs=[node.outputs[1]]))
            
            node.outputs = []
            node.op = "Identity"
            node.inputs = [q]
            
        elif node.op == "int4_moe_plugin":
            node.op = "Int4MoePlugin"
        elif node.op == "int4_gemm_plugin":
            node.op = "Int4GroupwiseGemmPlugin"

    return graph

def barrier_plugins(graph):
    """
    Injects a dynamic Slice barrier between Attention and MLP.
    """
    print("DEBUG Surgeon: Injecting block boundary barrier between Attention and MLP")
    
    boundary_tensor_name = "/layers.0/post_attention_layernorm/Cast_1_output_0"
    
    for node in list(graph.nodes):
        for i, inp in enumerate(node.inputs):
            if inp.name == boundary_tensor_name:
                inp_shape = gs.Variable(name=node.name + "_bound_sh", dtype=np.int64, shape=(len(inp.shape),))
                graph.nodes.append(gs.Node(op="Shape", inputs=[inp], outputs=[inp_shape]))
                
                barrier_out = gs.Variable(name=inp.name + "_bound_bar", dtype=inp.dtype)
                starts = gs.Constant(name=node.name + "_bound_st", values=np.zeros(len(inp.shape), dtype=np.int64))
                axes = gs.Constant(name=node.name + "_bound_ax", values=np.array(list(range(len(inp.shape))), dtype=np.int64))
                
                graph.nodes.append(gs.Node(op="Slice", name=node.name + "_bound_slice",
                                           inputs=[inp, starts, inp_shape, axes], 
                                           outputs=[barrier_out]))
                node.inputs[i] = barrier_out

    return graph

def cleanup_graph(graph):
    graph.cleanup().toposort()
    return graph
