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

# SPDX-FileCopyrightText: Copyright 2026 Google LLC and contributors
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright 2026 Google LLC and contributors
# SPDX-License-Identifier: Apache-2.0

import sys

import onnx
from onnx import numpy_helper


def patch_onnx(model_path):
    print(f"Patching ONNX shapes for {model_path}...")
    model = onnx.load(model_path)

    # Track which inputs we patched
    patched = []

    for i in model.graph.input:
        if i.name == 'inputs_embeds':
            i.type.tensor_type.shape.dim[0].dim_param = 'batch'
            i.type.tensor_type.shape.dim[1].dim_param = 'seq_len'
            patched.append(i.name)
        elif i.name == 'context_lengths':
            i.type.tensor_type.shape.dim[0].dim_param = 'batch'
            patched.append(i.name)
        elif i.name == 'kvcache_start_index':
            i.type.tensor_type.shape.dim[0].dim_param = 'batch'
            patched.append(i.name)
        elif i.name == 'last_token_ids':
            i.type.tensor_type.shape.dim[0].dim_param = 'batch'
            patched.append(i.name)

    # Patch Reshape shape initializers from [-1, 1, ...] to [-1, 0, ...]
    # to allow dynamic sequence length propagation natively in ONNX.
    patched_initializers = 0
    for initializer in model.graph.initializer:
        arr = numpy_helper.to_array(initializer).copy()
        if len(arr.shape) == 1 and arr.shape[0] in [
                3, 4
        ] and arr[0] == -1 and arr[1] == 1:
            arr[1] = 0  # Copy dim dynamically
            # Write back updated tensor
            patched_tensor = numpy_helper.from_array(arr,
                                                     name=initializer.name)
            initializer.CopyFrom(patched_tensor)
            patched_initializers += 1

    print(
        f"Patched {patched_initializers} Reshape initializers to dynamic copy (0)"
    )

    # Configure allowzero=0 on all Reshape nodes to ensure '0' values in shape inputs
    # are interpreted as 'copy dimension' wildcards rather than static 0-length constraints.
    patched_nodes = 0
    for n in model.graph.node:
        if n.op_type == 'Reshape':
            for attr in n.attribute:
                if attr.name == 'allowzero':
                    attr.i = 0
                    patched_nodes += 1

    print(f"Set allowzero=0 on {patched_nodes} Reshape nodes")

    # Also patch logits output to match inputs_embeds dynamic dimensions
    for o in model.graph.output:
        if o.name == 'logits':
            o.type.tensor_type.shape.dim[0].dim_param = 'batch'
            o.type.tensor_type.shape.dim[1].dim_param = 'num_selected'
            patched.append(o.name)

    onnx.save(model,
              model_path,
              save_as_external_data=True,
              all_tensors_to_one_file=True,
              location="model.onnx.data")
    print(
        f"Successfully patched and saved with external data: {', '.join(patched)}"
    )


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 patch_onnx_shapes.py <model.onnx>")
        sys.exit(1)
    patch_onnx(sys.argv[1])
