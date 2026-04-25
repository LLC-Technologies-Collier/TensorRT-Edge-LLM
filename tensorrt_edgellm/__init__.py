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
"""
TensorRT Edge-LLM - A Python package for quantizing and exporting LLMs for edge deployment.
"""

def patch_qwen3_5_config():
    """Monkey-patch Qwen3.5 config classes to expose text_config attributes directly."""
    try:
        import transformers
        # Try to patch Qwen3_5Config if it exists
        if hasattr(transformers, "Qwen3_5Config"):
            cls = transformers.Qwen3_5Config
            for attr in ["hidden_size", "num_attention_heads", "num_key_value_heads", 
                        "max_position_embeddings", "num_hidden_layers", "rms_norm_eps", "vocab_size"]:
                if not hasattr(cls, attr):
                    setattr(cls, attr, property(lambda self, a=attr: getattr(self.text_config, a)))
        
        # Try to patch Qwen3_5MoeConfig if it exists
        if hasattr(transformers, "Qwen3_5MoeConfig"):
            cls = transformers.Qwen3_5MoeConfig
            for attr in ["hidden_size", "num_attention_heads", "num_key_value_heads", 
                        "max_position_embeddings", "num_hidden_layers", "rms_norm_eps", "vocab_size"]:
                if not hasattr(cls, attr):
                    setattr(cls, attr, property(lambda self, a=attr: getattr(self.text_config, a)))
    except ImportError:
        pass

# Apply the patch immediately upon import
patch_qwen3_5_config()

from .onnx_export.audio_export import audio_export
from .onnx_export.llm_export import export_draft_model, export_llm_model
from .onnx_export.lora import (insert_lora_and_save,
                               process_lora_weights_and_save)
from .onnx_export.visual_export import visual_export
from .quantization.llm_quantization import (quantize_and_save_draft,
                                            quantize_and_save_llm)
from .vocab_reduction.vocab_reduction import reduce_vocab_size

try:
    from .version import __version__
except ImportError:
    __version__ = "unknown"

__all__ = [
    "quantize_and_save_llm",
    "quantize_and_save_draft",
    "export_draft_model",
    "export_llm_model",
    "visual_export",
    "audio_export",
    "insert_lora_and_save",
    "process_lora_weights_and_save",
    "reduce_vocab_size",
]
