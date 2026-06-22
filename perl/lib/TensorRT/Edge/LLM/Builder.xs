#include "cpp/bridge.h"

MODULE = TensorRT::Edge::LLM::Builder  PACKAGE = TensorRT::Edge::LLM::Builder

PROTOTYPES: ENABLE

bool
_xs_init_plugins(self, plugin_path)
    SV* self
    const char* plugin_path
    CODE:
        RETVAL = bridge_init_plugins(plugin_path);
    OUTPUT:
        RETVAL

bool
_xs_build(self, onnx_dir, engine_dir, max_input_len, max_kv_cache_capacity, max_batch_size, is_vlm, weight_streaming_budget, max_image_tokens, is_eagle)
    SV* self
    const char* onnx_dir
    const char* engine_dir
    int64_t max_input_len
    int64_t max_kv_cache_capacity
    int64_t max_batch_size
    bool is_vlm
    int64_t weight_streaming_budget
    int64_t max_image_tokens
    bool is_eagle
    CODE:
        struct bridge_builder_config bc;
        bc.max_input_len = (int32_t)max_input_len;
        bc.max_kv_cache_capacity = (int32_t)max_kv_cache_capacity;
        bc.max_batch_size = (int32_t)max_batch_size;
        bc.is_vlm = is_vlm;
        bc.weight_streaming_budget = weight_streaming_budget;
        bc.max_image_tokens = (int32_t)max_image_tokens;
        bc.is_eagle = is_eagle;

        RETVAL = bridge_build(onnx_dir, engine_dir, &bc);
    OUTPUT:
        RETVAL

const char*
_xs_get_runtime_version(self)
    SV* self
    CODE:
        RETVAL = bridge_get_runtime_version();
    OUTPUT:
        RETVAL
