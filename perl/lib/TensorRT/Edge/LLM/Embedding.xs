#include "cpp/bridge.h"

MODULE = TensorRT::Edge::LLM::Embedding  PACKAGE = TensorRT::Edge::LLM::Embedding

PROTOTYPES: ENABLE

C9h_LLM_Runtime
_xs_init_runtime(self, engine_dir)
    SV* self
    const char* engine_dir
    CODE:
        // Reuse bridge_init_runtime (defaults for multimodal and cuda_graph)
        RETVAL = bridge_init_runtime(engine_dir, "", true);
    OUTPUT:
        RETVAL

bool
_xs_init_plugins(self, plugin_path)
    SV* self
    const char* plugin_path
    CODE:
        #include <dlfcn.h>
        void* handle = dlopen(plugin_path, RTLD_NOW | RTLD_GLOBAL);
        if (!handle) {
            fprintf(stderr, "Failed to dlopen %s: %s\n", plugin_path, dlerror());
            RETVAL = false;
        } else {
            typedef bool (*InitFunc)(void*, const char*);
            InitFunc initFunc = (InitFunc)dlsym(handle, "initTrtLlmPlugins");
            if (!initFunc) initFunc = (InitFunc)dlsym(handle, "initLibNvInferPlugins");
            if (initFunc) initFunc(nullptr, "");
            RETVAL = true;
        }

        const char* edge_plugin = getenv("EDGELLM_PLUGIN_PATH");
        if (edge_plugin && strcmp(plugin_path, edge_plugin) != 0) {
            void* eh = dlopen(edge_plugin, RTLD_NOW | RTLD_GLOBAL);
            if (eh) fprintf(stderr, "Successfully loaded Edge-LLM plugins from %s\n", edge_plugin);
        }
    OUTPUT:
        RETVAL

void
_xs_destroy_runtime(self, ctx)
    SV* self
    C9h_LLM_Runtime ctx
    CODE:
        bridge_destroy_runtime(ctx);

AV*
_xs_get_embedding(self, ctx, text)
    SV* self
    C9h_LLM_Runtime ctx
    const char* text
    CODE:
        RETVAL = (AV*)bridge_get_embedding(aTHX_ ctx, text);
    OUTPUT:
        RETVAL
