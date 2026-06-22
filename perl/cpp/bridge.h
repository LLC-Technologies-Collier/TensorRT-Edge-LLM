#ifndef C9H_LLM_BRIDGE_H
#define C9H_LLM_BRIDGE_H

#ifdef __cplusplus
#include <builder/llmBuilder.h>
#include <runtime/llmInferenceRuntime.h>
#include <runtime/llmRuntimeUtils.h>
#include <runtime/imageUtils.h>
#include <string>
#include <memory>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#endif

#define PERL_NO_GET_CONTEXT
#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"

using namespace trt_edgellm;

struct RequestJob {
    long id;
    std::string prompt;
    std::vector<rt::Message> messages;
    std::vector<rt::imageUtils::ImageData> image_buffers;

    int max_gen_len;
    float temperature;
    float top_p;
    int top_k;
    uint64_t seed;
    
    // Result
    bool done;
    bool success;
    std::string output_text;
    double elapsed;
    int num_tokens;

    // Streaming support
    bool is_streaming;
    std::queue<int32_t> token_queue;
    std::mutex token_mtx;
};

struct LLMSession {
    long id;
    int slot = -1;
    std::vector<int32_t> input_ids;
    std::vector<int32_t> output_ids;
    int32_t context_length = 0;
    bool is_done = false;
    
    // KV Cache storage (swap area)
    rt::Tensor kv_cache_storage; 
    
    // Generation params
    int max_gen_len = 512;
    float temperature = 1.0f;
    float top_p = 1.0f;
    int top_k = 1;
    uint64_t seed = 42;

    std::queue<int32_t> new_tokens;
    std::mutex mtx;

    bool needs_kv_rewind = false;
    int32_t rewind_start_index = 0;
    int32_t rewind_length_to_clear = 0;
};

struct C9h_LLM_Runtime_Context {
    rt::LLMInferenceRuntime * runtime;
    cudaStream_t stream;
    
    // Async Worker
    std::thread *worker;
    std::mutex mtx;
    std::condition_variable cv;
    std::queue<RequestJob*> queue;
    
    // Session management
    std::unordered_map<long, LLMSession*> sessions;
    std::mutex sessions_mtx;
    
    bool stop = false;
    int notify_fd = -1; // Write end of pipe
    int read_fd = -1;   // Read end of pipe
};

typedef struct C9h_LLM_Runtime_Context * C9h_LLM_Runtime;
typedef struct LLMSession * C9h_LLM_Session;

#ifdef __cplusplus
extern "C" {
#endif

void worker_loop(C9h_LLM_Runtime_Context* ctx);
void parse_perl_messages(pTHX_ SV* request_ref, RequestJob* job);

C9h_LLM_Runtime bridge_init_runtime(const char* engine_dir, const char* multimodal_dir, bool enable_cuda_graph);
void bridge_destroy_runtime(C9h_LLM_Runtime ctx);
int bridge_get_notify_fd(C9h_LLM_Runtime ctx);
SV* bridge_generate_async(pTHX_ C9h_LLM_Runtime ctx, SV* request_ref, int max_gen_len, float temperature, float top_p, int top_k, unsigned long seed, bool is_streaming);
bool bridge_is_job_done(C9h_LLM_Runtime ctx, long job_id);
AV* bridge_poll_tokens(pTHX_ C9h_LLM_Runtime ctx, long job_id);
AV* bridge_tokenize(pTHX_ C9h_LLM_Runtime ctx, SV* request_ref, bool apply_chat_template);
SV* bridge_decode(pTHX_ C9h_LLM_Runtime ctx, SV* token_ids_ref);
SV* bridge_collect_job(pTHX_ C9h_LLM_Runtime ctx, long job_id);
AV* bridge_get_embedding(pTHX_ C9h_LLM_Runtime ctx, const char* text);
void bridge_release_spare_memory(C9h_LLM_Runtime ctx);
SV* bridge_generate(pTHX_ C9h_LLM_Runtime ctx, const char* prompt, int max_gen_len, float temperature, float top_p, int top_k, unsigned long seed);

// Session APIs for Real-time Live API
C9h_LLM_Session bridge_create_session(C9h_LLM_Runtime ctx);
void bridge_destroy_session(C9h_LLM_Runtime ctx, C9h_LLM_Session session);
void bridge_session_push_content(pTHX_ C9h_LLM_Runtime ctx, C9h_LLM_Session session, SV* content_ref);
AV* bridge_session_poll_tokens(pTHX_ C9h_LLM_Session session);
bool bridge_session_is_done(C9h_LLM_Session session);

struct bridge_builder_config {
    int32_t max_input_len;
    int32_t max_kv_cache_capacity;
    int32_t max_batch_size;
    bool is_vlm;
    int64_t weight_streaming_budget;
    int32_t max_image_tokens;
    bool is_eagle;
};

bool bridge_build(const char* onnx_dir, const char* engine_dir, struct bridge_builder_config* config);
bool bridge_init_plugins(const char* plugin_path);
const char* bridge_get_runtime_version();

#ifdef __cplusplus
}
#endif

#endif // C9H_LLM_BRIDGE_H
