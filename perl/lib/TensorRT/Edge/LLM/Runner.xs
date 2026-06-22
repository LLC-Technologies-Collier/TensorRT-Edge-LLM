#include "cpp/bridge.h"

MODULE = TensorRT::Edge::LLM::Runner  PACKAGE = TensorRT::Edge::LLM::Runner

PROTOTYPES: ENABLE

C9h_LLM_Runtime
_xs_init_runtime(self, engine_dir, multimodal_dir="", enable_cuda_graph=true)
    SV* self
    const char* engine_dir
    const char* multimodal_dir
    bool enable_cuda_graph
    CODE:
        RETVAL = bridge_init_runtime(engine_dir, multimodal_dir, enable_cuda_graph);
    OUTPUT:
        RETVAL

void
_xs_destroy_runtime(self, ctx)
    SV* self
    C9h_LLM_Runtime ctx
    CODE:
        bridge_destroy_runtime(ctx);

int
_xs_get_notify_fd(self, ctx)
    SV* self
    C9h_LLM_Runtime ctx
    CODE:
        RETVAL = bridge_get_notify_fd(ctx);
    OUTPUT:
        RETVAL

SV*
_xs_generate_async(self, ctx, request_ref, max_gen_len=128, temperature=1.0, top_p=1.0, top_k=1, seed=42, is_streaming=false)
    SV* self
    C9h_LLM_Runtime ctx
    SV* request_ref
    int max_gen_len
    float temperature
    float top_p
    int top_k
    unsigned long seed
    bool is_streaming
    CODE:
        RETVAL = bridge_generate_async(aTHX_ ctx, request_ref, max_gen_len, temperature, top_p, top_k, seed, is_streaming);
    OUTPUT:
        RETVAL

bool
_xs_is_job_done(self, ctx, job_id)
    SV* self
    C9h_LLM_Runtime ctx
    long job_id
    CODE:
        RETVAL = bridge_is_job_done(ctx, job_id);
    OUTPUT:
        RETVAL

AV*
_xs_poll_tokens(self, ctx, job_id)
    SV* self
    C9h_LLM_Runtime ctx
    long job_id
    CODE:
        RETVAL = (AV*)bridge_poll_tokens(aTHX_ ctx, job_id);
    OUTPUT:
        RETVAL

AV*
_xs_tokenize(self, ctx, request_ref, apply_chat_template=true)
    SV* self
    C9h_LLM_Runtime ctx
    SV* request_ref
    bool apply_chat_template
    CODE:
        RETVAL = (AV*)bridge_tokenize(aTHX_ ctx, request_ref, apply_chat_template);
    OUTPUT:
        RETVAL

SV*
_xs_decode(self, ctx, token_ids_ref)
    SV* self
    C9h_LLM_Runtime ctx
    SV* token_ids_ref
    CODE:
        RETVAL = bridge_decode(aTHX_ ctx, token_ids_ref);
    OUTPUT:
        RETVAL

SV*
_xs_collect_job(self, ctx, job_id)
    SV* self
    C9h_LLM_Runtime ctx
    long job_id
    CODE:
        RETVAL = bridge_collect_job(aTHX_ ctx, job_id);
    OUTPUT:
        RETVAL

void
_xs_release_spare_memory(self, ctx)
    SV* self
    C9h_LLM_Runtime ctx
    CODE:
        bridge_release_spare_memory(ctx);

SV*
_xs_generate(self, ctx, prompt, max_gen_len=128, temperature=1.0, top_p=1.0, top_k=1, seed=42)
    SV* self
    C9h_LLM_Runtime ctx
    const char* prompt
    int max_gen_len
    float temperature
    float top_p
    int top_k
    unsigned long seed
    CODE:
        RETVAL = bridge_generate(aTHX_ ctx, prompt, max_gen_len, temperature, top_p, top_k, seed);
    OUTPUT:
        RETVAL

C9h_LLM_Session
_xs_create_session(self, ctx)
    SV* self
    C9h_LLM_Runtime ctx
    CODE:
        RETVAL = bridge_create_session(ctx);
    OUTPUT:
        RETVAL

void
_xs_destroy_session(self, ctx, session)
    SV* self
    C9h_LLM_Runtime ctx
    C9h_LLM_Session session
    CODE:
        bridge_destroy_session(ctx, session);

void
_xs_session_push_content(self, ctx, session, content_ref)
    SV* self
    C9h_LLM_Runtime ctx
    C9h_LLM_Session session
    SV* content_ref
    CODE:
        bridge_session_push_content(aTHX_ ctx, session, content_ref);

AV*
_xs_session_poll_tokens(self, session)
    SV* self
    C9h_LLM_Session session
    CODE:
        RETVAL = (AV*)bridge_session_poll_tokens(aTHX_ session);
    OUTPUT:
        RETVAL

bool
_xs_session_is_done(self, session)
    SV* self
    C9h_LLM_Session session
    CODE:
        RETVAL = bridge_session_is_done(session);
    OUTPUT:
        RETVAL

AV*
_xs_get_embedding(self, ctx, text)
    SV* self
    C9h_LLM_Runtime ctx
    const char* text
    CODE:
        RETVAL = (AV*)bridge_get_embedding(aTHX_ ctx, text);
    OUTPUT:
        RETVAL
