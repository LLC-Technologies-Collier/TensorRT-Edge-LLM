#include "bridge.h"
#include "common/version.h"
#include <unistd.h>
#include <malloc.h>
#include <algorithm>
#include <cstring>
#include <vector>
#undef seed
#include <random>
#include <cmath>
#include <utility>
#include "common/bindingNames.h"
#include "kernels/embeddingKernels/embeddingKernels.h"

// Correct FP32 sampling with argmax
int32_t sample_token(rt::Tensor& logits, int32_t slot, int32_t vocab_size, float temp, int top_k, float top_p, uint64_t seed, cudaStream_t stream) {
    std::vector<float> host_logits(vocab_size);
    cudaMemcpyAsync(host_logits.data(), (float*)logits.rawPointer() + slot * vocab_size, vocab_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 1. Handle Greedy Search edge case (temp <= 0.0f)
    if (temp <= 0.0f) {
        return (int32_t)std::distance(host_logits.begin(), std::max_element(host_logits.begin(), host_logits.end()));
    }

    // 2. Apply Temperature and find max logit for numerical stability
    float max_logit = -1e9;
    for (int i = 0; i < vocab_size; ++i) {
        host_logits[i] /= temp;
        if (host_logits[i] > max_logit) max_logit = host_logits[i];
    }

    // 3. Compute Softmax probabilities
    std::vector<std::pair<float, int32_t>> probs(vocab_size);
    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        float p = std::exp(host_logits[i] - max_logit);
        probs[i] = {p, (int32_t)i};
        sum_exp += p;
    }
    for (int i = 0; i < vocab_size; ++i) {
        probs[i].first /= sum_exp; 
    }

    // 4. Sort by probability (descending) for Top-K / Top-P
    std::sort(probs.begin(), probs.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    // 5. Apply Top-K
    if (top_k > 0 && top_k < vocab_size) {
        probs.resize(top_k);
    }

    // 6. Apply Top-P (Nucleus Sampling)
    if (top_p > 0.0f && top_p < 1.0f) {
        float cumulative_prob = 0.0f;
        size_t valid_count = 0;
        for (size_t i = 0; i < probs.size(); ++i) {
            cumulative_prob += probs[i].first;
            valid_count++;
            if (cumulative_prob >= top_p) break;
        }
        probs.resize(valid_count);
    }

    // 7. Sample from the final distribution
    std::vector<float> final_weights(probs.size());
    for (size_t i = 0; i < probs.size(); ++i) {
        final_weights[i] = probs[i].first;
    }

    std::mt19937 gen(seed);
    std::discrete_distribution<> dist(final_weights.begin(), final_weights.end());
    
    int sampled_index = dist(gen);
    return probs[sampled_index].second;
}

void worker_loop(C9h_LLM_Runtime_Context* ctx) {
    printf("[Worker] Loop start\n");
    auto* runner = ctx->runtime->getEngineRunner();
    auto const& config = runner->getEngineConfig();
    int32_t max_batch = config.maxSupportedBatchSize;
    int32_t vocab_size = config.outputVocabSize;
    int32_t max_input_len = config.maxSupportedInputLength;

    printf("[Worker] Config: max_batch=%d, vocab_size=%d, max_input_len=%d\n", max_batch, vocab_size, max_input_len);
    if (max_batch <= 0) max_batch = 1;
    if (vocab_size <= 0) vocab_size = 1;
    if (max_input_len <= 0) max_input_len = 1;

    printf("[Worker] Allocating tensors...\n");
    // Pre-allocate batch-sized tensors
    rt::Tensor step_input_ids({max_batch, max_input_len}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "input_ids");
    rt::Tensor step_inputs_embeds({max_batch, max_input_len, config.hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, trt_edgellm::binding_names::kInputsEmbeds);
    rt::Tensor step_logits({max_batch, vocab_size}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, trt_edgellm::binding_names::kLogits);
    rt::Tensor step_context_lengths({max_batch}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, trt_edgellm::binding_names::kContextLengths);
    rt::Tensor kv_cache_start_indices({max_batch}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, trt_edgellm::binding_names::kKVCacheStartIndex);
    
    rt::Tensor batch_reuse_lengths({max_batch}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32);
    std::vector<int32_t> host_context_lengths(max_batch, 0);
    std::vector<int32_t> host_reuse_lengths(max_batch, 0);
    std::vector<int32_t> host_start_indices(max_batch, 0);
    printf("[Worker] Tensors allocated. Entering loop.\n");

    while (true) {
        std::vector<RequestJob*> batch_jobs;
        std::vector<LLMSession*> active_sessions;
        
        {
            std::unique_lock<std::mutex> lock(ctx->mtx);
            ctx->cv.wait_for(lock, std::chrono::milliseconds(10), [ctx]{ return ctx->stop || !ctx->queue.empty(); });
            if (ctx->stop && ctx->queue.empty()) return;
            while (!ctx->queue.empty() && batch_jobs.size() < max_batch) {
                batch_jobs.push_back(ctx->queue.front());
                ctx->queue.pop();
            }
        }

        // Process standard requests in a batch
        if (!batch_jobs.empty()) {
            rt::LLMGenerationRequest batch_request;
            // Use parameters from the first job for the entire batch
            batch_request.maxGenerateLength = batch_jobs[0]->max_gen_len;
            batch_request.temperature = batch_jobs[0]->temperature;
            batch_request.topP = batch_jobs[0]->top_p;
            batch_request.topK = batch_jobs[0]->top_k;
            batch_request.randomSeed = batch_jobs[0]->seed;

            batch_request.tokenCallback = [&](int32_t batch_idx, int32_t token_id) {
                if (batch_idx >= 0 && (size_t)batch_idx < batch_jobs.size()) {
                    auto* job = batch_jobs[batch_idx];
                    if (job->is_streaming) {
                        std::lock_guard<std::mutex> l(job->token_mtx);
                        job->token_queue.push(token_id);
                        if (ctx->notify_fd != -1) { long jid = job->id; write(ctx->notify_fd, &jid, sizeof(jid)); }
                    }
                }
            };

            for (auto* job : batch_jobs) {
                rt::LLMGenerationRequest::Request sub;
                if (!job->messages.empty()) {
                    sub.messages = job->messages;
                } else {
                    rt::Message m;
                    m.role = "user";
                    m.contents.push_back({"text", job->prompt});
                    sub.messages.push_back(m);
                }
                if (!job->image_buffers.empty()) {
                    sub.imageBuffers = job->image_buffers;
                }
                batch_request.requests.push_back(sub);
            }

            rt::LLMGenerationResponse batch_res;
            bool success = ctx->runtime->handleRequest(batch_request, batch_res, ctx->stream);

            for (size_t i = 0; i < batch_jobs.size(); ++i) {
                auto* job = batch_jobs[i];
                job->success = success;
                if (success && i < batch_res.outputTexts.size()) {
                    job->output_text = batch_res.outputTexts[i];
                } else {
                    job->output_text = "";
                }
                job->done = true;
                if (ctx->notify_fd != -1) { long jid = job->id; write(ctx->notify_fd, &jid, sizeof(jid)); }
            }
        }

        // Process sessions (Phase 5)
        {
            std::lock_guard<std::mutex> lock(ctx->sessions_mtx);
            for (auto const& [id, sess] : ctx->sessions) {
                if (sess->slot < 0) {
                    for(int i=0; i<max_batch; ++i) {
                        bool slot_taken = false;
                        for(auto const& [_, other] : ctx->sessions) if(other->slot == i) { slot_taken = true; break; }
                        if(!slot_taken) { sess->slot = i; break; }
                    }
                }
                // Include sessions that have output_ids (ready for decoding) and are not done
                if (sess->slot >= 0 && !sess->is_done && (!sess->input_ids.empty() || !sess->output_ids.empty())) {
                    active_sessions.push_back(sess);
                }
            }
        }

        for (auto* sess : active_sessions) {
            std::lock_guard<std::mutex> sl(sess->mtx);
            
            // Fix active batch size to max_batch for consistent tensor addressing
            runner->getLinearKVCache().setActiveBatchSize(max_batch);
            
            if (!sess->input_ids.empty()) {
                // --- PREFILL STEP ---
                int32_t num_new = (int32_t)sess->input_ids.size();
                
                // 1. Prepare REUSE lengths: Must preserve state for ALL slots
                std::fill(host_reuse_lengths.begin(), host_reuse_lengths.end(), 0);
                
                // Reconstruct current lengths for all sessions (including self)
                {
                    std::lock_guard<std::mutex> lock(ctx->sessions_mtx);
                    for (auto const& [id, other] : ctx->sessions) {
                        if (other->slot >= 0 && other->slot < max_batch) {
                            host_reuse_lengths[other->slot] = other->context_length;
                        }
                    }
                }
                
                // For the current session, the "reuse" length is its *current* context length 
                // (before adding new tokens).
                host_reuse_lengths[sess->slot] = sess->context_length;

                // 2. Prepare Context Lengths (The *new* tokens to add)
                std::fill(host_context_lengths.begin(), host_context_lengths.end(), 0);
                host_context_lengths[sess->slot] = num_new;

                // 3. Prepare Start Indices (Where to write the new tokens)
                std::fill(host_start_indices.begin(), host_start_indices.end(), 0);
                // For others, start index doesn't matter as context_len is 0
                // For self, start index is the current end of buffer
                host_start_indices[sess->slot] = sess->context_length;

                // Sync tensors to device
                (void)step_input_ids.reshape({max_batch, num_new});
                cudaMemsetAsync(step_input_ids.rawPointer(), 0, step_input_ids.getMemoryCapacity(), ctx->stream);
                // Copy new tokens to the correct slot lane
                cudaMemcpyAsync((int32_t*)step_input_ids.rawPointer() + sess->slot * num_new, sess->input_ids.data(), num_new * sizeof(int32_t), cudaMemcpyHostToDevice, ctx->stream);
                
                (void)step_context_lengths.reshape({max_batch});
                memcpy(step_context_lengths.rawPointer(), host_context_lengths.data(), max_batch * sizeof(int32_t));
                
                (void)batch_reuse_lengths.reshape({max_batch});
                memcpy(batch_reuse_lengths.rawPointer(), host_reuse_lengths.data(), max_batch * sizeof(int32_t));
                
                // CRITICAL: This call updates the device-side KV cache lengths.
                // By passing correct reuse lengths for neighbors, we preserve their state.
                runner->getLinearKVCache().resetForNewSequences(batch_reuse_lengths, ctx->stream);

                (void)kv_cache_start_indices.reshape({max_batch});
                cudaMemcpyAsync(kv_cache_start_indices.rawPointer(), host_start_indices.data(), max_batch * sizeof(int32_t), cudaMemcpyHostToDevice, ctx->stream);

                (void)step_inputs_embeds.reshape({max_batch, num_new, config.hiddenSize});
                trt_edgellm::kernel::embeddingLookup(step_input_ids, ctx->runtime->getEmbeddingTable(), step_inputs_embeds, ctx->stream);

                // Explicitly clear dirty KV cache memory if a rollback occurred
                if (sess->needs_kv_rewind) {
                    auto& kv_cache_obj = runner->getLinearKVCache();
                    rt::Tensor kv_cache_buffer = kv_cache_obj.getKVCacheBuffer();
                    
                    // Memory Layout: [numDecoderLayers, maxBatchSize, 2, numKVHeads, maxSequenceLength, headDim]
                    // We want to clear both K and V (the '2' dimension) for this slot and token range.
                    // Since K and V are contiguous at this level, we clear a block of:
                    // 2 * numKVHeads * maxSequenceLength * headDim
                    
                    auto cache_config = kv_cache_obj.getConfig();
                    size_t bytes_per_token = cache_config.headDim * sizeof(__half);
                    size_t tokens_to_clear = sess->rewind_length_to_clear;
                    
                    // Calculate strides
                    size_t slot_stride = cache_config.maxSequenceLength * cache_config.numKVHeads * 2 * bytes_per_token;
                    size_t layer_stride = cache_config.maxBatchSize * slot_stride;

                    for (int l = 0; l < cache_config.numAttentionLayers; ++l) {
                        size_t layer_offset = l * layer_stride;
                        size_t slot_offset = sess->slot * slot_stride;
                        size_t token_offset = sess->rewind_start_index * cache_config.numKVHeads * 2 * bytes_per_token;
                        
                        void* ptr = (char*)kv_cache_buffer.rawPointer() + layer_offset + slot_offset + token_offset;
                        size_t size = tokens_to_clear * cache_config.numKVHeads * 2 * bytes_per_token;
                        
                        cudaMemsetAsync(ptr, 0, size, ctx->stream);
                    }
                    
                    sess->needs_kv_rewind = false;
                    sess->rewind_start_index = 0;
                    sess->rewind_length_to_clear = 0;
                }

                runner->executePrefillStep(step_inputs_embeds, step_context_lengths, {}, step_logits, std::nullopt, ctx->stream);
                
                // Update Session State
                sess->context_length += num_new;
                sess->input_ids.clear();
                
                int32_t next_token = sample_token(step_logits, sess->slot, vocab_size, sess->temperature, sess->top_k, sess->top_p, sess->seed, ctx->stream);
                sess->output_ids.push_back(next_token);
                sess->new_tokens.push(next_token);
                if (next_token == ctx->runtime->getTokenizer().getEosId()) sess->is_done = true;
                if (ctx->notify_fd != -1) { long jid = sess->id; write(ctx->notify_fd, &jid, sizeof(jid)); }

            } else if (!sess->output_ids.empty()) {
                // --- DECODING STEP ---
                int32_t last_token = sess->output_ids.back();
                (void)step_input_ids.reshape({max_batch, 1});
                
                // We must zero out other slots in input_ids to be safe, though context_lengths controls execution
                cudaMemsetAsync(step_input_ids.rawPointer(), 0, step_input_ids.getMemoryCapacity(), ctx->stream);
                cudaMemcpyAsync((int32_t*)step_input_ids.rawPointer() + sess->slot, &last_token, sizeof(int32_t), cudaMemcpyHostToDevice, ctx->stream);
                
                (void)step_inputs_embeds.reshape({max_batch, 1, config.hiddenSize});
                trt_edgellm::kernel::embeddingLookup(step_input_ids, ctx->runtime->getEmbeddingTable(), step_inputs_embeds, ctx->stream);

                // executeVanillaDecodingStep uses the device-side KV cache lengths (which we preserved!)
                runner->executeVanillaDecodingStep(step_inputs_embeds, step_logits, std::nullopt, ctx->stream);
                
                int32_t next_token = sample_token(step_logits, sess->slot, vocab_size, sess->temperature, sess->top_k, sess->top_p, sess->seed, ctx->stream);
                sess->output_ids.push_back(next_token);
                sess->new_tokens.push(next_token);
                
                // Increment context length locally to stay in sync with GPU
                sess->context_length += 1;
                
                if (next_token == ctx->runtime->getTokenizer().getEosId()) sess->is_done = true;
                if (ctx->notify_fd != -1) { long jid = sess->id; write(ctx->notify_fd, &jid, sizeof(jid)); }
            }
        }
    }
}

void parse_perl_messages(pTHX_ SV* request_ref, RequestJob* job) {
    if (SvROK(request_ref) && SvTYPE(SvRV(request_ref)) == SVt_PVAV) {
        AV* av = (AV*)SvRV(request_ref);
        for (int i = 0; i <= av_len(av); i++) {
            SV** svp = av_fetch(av, i, 0);
            if (svp && SvROK(*svp) && SvTYPE(SvRV(*svp)) == SVt_PVHV) {
                HV* hv = (HV*)SvRV(*svp);
                rt::Message msg;
                SV** role_svp = hv_fetch(hv, "role", 4, 0);
                if (role_svp) msg.role = SvPV_nolen(*role_svp);
                SV** parts_svp = hv_fetch(hv, "parts", 5, 0);
                if (parts_svp && SvROK(*parts_svp) && SvTYPE(SvRV(*parts_svp)) == SVt_PVAV) {
                    AV* parts_av = (AV*)SvRV(*parts_svp);
                    for (int j = 0; j <= av_len(parts_av); j++) {
                        SV** psvp = av_fetch(parts_av, j, 0);
                        if (psvp && SvROK(*psvp) && SvTYPE(SvRV(*psvp)) == SVt_PVHV) {
                            HV* phv = (HV*)SvRV(*psvp);
                            SV** text_svp = hv_fetch(phv, "text", 4, 0);
                            if (text_svp) msg.contents.push_back({"text", SvPV_nolen(*text_svp)});
                            SV** data_svp = hv_fetch(phv, "inline_data", 11, 0);
                            if (data_svp && SvROK(*data_svp) && SvTYPE(SvRV(*data_svp)) == SVt_PVHV) {
                                HV* dhv = (HV*)SvRV(*data_svp);
                                SV** bytes_svp = hv_fetch(dhv, "data", 4, 0);
                                if (bytes_svp) {
                                    STRLEN len; unsigned char* bytes = (unsigned char*)SvPV(*bytes_svp, len);
                                    job->image_buffers.push_back(rt::imageUtils::loadImageFromMemory(bytes, len));
                                    msg.contents.push_back({"image", ""});
                                }
                            }
                        }
                    }
                }
                job->messages.push_back(msg);
            }
        }
    } else { job->prompt = SvPV_nolen(request_ref); }
}

C9h_LLM_Runtime bridge_init_runtime(const char* engine_dir, const char* multimodal_dir, bool enable_cuda_graph) {
    printf("[Bridge] bridge_init_runtime start\n");
    std::unordered_map<std::string, std::string> loraMap; cudaStream_t stream; cudaStreamCreate(&stream);
    rt::LLMInferenceRuntime * rt_ptr = nullptr;
    try { 
        printf("[Bridge] Creating LLMInferenceRuntime...\n");
        rt_ptr = new rt::LLMInferenceRuntime(engine_dir, multimodal_dir, loraMap, stream, enable_cuda_graph); 
    }
    catch (std::exception const& e) { 
        printf("[Bridge] FAILED to create LLMInferenceRuntime: %s\n", e.what());
        cudaStreamDestroy(stream); return nullptr; 
    }
    catch (...) { 
        printf("[Bridge] FAILED to create LLMInferenceRuntime (unknown error)\n");
        cudaStreamDestroy(stream); return nullptr; 
    }
    if (enable_cuda_graph) {
        printf("[Bridge] Capturing CUDA Graph...\n");
        rt_ptr->captureDecodingCUDAGraph(stream);
    }
    printf("[Bridge] Creating context and worker thread...\n");
    C9h_LLM_Runtime ctx = new struct C9h_LLM_Runtime_Context();
    ctx->runtime = rt_ptr; ctx->stream = stream;
    int pipe_fds[2];
    if (pipe(pipe_fds) == 0) { 
        ctx->read_fd = pipe_fds[0]; ctx->notify_fd = pipe_fds[1]; 
        ctx->worker = new std::thread(worker_loop, ctx); 
        printf("[Bridge] Worker thread started\n");
    }
    printf("[Bridge] bridge_init_runtime return\n");
    return ctx;
}

void bridge_destroy_runtime(C9h_LLM_Runtime ctx) {
    if (ctx) {
        if (ctx->worker) { 
            { 
                std::unique_lock<std::mutex> lock(ctx->mtx); 
                ctx->stop = true; 
            }
            ctx->cv.notify_all(); 
            if (ctx->worker->joinable()) {
                ctx->worker->join(); 
            }
            delete ctx->worker;
            ctx->worker = nullptr;
        }
        if (ctx->notify_fd != -1) close(ctx->notify_fd); 
        if (ctx->read_fd != -1) close(ctx->read_fd);
        if (ctx->runtime) delete ctx->runtime; 
        if (ctx->stream) cudaStreamDestroy(ctx->stream); 
        
        // Clean up any remaining sessions
        {
            std::lock_guard<std::mutex> lock(ctx->sessions_mtx);
            for (auto& [id, sess] : ctx->sessions) {
                delete sess;
            }
            ctx->sessions.clear();
        }
        
        delete ctx;
    }
}

int bridge_get_notify_fd(C9h_LLM_Runtime ctx) { return ctx ? dup(ctx->read_fd) : -1; }

SV* bridge_generate_async(pTHX_ C9h_LLM_Runtime ctx, SV* request_ref, int max_gen_len, float temperature, float top_p, int top_k, unsigned long seed, bool is_streaming) {
    RequestJob* job = new RequestJob; job->id = (long)job; job->max_gen_len = max_gen_len; job->temperature = temperature;
    job->top_p = top_p; job->top_k = top_k; job->seed = seed; job->done = false; job->is_streaming = is_streaming;
    parse_perl_messages(aTHX_ request_ref, job);
    { std::unique_lock<std::mutex> lock(ctx->mtx); ctx->queue.push(job); }
    ctx->cv.notify_one(); return newSViv(job->id);
}

bool bridge_is_job_done(C9h_LLM_Runtime ctx, long job_id) { RequestJob* job = (RequestJob*)job_id; return (job && job->done); }

AV* bridge_poll_tokens(pTHX_ C9h_LLM_Runtime ctx, long job_id) {
    RequestJob* job = (RequestJob*)job_id; AV* av = newAV();
    if (job) { std::lock_guard<std::mutex> lock(job->token_mtx);
        while (!job->token_queue.empty()) { av_push(av, newSViv(job->token_queue.front())); job->token_queue.pop(); } }
    return av;
}

AV* bridge_tokenize(pTHX_ C9h_LLM_Runtime ctx, SV* request_ref, bool apply_chat_template) {
    RequestJob tmp; parse_perl_messages(aTHX_ request_ref, &tmp);
    rt::LLMGenerationRequest::Request req; 
    if (!tmp.messages.empty()) req.messages = tmp.messages;
    else { rt::Message m; m.role = "user"; m.contents.push_back({"text", tmp.prompt}); req.messages.push_back(m); }
    std::vector<int32_t> ids = ctx->runtime->tokenize(req, apply_chat_template);
    AV* av = newAV(); for (int32_t id : ids) av_push(av, newSViv(id)); return av;
}

SV* bridge_decode(pTHX_ C9h_LLM_Runtime ctx, SV* token_ids_ref) {
    if (!SvROK(token_ids_ref) || SvTYPE(SvRV(token_ids_ref)) != SVt_PVAV) return &PL_sv_undef;
    AV* av = (AV*)SvRV(token_ids_ref); std::vector<int32_t> ids;
    for (int i = 0; i <= av_len(av); i++) {
        SV** svp = av_fetch(av, i, 0);
        if (svp) ids.push_back(SvIV(*svp));
    }
    std::string text = ctx->runtime->decode(ids, true); return newSVpvn(text.c_str(), text.length());
}

SV* bridge_collect_job(pTHX_ C9h_LLM_Runtime ctx, long job_id) {
    RequestJob* job = (RequestJob*)job_id;
    if (job && job->done) { SV* res = job->success ? newSVpvn(job->output_text.c_str(), job->output_text.length()) : &PL_sv_undef; delete job; return res; }
    return &PL_sv_undef;
}

AV* bridge_get_embedding(pTHX_ C9h_LLM_Runtime ctx, const char* text) {
    auto* runner = ctx->runtime->getEngineRunner();
    auto const& config = runner->getEngineConfig();
    int32_t hidden_size = config.hiddenSize;

    // 1. Tokenize
    rt::LLMGenerationRequest::Request req;
    rt::Message m; m.role = "user"; m.contents.push_back({"text", text});
    req.messages.push_back(m);
    std::vector<int32_t> ids = ctx->runtime->tokenize(req, true);
    int32_t num_tokens = (int32_t)ids.size();
    if (num_tokens == 0) return newAV();

    // 2. Prepare Tensors
    rt::Tensor input_ids({1, num_tokens}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor inputs_embeds({1, num_tokens, hidden_size}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor context_lengths({1}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32);
    
    // Query engine for expected output shapes and types
    nvinfer1::Dims logits_dims = runner->getTRTEngine()->getTensorShape(trt_edgellm::binding_names::kLogits);
    nvinfer1::DataType logits_type = runner->getTRTEngine()->getTensorDataType(trt_edgellm::binding_names::kLogits);
    
    const char* hidden_name = trt_edgellm::binding_names::kOutputHiddenStates;
    nvinfer1::Dims hidden_dims = {0, {0}};
    nvinfer1::DataType hidden_type = nvinfer1::DataType::kHALF;
    if (runner->getTRTEngine()->getTensorIOMode(hidden_name) != nvinfer1::TensorIOMode::kNONE) {
        hidden_dims = runner->getTRTEngine()->getTensorShape(hidden_name);
        hidden_type = runner->getTRTEngine()->getTensorDataType(hidden_name);
    }

    // Logits in prefill step (with default last-token selection) is 2D: [batch, vocab]
    // UNLESS the engine explicitly wants 3D.
    rt::Tensor logits;
    if (logits_dims.nbDims == 3) {
        logits = rt::Tensor({1, 1, config.outputVocabSize}, rt::DeviceType::kGPU, logits_type);
    } else {
        logits = rt::Tensor({1, config.outputVocabSize}, rt::DeviceType::kGPU, logits_type);
    }
    
    // Hidden States might be 3D [batch, seq, dim] in fragmented/Blackwell engines
    // Standard prefill usually only selects the last token IF last_token_ids is used.
    // However, the engine might still produce the full sequence output.
    rt::Tensor hidden_states;
    if (hidden_dims.nbDims == 3) {
        hidden_states = rt::Tensor({1, num_tokens, hidden_size}, rt::DeviceType::kGPU, hidden_type);
    } else {
        hidden_states = rt::Tensor({1, hidden_size}, rt::DeviceType::kGPU, hidden_type);
    }

    cudaMemcpyAsync(input_ids.rawPointer(), ids.data(), num_tokens * sizeof(int32_t), cudaMemcpyHostToDevice, ctx->stream);
    *(int32_t*)context_lengths.rawPointer() = num_tokens;

    trt_edgellm::kernel::embeddingLookup(input_ids, ctx->runtime->getEmbeddingTable(), inputs_embeds, ctx->stream);

    // 3. Execute Prefill (to get hidden states)
    rt::LinearKVCache& kv_cache = runner->getLinearKVCache();
    kv_cache.setActiveBatchSize(1);
    rt::Tensor reuse_lengths({1}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32);
    *(int32_t*)reuse_lengths.rawPointer() = 0;
    kv_cache.resetForNewSequences(reuse_lengths, ctx->stream);

    bool ok = runner->executePrefillStep(inputs_embeds, context_lengths, {}, logits, std::ref(hidden_states), ctx->stream);
    if (!ok) return newAV(); 

    // 4. Extract Last Token Hidden State (The "Embedding")
    std::vector<__half> host_hidden(hidden_size);
    // Try both offsets (0 and num_tokens-1) to handle different engine behaviors
    void* last_token_ptr = (char*)hidden_states.rawPointer() + (num_tokens - 1) * hidden_size * sizeof(__half);
    if (hidden_states.getShape().getNumDims() == 2 || hidden_states.getShape()[1] == 1) {
        last_token_ptr = hidden_states.rawPointer();
    }
    
    cudaMemcpyAsync(host_hidden.data(), last_token_ptr, hidden_size * sizeof(__half), cudaMemcpyDeviceToHost, ctx->stream);
    cudaStreamSynchronize(ctx->stream);

    // If all zeros at the end, try the beginning (fallback for engines that only produce 1 token hidden state)
    bool all_zeros = true;
    for (int i=0; i<std::min(10, hidden_size); ++i) if (host_hidden[i] != (__half)0) all_zeros = false;
    if (all_zeros && num_tokens > 1 && last_token_ptr != hidden_states.rawPointer()) {
        last_token_ptr = hidden_states.rawPointer();
        cudaMemcpyAsync(host_hidden.data(), last_token_ptr, hidden_size * sizeof(__half), cudaMemcpyDeviceToHost, ctx->stream);
        cudaStreamSynchronize(ctx->stream);
    }

    AV* av = newAV();
    for (int i = 0; i < hidden_size; ++i) {
        av_push(av, newSVnv((double)__half2float(host_hidden[i])));
    }
    return av;
}

void bridge_release_spare_memory(C9h_LLM_Runtime ctx) {
    if (ctx) { cudaDeviceSynchronize(); int dev = 0; if (cudaGetDevice(&dev) == cudaSuccess) {
        cudaMemPool_t p; if (cudaDeviceGetDefaultMemPool(&p, dev) == cudaSuccess) cudaMemPoolTrimTo(p, 0); }
        malloc_trim(0); }
}

SV* bridge_generate(pTHX_ C9h_LLM_Runtime ctx, const char* prompt, int max_gen_len, float temperature, float top_p, int top_k, unsigned long seed) {
    rt::LLMGenerationRequest req; rt::LLMGenerationRequest::Request sub; rt::Message m; m.role = "user";
    m.contents.push_back({"text", prompt}); sub.messages.push_back(m); req.requests.push_back(sub);
    req.maxGenerateLength = max_gen_len; req.temperature = temperature; req.topP = top_p; req.topK = top_k; req.randomSeed = seed;
    rt::LLMGenerationResponse res; if (ctx->runtime->handleRequest(req, res, ctx->stream))
        if (!res.outputTexts.empty()) return newSVpvn(res.outputTexts[0].c_str(), res.outputTexts[0].length());
    return &PL_sv_undef;
}

C9h_LLM_Session bridge_create_session(C9h_LLM_Runtime ctx) {
    LLMSession* s = new LLMSession(); 
    s->id = (long)s;
    s->temperature = 0.8f;
    s->top_p = 0.95f;
    s->top_k = 40;
    { std::lock_guard<std::mutex> l(ctx->sessions_mtx); ctx->sessions[s->id] = s; } return s;
}

void bridge_destroy_session(C9h_LLM_Runtime ctx, C9h_LLM_Session s) {
    if (ctx && s) { std::lock_guard<std::mutex> l(ctx->sessions_mtx); ctx->sessions.erase(s->id); delete s; }
}

void bridge_session_push_content(pTHX_ C9h_LLM_Runtime ctx, C9h_LLM_Session s, SV* c) {
    rt::LLMGenerationRequest::Request req; rt::Message m; m.role = "user"; m.contents.push_back({"text", SvPV_nolen(c)});
    req.messages.push_back(m); std::vector<int32_t> ids = ctx->runtime->tokenize(req, false);
    std::lock_guard<std::mutex> l(s->mtx); 
    
    // INTERRUPTION LOGIC:
    // If the model has generated speculative tokens (output_ids) since the last input,
    // we must rollback the context length to overwrite them with the new input.
    if (!s->output_ids.empty()) {
        s->rewind_start_index = s->context_length - (int32_t)s->output_ids.size();
        s->rewind_length_to_clear = (int32_t)s->output_ids.size();
        s->context_length = s->rewind_start_index;
        
        s->output_ids.clear();
        s->new_tokens = std::queue<int32_t>(); // Clear pending tokens from queue
        s->needs_kv_rewind = true;
    }

    s->input_ids.insert(s->input_ids.end(), ids.begin(), ids.end()); s->is_done = false;
}

AV* bridge_session_poll_tokens(pTHX_ C9h_LLM_Session s) {
    AV* av = newAV(); if (s) { std::lock_guard<std::mutex> l(s->mtx);
        while (!s->new_tokens.empty()) { av_push(av, newSViv(s->new_tokens.front())); s->new_tokens.pop(); } } return av;
}

bool bridge_session_is_done(C9h_LLM_Session s) { return s ? s->is_done : true; }

bool bridge_build(const char* o, const char* e, struct bridge_builder_config* bc) {
    if (!bc) return false;
    printf("[Bridge] bridge_build entered. onnx=%s, engine=%s\n", o, e);
    printf("[Bridge] config inputs: mi=%d, mk=%d, mb=%d, v=%d, w=%ld, mt=%d, ie=%d\n", 
             bc->max_input_len, bc->max_kv_cache_capacity, bc->max_batch_size, (int)bc->is_vlm, 
             bc->weight_streaming_budget, bc->max_image_tokens, (int)bc->is_eagle);

    trt_edgellm::builder::LLMBuilderConfig c{}; 
    c.maxInputLen = bc->max_input_len;
    c.maxKVCacheCapacity = bc->max_kv_cache_capacity;
    c.maxBatchSize = bc->max_batch_size;
    c.isVlm = bc->is_vlm; 
    c.weightStreamingBudget = bc->weight_streaming_budget; 
    c.maxImageTokens = bc->max_image_tokens;
    c.minImageTokens = bc->max_image_tokens;
    c.eagleBase = bc->is_eagle;
    
    printf("[Bridge] Initializing LLMBuilder object...\n");
    try {
        builder::LLMBuilder b(o, e, c); 
        printf("[Bridge] Starting builder.build()...\n");
        bool result = b.build();
        printf("[Bridge] builder.build() returned %d\n", (int)result);
        return result;
    } catch (const std::exception& ex) {
        printf("[Bridge] FATAL: Exception during build: %s\n", ex.what());
        return false;
    } catch (...) {
        printf("[Bridge] FATAL: Unknown exception during build\n");
        return false;
    }
}


const char* bridge_get_runtime_version() {
    return trt_edgellm::version::kRUNTIME_VERSION;
}

static void* g_plugin_handle = nullptr;
static void* g_trt_plugin_handle = nullptr;
static bool g_bridge_plugins_initialized = false;

bool bridge_init_plugins(const char* plugin_path) {
    if (!plugin_path) return false;
    
    // Load standard TRT plugins first
    if (!g_trt_plugin_handle) {
        g_trt_plugin_handle = dlopen("libnvinfer_plugin.so.10", RTLD_NOW | RTLD_GLOBAL);
        if (g_trt_plugin_handle) {
            typedef bool (*InitFunc)(void*, const char*);
            auto initTRT = reinterpret_cast<InitFunc>(dlsym(g_trt_plugin_handle, "initLibNvInferPlugins"));
            if (initTRT) initTRT(nullptr, "");
        }
    }

    // Load the specified plugin library
    void* handle = dlopen(plugin_path, RTLD_NOW | RTLD_GLOBAL);
    if (handle) {
        // Assume this might be the primary g_plugin_handle for legacy reasons
        if (!g_plugin_handle) g_plugin_handle = handle;
        
        typedef bool (*InitFunc)(void*, const char*);
        // Try to initialize as EdgeLLM plugins
        auto initEdge = reinterpret_cast<InitFunc>(dlsym(handle, "initEdgellmPlugins"));
        if (initEdge) {
            // Initialize with empty namespace for standard global compatibility
            initEdge(static_cast<nvinfer1::ILogger*>(&trt_edgellm::gLogger), "");
            trt_edgellm::gLogger.setLevel(trt_edgellm::gLogger.getLevel());
        }
        
        // Also try to initialize as TRT-LLM plugins if it's the TRT-LLM library
        auto initTRTLLM = reinterpret_cast<InitFunc>(dlsym(handle, "initTrtLlmPlugins"));
        if (initTRTLLM) {
            // TRT-LLM plugins usually register in "" namespace
            initTRTLLM(static_cast<nvinfer1::ILogger*>(&trt_edgellm::gLogger), "");
        }
    } else {
        fprintf(stderr, "[Bridge] Failed to dlopen plugin: %s - %s\n", plugin_path, dlerror());
        return false;
    }
    
    return true;
}
