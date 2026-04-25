#include "qwen3DeltaAttention.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>

using namespace trt_edgellm::kernels;

// --- Helpers ---
__host__ half f2h(float f) { return __float2half(f); }
__host__ float h2f(half h) { return __half2float(h); }

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float silu(float x) { return x * sigmoid(x); }

// --- CPU Reference Implementation ---
void cpu_delta_attention(Qwen3DeltaAttentionParams const& params, 
                         const std::vector<float>& q, 
                         const std::vector<float>& k, 
                         const std::vector<float>& v, 
                         const std::vector<float>& g, 
                         const std::vector<float>& beta,
                         const std::vector<float>& z,
                         const std::vector<float>& norm_weight,
                         std::vector<float>& state,
                         std::vector<float>& output) {
    
    int D = params.head_size;
    int H = params.num_q_heads;
    int KV_H = params.num_kv_heads;
    int G_ratio = H / KV_H;

    for (int b = 0; b < params.batch_size; ++b) {
        for (int t = 0; t < params.seq_len; ++t) {
            for (int h = 0; h < H; ++h) {
                int kv_h = h / G_ratio;
                
                float gate = g[((b * params.seq_len + t) * H + h)];
                float b_val = beta[((b * params.seq_len + t) * H + h)];
                float exp_g = std::exp(gate);
                float sig_b = sigmoid(b_val);

                // Current q, k, v vectors
                std::vector<float> cur_q(D), cur_k(D), cur_v(D);
                for(int d=0; d<D; ++d) {
                    cur_q[d] = q[((b * params.seq_len + t) * H + h) * D + d];
                    cur_k[d] = k[((b * params.seq_len + t) * KV_H + kv_h) * D + d];
                    cur_v[d] = v[((b * params.seq_len + t) * KV_H + kv_h) * D + d];
                }
                
                // L2 Norm for Q and K (matching kernel parity)
                auto l2_norm = [](std::vector<float>& vec) {
                    float sum = 0.0f;
                    for(float val : vec) sum += val * val;
                    float inv_norm = 1.0f / std::sqrt(sum + 1e-6f);
                    for(float& val : vec) val *= inv_norm;
                };
                l2_norm(cur_q);
                l2_norm(cur_k);

                // State pointer for this batch/head
                float* cur_state = &state[(b * H + h) * D * D];

                // 1. Gated Delta Update Rule:
                // a. h = h * exp(g)
                for(int i=0; i<D; ++i) {
                    for(int j=0; j<D; ++j) {
                        cur_state[i * D + j] *= exp_g;
                    }
                }

                // b. v_update = (v - h * k) * sigmoid(beta)
                std::vector<float> dot_h_k(D, 0.0f);
                for(int i=0; i<D; ++i) {
                    for(int j=0; j<D; ++j) {
                        dot_h_k[i] += cur_state[i * D + j] * cur_k[j];
                    }
                }

                std::vector<float> v_update(D);
                for(int i=0; i<D; ++i) v_update[i] = (cur_v[i] - dot_h_k[i]) * sig_b;

                // c. h = h + k_outer_v
                for(int i=0; i<D; ++i) {
                    for(int j=0; j<D; ++j) {
                        cur_state[i * D + j] += cur_k[i] * v_update[j];
                    }
                }

                // 2. Output: o = h * q
                std::vector<float> o_vec(D, 0.0f);
                float o_sq_sum = 0.0f;
                for(int i=0; i<D; ++i) {
                    float o_val = 0.0f;
                    for(int j=0; j<D; ++j) {
                        o_val += cur_state[i * D + j] * cur_q[j];
                    }
                    o_vec[i] = o_val;
                    o_sq_sum += o_val * o_val;
                }
                
                // 3. Post-processing: RMSNormGated
                float rms = 1.0f / std::sqrt(o_sq_sum / D + params.eps);
                for(int i=0; i<D; ++i) {
                    float val = o_vec[i] * rms * norm_weight[h * D + i];
                    float z_val = z[((b * params.seq_len + t) * H + h) * D + i];
                    output[((b * params.seq_len + t) * H + h) * D + i] = val * silu(z_val) * params.scale;
                }
            }
        }
    }
}

bool run_test_case(const std::string& name, int B, int T, int H, int KV_H, int D) {
    std::cout << "\n[TEST] " << name << " (B=" << B << ", T=" << T << ", H=" << H << ", KV_H=" << KV_H << ", D=" << D << ")" << std::endl;

    size_t q_size = B * T * H * D;
    size_t kv_size = B * T * KV_H * D;
    size_t g_size = B * T * H;
    size_t state_size = B * H * D * D;
    size_t nw_size = H * D;

    std::vector<float> h_q(q_size), h_k(kv_size), h_v(kv_size), h_g(g_size), h_beta(g_size), h_z(q_size);
    std::vector<float> h_state(state_size, 0.0f), h_out(q_size, 0.0f), h_nw(nw_size, 1.0f);

    // Random-ish init
    for(int i=0; i<q_size; ++i) { h_q[i] = (float)(i % 7) / 10.0f; h_z[i] = (float)(i % 3) / 10.0f; }
    for(int i=0; i<kv_size; ++i) {
        h_k[i] = (float)(i % 5) / 10.0f;
        h_v[i] = (float)(i % 11) / 10.0f;
    }
    for(int i=0; i<g_size; ++i) {
        h_g[i] = -0.1f; // Slight decay
        h_beta[i] = 0.5f; 
    }
    for(int i=0; i<nw_size; ++i) h_nw[i] = 1.0f + (float)(i % 10) / 100.0f;

    // CUDA buffers
    half *d_q, *d_k, *d_v, *d_g, *d_beta, *d_z, *d_nw, *d_out;
    float *d_state;
    cudaMalloc(&d_q, q_size * sizeof(half));
    cudaMalloc(&d_k, kv_size * sizeof(half));
    cudaMalloc(&d_v, kv_size * sizeof(half));
    cudaMalloc(&d_g, g_size * sizeof(half));
    cudaMalloc(&d_beta, g_size * sizeof(half));
    cudaMalloc(&d_z, q_size * sizeof(half));
    cudaMalloc(&d_nw, nw_size * sizeof(half));
    cudaMalloc(&d_state, state_size * sizeof(float));
    cudaMalloc(&d_out, q_size * sizeof(half));

    // Convert and copy
    std::vector<half> tmp_q(q_size), tmp_k(kv_size), tmp_v(kv_size), tmp_g(g_size), tmp_beta(g_size), tmp_z(q_size), tmp_nw(nw_size);
    std::vector<float> tmp_state(state_size, 0.0f);
    for(int i=0; i<q_size; ++i) { 
        tmp_q[i] = f2h(h_q[i]); h_q[i] = h2f(tmp_q[i]); 
        tmp_z[i] = f2h(h_z[i]); h_z[i] = h2f(tmp_z[i]); 
    }
    for(int i=0; i<kv_size; ++i) { 
        tmp_k[i] = f2h(h_k[i]); h_k[i] = h2f(tmp_k[i]); 
        tmp_v[i] = f2h(h_v[i]); h_v[i] = h2f(tmp_v[i]); 
    }
    for(int i=0; i<g_size; ++i) { 
        tmp_g[i] = f2h(h_g[i]); h_g[i] = h2f(tmp_g[i]); 
        tmp_beta[i] = f2h(h_beta[i]); h_beta[i] = h2f(tmp_beta[i]); 
    }
    for(int i=0; i<nw_size; ++i) { tmp_nw[i] = f2h(h_nw[i]); h_nw[i] = h2f(tmp_nw[i]); }

    cudaMemcpy(d_q, tmp_q.data(), q_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, tmp_k.data(), kv_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, tmp_v.data(), kv_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, tmp_g.data(), g_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, tmp_beta.data(), g_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, tmp_z.data(), q_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nw, tmp_nw.data(), nw_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_state, tmp_state.data(), state_size * sizeof(float), cudaMemcpyHostToDevice);

    Qwen3DeltaAttentionParams params;
    params.q = d_q; params.k = d_k; params.v = d_v; params.g = d_g; params.beta = d_beta;
    params.z = d_z; params.norm_weight = d_nw;
    params.state = d_state; params.output = d_out;
    params.batch_size = B; params.seq_len = T;
    params.num_q_heads = H; params.num_kv_heads = KV_H;
    params.head_size = D; params.scale = 1.0f / std::sqrt((float)D);
    params.eps = 1e-6f;
    params.is_prefill = (T > 1);

    // Run GPU
    invokeQwen3DeltaAttention(params, 0);
    cudaDeviceSynchronize();

    // Run CPU
    std::vector<float> cpu_state(state_size, 0.0f), cpu_out(q_size, 0.0f);
    cpu_delta_attention(params, h_q, h_k, h_v, h_g, h_beta, h_z, h_nw, cpu_state, cpu_out);

    // Verify Output
    std::vector<half> gpu_out_h(q_size);
    cudaMemcpy(gpu_out_h.data(), d_out, q_size * sizeof(half), cudaMemcpyDeviceToHost);
    
    float max_err = 0.0f;
    for(int i=0; i<q_size; ++i) {
        max_err = std::max(max_err, std::abs(h2f(gpu_out_h[i]) - cpu_out[i]));
    }
    std::cout << "  Max Output Error: " << max_err << std::endl;

    // Verify State
    std::vector<float> gpu_state_h(state_size);
    cudaMemcpy(gpu_state_h.data(), d_state, state_size * sizeof(float), cudaMemcpyDeviceToHost);
    float state_err = 0.0f;
    for(int i=0; i<state_size; ++i) {
        state_err = std::max(state_err, std::abs(gpu_state_h[i] - cpu_state[i]));
    }
    std::cout << "  Max State Error: " << state_err << std::endl;

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_g); cudaFree(d_beta); cudaFree(d_z); cudaFree(d_nw); cudaFree(d_state); cudaFree(d_out);

    if (max_err > 0.05f || state_err > 0.05f) {
        std::cout << "  [RESULT] FAILED" << std::endl;
        return false;
    }
    std::cout << "  [RESULT] PASSED" << std::endl;
    return true;
}

int main() {
    bool all_ok = true;
    
    // 1. Basic Single Token
    all_ok &= run_test_case("Single Token", 1, 1, 1, 1, 128);
    
    // 2. Prefill Sequence (Exercises loop)
    all_ok &= run_test_case("Prefill Sequence", 1, 8, 1, 1, 128);
    
    // 3. GQA (Grouped Query Attention) 0.8B Config
    all_ok &= run_test_case("GQA 0.8B Hybrid Config", 1, 1, 16, 16, 128);
    all_ok &= run_test_case("GQA 0.8B Standard Config (Splitting)", 1, 1, 16, 4, 128);

    // 4. Batching
    all_ok &= run_test_case("Batched Inference", 2, 1, 8, 8, 128);

    // 5. Extreme GQA (Many Q, few KV)
    all_ok &= run_test_case("Extreme GQA (32Q, 1KV)", 1, 1, 32, 1, 128);

    // 6. Long Sequence (Prefill)
    all_ok &= run_test_case("Long Sequence (T=128)", 1, 128, 16, 16, 128);

    if (all_ok) {
        std::cout << "\nOVERALL STANDALONE TEST PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "\nOVERALL STANDALONE TEST FAILED!" << std::endl;
        return 1;
    }
}
