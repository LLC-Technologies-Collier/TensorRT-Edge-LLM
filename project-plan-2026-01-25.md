# Project Plan: Porting Gemma 3 27B to TensorRT-Edge-LLM (Thor)
**Date:** Sunday, January 25, 2026

## Overview
This plan outlines the strategy to migrate the **Gemma 3 27B** model from a manually compiled TensorRT-LLM environment to the specialized **TensorRT-Edge-LLM** framework. The primary goal is to bypass the current "Duplicate GPU" NCCL blocker by utilizing **NVFP4 quantization**, allowing the 27B model to run on a single logical GPU instance on the NVIDIA Jetson Thor.

## Success Criteria
*   **Functional:** Gemma 3 27B model generates coherent text on Jetson Thor using `TensorRT-Edge-LLM`.
*   **Architecture:** Model runs as a single engine (TP=1), eliminating NCCL/MIG dependencies.
*   **Performance:**
    *   **Memory:** Total runtime memory usage < 16GB (fitting within single Thor GPU/instance limits).
    *   **Throughput:** Target > 10 Tokens Per Second (TPS) for interactive use.
*   **Quality:** NVFP4 quantization accuracy degradation < 5% (perplexity/eval metric) compared to BF16 baseline.

---

## Timeline & Milestones
| Phase | Duration | Start Date | End Date | Key Deliverable |
| :--- | :--- | :--- | :--- | :--- |
| **1. Bootstrap** | 1 Week | Jan 26, 2026 | Jan 31, 2026 | Functional BF16 Engine running on Thor (TP=1). |
| **2. Optimization** | 2 Weeks | Feb 01, 2026 | Feb 14, 2026 | Optimized NVFP4 Engine meeting memory (<16GB) & TPS targets. |
| **3. Integration** | 1 Week | Feb 15, 2026 | Feb 21, 2026 | Merged PR upstream to NVIDIA/TensorRT-Edge-LLM. |

---

## Roles & Responsibilities
*   **Lead Developer (User/Agent):** Implementation of model wrapper, export pipeline modifications, engine compilation, and on-device testing.
*   **Validation Engineer (User):** Execution of on-device benchmarks, hardware monitoring (tegrastats), and subjective quality checks.
*   **Reviewers (NVIDIA Team):** Code review for PR submission and guidance on Blackwell-specific kernel usage.

---

## Tracking Mechanism
*   **Primary Tracker:** GitHub Issues (or internal Buganizer equivalent).
*   **Structure:**
    *   **Parent Issue:** "Port Gemma 3 27B to TensorRT-Edge-LLM (Thor)"
    *   **Phase 1 Tracking:** `[Phase 1] BF16 Architecture Enablement`
    *   **Phase 2 Tracking:** `[Phase 2] NVFP4 Quantization & Optimization`
    *   **Phase 3 Tracking:** `[Phase 3] Cleanup & Upstream PR`

---

## Phase 1: Bootstrap (BF16 Architecture Support)
**Objective:** Confirm the model architecture is compatible with the Edge runtime using standard precision.

1.  **Define Gemma 3 Model Wrapper:**
    *   Create `tensorrt_edgellm/llm_models/models/gemma_model.py`.
    *   Implement `EdgeLLMModelForGemma` by adapting `EdgeLLMModel` to handle Gemma-specific normalization (RMSNorm with offset 1.0) and activation patterns (GeGLU).
    *   Register the model in the library's `__init__.py`.
    *   **Testing:** Add unit tests to `unittests/` to verify layer construction and forward pass shapes.

2.  **Export Pipeline Adaptation:**
    *   Modify `tensorrt_edgellm/onnx_export/llm_export.py` to recognize `gemma` model types.
    *   Ensure the export process correctly handles the consolidated `rank0.safetensors` weight file.

3.  **Engine Build & Verification:**
    *   Build a TP=1 engine in BF16/FP16 precision using the C++ builder.
    *   **Verification:** Run end-to-end inference on the Thor device.
    *   **Exit Criteria:** Successful generation of "Hello, world!" or similar prompt without crash.

---

## Phase 2: Optimization (NVFP4 Enablement)
**Objective:** Reduce the 54GB footprint to ~13.5GB to fit on a single Thor GPU rank.

1.  **Baseline Establishment:**
    *   Measure perplexity of the Phase 1 BF16 model on `wikitext-2-raw-v1`.

2.  **Apply NVFP4 Quantization:**
    *   Utilize `tensorrt_edgellm.quantization.llm_quantization` to apply `NVFP4_DEFAULT_CFG`.
    *   Validate the insertion of `TRT_FP4DynamicQuantize` ONNX nodes during the Python export phase.

3.  **Single-Rank Engine Build:**
    *   Compile the NVFP4 engine on the Thor platform.
    *   **Result:** This eliminates the need for Tensor Parallelism (TP=2), NCCL, and MIG virtualization.

4.  **Performance & Quality Tuning:**
    *   **Benchmarks:** Run `llm_inference` with `--benchmark` flag to capture TPS.
    *   **Quality:** Re-run `wikitext-2-raw-v1` perplexity check.
    *   **Exit Criteria:** Memory < 16GB, TPS > 10, Accuracy Delta < 5%.

---

## Phase 3: Integration & Contribution
**Objective:** Finalize the port and share the implementation.

1.  **Code Cleanup:**
    *   Align the new model implementation with NVIDIA's `CODING_GUIDELINES.md`.
    *   Ensure all new code is covered by unit tests in `unittests/`.

2.  **Upstream Contribution:**
    *   Prepare a pull request for the NVIDIA `TensorRT-Edge-LLM` repository.
    *   Assist the NVIDIA team in validating the Gemma 3 FP4 support for official release.

---

## Detailed Test Plan
| Test ID | Type | Description | Dataset/Prompt | Pass Criteria |
| :--- | :--- | :--- | :--- | :--- |
| **T1.1** | Unit | Validate `GemmaModel` layer construction. | Random Tensor Input | Output shape matches expected `(B, S, H)`. |
| **T1.2** | Integration | Export to ONNX. | Dummy Weights | Valid `model.onnx` generated. |
| **T2.1** | Quality | BF16 Baseline Perplexity. | `wikitext-2-raw-v1` | PPL < Threshold (TBD). |
| **T2.2** | Quality | NVFP4 Perplexity. | `wikitext-2-raw-v1` | PPL < Baseline * 1.05. |
| **T2.3** | Perf | Throughput Benchmark. | Input: 128 tok, Output: 128 tok | > 10 tokens/sec. |
| **T2.4** | System | Memory Usage Check. | `tegrastats` | VRAM Usage < 16GB. |

---

## Risk Assessment & Mitigation
| Risk | Probability | Impact | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **NVFP4 Accuracy Loss** | Medium | High | Fallback to INT4 AWQ or FP8 if supported; investigate layer-specific quantization sensitivity. |
| **Unsupported Kernels** | Low | High | Gemma 3 uses GeGLU; ensure Blackwell `sm_110` kernels support this activation or fall back to generic CUDA implementation. |
| **Driver/Runtime Issues** | Medium | Medium | Maintain the working "MIG workaround" environment as a backup for development if the single-rank approach stalls. |

## Alternatives Considered
*   **Tensor Parallelism (TP=2) with MIG:** Technically feasible but requires complex NCCL setup and virtualization (MIG), introducing the "Duplicate GPU" blocker.
*   **INT4 AWQ:** A viable alternative for memory reduction, but typically slower than native NVFP4 on Blackwell hardware.

## Technical Constraints
*   **Hardware:** NVIDIA Jetson Thor (sm_110).
*   **Software:** JetPack 7.1, CUDA 13.0.