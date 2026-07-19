# CUDA vs Vulkan Inference Comparison -- Final (Week 6 Fri)

## Setup
Same quantized checkpoint across all runs:
`models/gguf_export/model-Q4_K_M.gguf` (Llama3.1-8B rank-4 QLoRA, Q4_K_M
quantization, 4.7GB). Same prompt: "Write a Python function that reverses
a string.", same generation length (-n 128), full layer offload (-ngl 99)
unless noted.

## Results

| Backend | Device | Prompt (t/s) | Generation (t/s) |
|---|---|---|---|
| CUDA | RTX 4050 (discrete) | 536.5 | 31.1 |
| Vulkan | RTX 4050 (discrete, `--device Vulkan1`) | 352.0 | 27.4 |
| Vulkan | Radeon 740M (iGPU, `--device Vulkan0`, full offload) | -- | FAILED (OOM) |
| Vulkan | Radeon 740M (iGPU, `--device Vulkan0`, partial offload -ngl 20) | 7.3 | 5.2 |

## Takeaways
- **CUDA is faster than Vulkan on the same GPU** for this model: ~52% faster
  prompt processing (536.5 vs 352.0 t/s), ~13% faster generation (31.1 vs
  27.4 t/s). Expected -- CUDA is NVIDIA's native, more optimized backend;
  Vulkan is the portable cross-vendor path, useful specifically because it
  also runs on the AMD iGPU (which CUDA cannot).
- **The iGPU cannot fully offload this 8B model** despite reporting enough
  "free" shared memory on paper -- real allocation failed at full offload.
  Partial offload (20 of 32 layers) works but is far slower (5.2 t/s vs
  27-31 t/s on the discrete GPU) -- roughly 5-6x slower.
- **Practical implication:** for this model size, the discrete RTX 4050 via
  CUDA is clearly the best inference path on this machine. The Vulkan/iGPU
  path is a fallback for machines without a capable discrete GPU, not a
  performance win here -- but it does prove the dual-backend build from
  Week 1 works end-to-end across the whole pipeline (train -> export ->
  quantize -> infer on either backend).

## Related docs
- `docs/gguf_cuda_test.md` -- initial CUDA test detail
- `docs/gguf_vulkan_test.md` -- initial Vulkan test detail (default device,
  OOM investigation)