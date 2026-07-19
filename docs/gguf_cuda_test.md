# GGUF Export & CUDA Inference Test -- Week 6 Wed

## Export pipeline
- Checkpoint: `models/sakshi_llama31_8b_rank4/checkpoint-9`
- Merged to 16-bit HF format via Unsloth (`save_method="merged_16bit"`)
- Converted to F16 GGUF via llama.cpp's `convert_hf_to_gguf.py` (16.1GB)
- Quantized to Q4_K_M via `llama-quantize.exe` (4.7GB, down from 15.3GB
  -- 4.89 bits per weight)

## CUDA inference test
Ran via llama.cpp's CUDA build (`build-cuda/bin/Release/llama-cli.exe`),
all 32 layers offloaded to GPU (`-ngl 99`):

Prompt: "Write a Python function that reverses a string."

Output: correct, well-formatted `reverse_string()` function using slice
notation, with docstring and example usage -- coherent and usable.

**Performance:**
- Prompt processing: 536.5 tokens/sec
- Generation: 31.1 tokens/sec

## Notes
- One-time cost: exporting required downloading the full 16.1GB base model
  into the HuggingFace cache (Llama3.1-8B-Instruct, not previously cached
  locally in full precision) -- significant disk usage, flagged in the
  laptop cleanup plan for end of project.
- This CUDA baseline (31.1 tok/s generation) will be compared against the
  Vulkan backend (iGPU) tomorrow (Week 6 Thu task).