# Sakshi's Llama3.1-8B Experiment -- Final Checkpoint (Week 6 Mon)

## Official checkpoint
- Location: `models/sakshi_llama31_8b_rank4/checkpoint-9` (gitignored,
  local only -- adapter files, not committed)
- Base model: `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`
- LoRA rank: 4, alpha 16, dropout 0.05
- Training: 1 epoch, 9 steps, batch 1, grad accum 16, seq_len 512
- Save method: `save_method="lora"` (adapter-only, no merge)

## Metrics
- Train loss: 1.117
- Eval loss: 1.022
- pass@1 (val set): 31.2% -- measured via `evaluation/live_checkpoint_eval.py`

## Status
Only 1 epoch was run (each step took ~5 min on 6GB VRAM with an 8B model),
so this is the only checkpoint that exists for this experiment -- no
epoch-over-epoch comparison to pick a "best" one from. This checkpoint
(`checkpoint-9`) is the final/official one for the Llama3.1-8B secondary
experiment track.

See `docs/base_model_comparison.md` for comparison against Yeshita's/Faiza's
Qwen2.5-based runs.

## Next steps (Week 6 Tue-Thu)
This checkpoint will be used as input for `inference/export_gguf.py`
(GGUF conversion + quantization) once that script is built, to test
CUDA and Vulkan backend inference.