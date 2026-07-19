## Ablation Run (Yeshita, rank 8, seq_len 1024) — Final

**Config:**
- Base model: Qwen2.5-Coder-7B-Instruct, 4-bit QLoRA
- LoRA rank 8, alpha 32, dropout 0.05
- Batch size 1, gradient accumulation 16, 3 epochs
- Hardware: RTX 4050 Laptop, 6GB VRAM

**Data:**
- 139/17/18 train/val/test (canonical split from Faiza's `datasets/splits/`, 174 total passing examples)

**Results:**
| Epoch | Train Loss | Eval Loss |
|---|---|---|
| 1 | 0.869 | 0.686 |
| 2 | 0.635 | 0.563 |
| 3 | 0.536 | 0.538 |

Clean, consistent decrease across all 3 epochs — no overfitting despite small dataset size.

**Known issues encountered and resolved:**
- Windows-specific bug in `unsloth_zoo`'s fused cross-entropy loss ([unslothai/unsloth#3827](https://github.com/unslothai/unsloth/issues/3827)) — free-GPU-memory detection returns near-zero on WDDM, crashing training. Fixed via monkey-patch in `train_qlora.py` (forces a safe fixed memory budget instead of relying on the broken auto-detection).
- `SFTConfig`'s `completion_only_loss` default conflicted with custom `formatting_func` — resolved by explicitly setting `completion_only_loss=False`.
- 4-bit quantized model rejected automatic CPU/GPU device-map splitting — resolved with explicit `device_map={"": 0}`.

**Known limitation:** 174 total examples is below the plan's 800-1500 target. Team decision (per Faiza) was to train first and treat "collect more data" as a Week 6 decision based on real results — these numbers suggest the pipeline itself works correctly; more data would likely improve generalization further, but the training process is validated end-to-end.

**Qualitative eval (3 HumanEval test samples, greedy decoding):**
- 2/3 produced syntactically valid, reasonably-argued code
- 1/3 (`factorize`) had a missing `import math` — real runtime bug
- 1/3 (`odd_count`) showed a subtle logic gap vs. the docstring's exact expected behavior (correct approach, imprecise edge-case handling)
- Note: initial eval attempts returned empty completions until `min_new_tokens=50` was set — likely a `bos_token_id` tokenizer patching side-effect from Unsloth during training (`bos_token_id: None` reassignment observed in training logs). Worth flagging for anyone else evaluating checkpoints from this pipeline.