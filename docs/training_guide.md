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
---

## Primary Run (Faiza, rank 32) — Final

**Config:**
- Base model: Qwen2.5-Coder-7B-Instruct, 4-bit QLoRA
- LoRA rank 32, alpha 32, dropout 0
- Batch size 2, gradient accumulation 8, 4 epochs total
- Learning rate: 2e-4 (epochs 1-3), lowered to 1e-4 for epoch 4 after pass@1 plateau
- Hardware: RTX 5070 Laptop, 8GB VRAM (borrowed laptop, first-time GPU setup)

**Data:**
- Same canonical split as ablation run — 139/17/18 train/val/test, 174 total passing examples
- Teacher: qwen2.5:14b-instruct-q4_K_M (Ollama, local)

**Results:**
| Epoch | Eval Loss | Val pass@1 |
|---|---|---|
| 1 | 0.6163 | 88.2% (15/17) |
| 2 | 0.5461 | — |
| 3 | 0.505  | 87.5% (14/16) |
| 4 | 0.495  | 87.5% (14/16) — unchanged, lowered LR |

Loss improved steadily through all 4 epochs. pass@1 plateaued from epoch 3
onward despite continued loss improvement — diagnosed as a data-size
ceiling (174 examples), not a training configuration issue, since a
learning-rate drop in epoch 4 improved loss further without moving pass@1.
**Selected checkpoint: epoch 4 (checkpoint-36)** — best loss, no pass@1
regression vs epoch 3.

**Known issues encountered and resolved (in addition to the Windows
fused-CE bug above, which also applied here):**
- `datasets/__init__.py` (added during a merge) shadowed pip's real
  `datasets` package for any script importing `unsloth` from repo root —
  caused `ImportError: cannot import name 'Dataset'`. Worked around locally
  by temporarily removing repo root from `sys.path` before the unsloth
  import; fixed properly team-wide by removing the stray `__init__.py`.
- HumanEval-format test cases define `check(candidate)` but never call it —
  combined with the sandbox running tests as a plain script (not pytest),
  this meant broken code could silently report `passed: True`. Fixed in
  `agents/tester.py` by auto-appending the `check(fn_name)` invocation.
- Debugger agent wasn't stripping ` ```python ` code fences from its
  output (unlike the coder agent) — any task that needed a retry/debug
  cycle failed with broken syntax regardless of whether the fix was
  logically correct. Fixed by reusing `strip_code_fences()` in
  `debugger.py`.
- `.gitignore` had `models/` (whole-directory exclusion) which silently
  blocks `!` negation exceptions for files inside it — changed to
  `models/*` so the metadata JSON could still be tracked while the actual
  model weights stay ignored.

**Team outcome:** all three configs (this rank-32 primary run, Yeshita's
rank-8 ablation, Sakshi's Llama3.1-8B experiment) converged to roughly the
same ~87-88% pass@1 ceiling. Team decision: accept this as the final result
rather than pursue further data collection — treated as a genuine,
well-diagnosed finding for the report rather than a limitation to hide.
