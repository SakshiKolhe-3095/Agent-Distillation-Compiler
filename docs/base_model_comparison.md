# Base Model Comparison -- Sakshi's Secondary Experiment (Week 5 Fri)

## Setup
Same training data (139 train / 17 val examples, compressed SFT format)
and same hardware constraints (RTX 4050, 6GB VRAM) as the ablation run,
but a different base model -- to see whether base model choice matters
more than QLoRA rank/hyperparameters for this task.

| | Sakshi (this experiment) | Yeshita (ablation) |
|---|---|---|
| Base model | Meta-Llama-3.1-8B-Instruct | Qwen2.5-Coder-7B-Instruct |
| LoRA rank | 4 | 8 |
| Max seq length | 512 | 1024 |
| Epochs | 1 | 3 |
| Batch / grad accum | 1 / 16 | 1 / 16 |

Rank and seq_len were both cut more aggressively than Yeshita's run because
the 8B model needed more headroom than the 7B on the same 6GB card --
even after cuts, each training step still took ~5 minutes (vs Yeshita's
faster per-step time on the smaller 7B model). Only 1 epoch was run given
the time cost per step.

## Results

Sakshi's Llama3.1-8B (this run):
- Train loss: 1.117, eval loss: 1.022 (after 1 epoch)
- pass@1 on val set: 31.2% (checkpoint-9), measured via the new
  live checkpoint evaluation hook (`evaluation/live_checkpoint_eval.py`)

Yeshita's Qwen2.5-Coder-7B ablation:
- Eval loss after 3 epochs: 0.538 (down from 0.686 after epoch 1) -- see
  `docs/training_guide.md` for full detail
- No pass@1 number logged yet for this run (loss-only so far)

## Honest caveats
- Not a clean apples-to-apples comparison. Sakshi's run had 1 epoch,
  Yeshita's had 3; different rank, different seq_len. A lower loss or
  higher pass@1 here doesn't cleanly isolate "base model effect" from
  "training budget effect."
- Llama3.1-8B eval loss (1.022) is noticeably higher than Qwen's
  (0.538), but that's expected given only 1 epoch vs 3 -- loss was still
  decreasing (1.186 train loss mid-run, 1.117 at the end).
- 31.2% pass@1 is a first real number from the new live-eval hook -- no
  equivalent number exists yet for Yeshita's or Faiza's checkpoints, so
  it can't be directly compared to them until they run the same hook on
  their checkpoints.

## Recommendation
If time allows in a future week, run Yeshita's/Faiza's checkpoints through
`evaluation/live_checkpoint_eval.py` too, so all three runs have a directly
comparable pass@1 number rather than only loss curves for two of them and
pass@1 for one.