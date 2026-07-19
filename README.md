# Agent Distillation Compiler

A multi-agent pipeline (Planner -> Coder -> Sandbox Tester -> Debugger) used
to generate coding-problem trajectories from teacher LLMs, distilled into a
smaller student model via QLoRA fine-tuning.

Full architecture, dataset, and training plan: `docs/architecture.md`

## Team
- Yeshita Motwani
- Faiza Bagban
- Sakshi Kolhe

## Setup
See `docs/env_sakshi.md`, or the equivalent per-member environment notes,
for local environment setup (Ollama, CUDA, conda env `adc`).

## Training tracking (W&B)

All training runs (primary QLoRA, ablation run, and any future experiments)
log to a single shared Weights & Biases project so results are directly
comparable across configurations.

**Project:** `agent-distillation-compiler`

**To log a training run to the shared project:**

1. Make sure you're logged in to W&B locally:
wandb login
   (get your API key from https://wandb.ai/authorize if you don't have one
   pasted already)

2. Ask Sakshi to add you as a collaborator on the shared project if you
   haven't been invited yet (wandb.ai project settings -> Users).

3. In your training script, use the shared config helper instead of calling
   `wandb.init()` directly:
```python
   from training.wandb_config import init_wandb_run

   run = init_wandb_run(
       run_name="faiza-primary-rank32",   # descriptive, includes your name + config
       config={"rank": 32, "batch_size": 2, "grad_accum": 8},
       tags=["primary"],                  # or ["ablation"], etc.
   )
```

4. Log metrics as usual (`wandb.log({...})` inside your training loop) and
   call `run.finish()` at the end.

This keeps every run -- primary, ablation, and any later experiments --
visible on one shared dashboard for direct comparison.

## Repo structure (high level)
- `agents/` -- the 4-agent pipeline + teacher router + sandbox executor
- `datasets/` -- trajectory collection, merging, schema validation, splits
- `evaluation/` -- benchmark harness (pass@1, latency, cost, VRAM)
- `training/` -- QLoRA training scripts + shared W&B config
- `docs/` -- architecture, dataset reports, training notes
- `docker_configs/` -- sandbox Docker image definition