# Model Card — Agent Distillation Compiler Student Model

## Overview
A small, locally-runnable code-generation model distilled from a
multi-agent teacher pipeline (Planner → Coder → Tester → Debugger), trained
via QLoRA supervised fine-tuning followed by DPO preference alignment.
Goal: retain most of the teacher pipeline's task-solving ability in a
single-pass, low-latency, low-cost student model.

## Base model
- **Model:** Qwen2.5-Coder-7B-Instruct
- **Quantization:** 4-bit (bnb-4bit, via Unsloth)

## Training method
- **SFT:** QLoRA, rank 32, alpha 32, dropout 0
  - Batch size 2, gradient accumulation 8 (effective batch 16)
  - 4 epochs; learning rate 2e-4 (epochs 1-3), reduced to 1e-4 (epoch 4)
    after pass@1 plateaued
- **DPO:** rank 8 checkpoint (substituted due to a training-environment
  issue on the primary rank-32 run's machine — see Known Limitations),
  97 preference pairs, 1 epoch

## Training data
- **Source:** HumanEval + MBPP problems, solved via a 4-agent teacher
  pipeline using a 14B teacher model (qwen2.5:14b-instruct), trajectories
  merged and deduplicated across 3 team members' collection runs
- **Size:** 358 total trajectories collected, 174 passing (used for
  training) — below the project's original 800-1500 example target
- **Split:** 139 train / 17 val / 18 test (80/10/10), stratified check
  across source (HumanEval/MBPP) and difficulty (retry-count proxy)
- **Format:** compressed single-pass chain-of-thought (teacher's
  multi-step reasoning summarized into one coherent explanation + code)
- **DPO pairs:** 97 preference pairs (rejected = earlier failing attempt,
  preferred = final passing attempt) from tasks that needed debugger
  retries during trajectory collection

## Hyperparameters (primary SFT run)
| Parameter | Value |
|---|---|
| LoRA rank | 32 |
| LoRA alpha | 32 |
| Batch size | 2 |
| Gradient accumulation | 8 |
| Epochs | 4 |
| Learning rate | 2e-4 → 1e-4 |
| Max sequence length | 2048 |

## Results
| Metric | Value |
|---|---|
| Final eval_loss (SFT, epoch 4) | 0.495 |
| Val pass@1 (SFT, epoch 3-4) | 87.5% (14/16) |
| DPO rewards/accuracies | 0.225 → 0.588 |
| DPO mean_token_accuracy | 0.839 → 0.847 |

Three independently trained configs (this rank-32 primary run, a rank-8
ablation, and a Llama3.1-8B experiment) converged to a similar ~87-88%
pass@1 ceiling — consistent with a data-size limitation rather than a
training configuration issue.

## Intended use
Single-pass Python code generation for well-scoped programming problems
(function-level, competitive-programming style — HumanEval/MBPP
difficulty range). Designed to run locally on consumer GPU hardware
(6-8GB VRAM) without needing the full multi-agent teacher pipeline at
inference time.

## Limitations
- Trained on 174 passing examples, below the original 800-1500 target —
  pass@1 plateaued at ~87-88%, likely a data-size ceiling rather than a
  model-capacity limit
- DPO alignment was run on a substitute (rank-8) checkpoint, not the
  primary rank-32 model, due to a Windows Smart App Control policy
  blocking native ML library DLLs (triton, pyarrow) on the primary
  training machine mid-Week-7 — a hard device-level block, not a code
  issue, that could not be resolved by package version changes or WSL2
  (no GPU-enabled Linux distro available in time)
- Post-DPO pass@1 was not independently re-verified due to a separate
  environment issue during re-evaluation
- Evaluated only on HumanEval/MBPP-style problems; real-world/domain-
  specific coding tasks not covered by this evaluation

## Files
- Metadata: `models/qlora-primary-final-metadata.json`
- Training guide: `docs/training_guide.md`
- DPO results: `docs/dpo_alignment_results.md`
- Dataset split details: `docs/dataset_split_final.md`
