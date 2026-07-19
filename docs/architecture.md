# Architecture -- Full Document (Sakshi, Week 4 Wed)

This document ties together the pipeline, dataset, and training plan for the
Agent Distillation Compiler project. Detailed supporting docs are linked
throughout rather than duplicated here.

## 1. Pipeline

The core is a 4-agent LangGraph pipeline (`agents/graph.py`):
Problem + tests
|
v
Planner  ---- breaks problem into numbered implementation steps
|
v
Coder    ---- writes the solution based on the plan
|
v
Sandbox Tester ---- runs solution + tests inside an isolated Docker container
|
+--- pass ---> Done
|
+--- fail ---> Debugger ---- fixes code using the test error
|
v
back to Sandbox Tester (retry, up to MAX_RETRIES)

All three LLM-driven agents (Planner, Coder, Debugger) call a shared
`agents/teacher_router.py`, which tries the local Ollama model first and
falls back to the Groq/Gemini API teacher after 3 connection failures
(a separate mechanism from the Debugger's own test-failure retry loop).

Full diagram + rationale: `docs/architecture_diagram.md`

### Sandbox isolation
The Sandbox Tester runs untrusted code in a hardened Docker container
(`agents/sandbox_executor.py`): no network access, 256MB memory cap, 50%
CPU quota, 64 process limit, all Linux capabilities dropped, read-only
root filesystem, configurable timeout with forced kill.

## 2. Teacher sources

Three independent teacher sources were used to collect training
trajectories, each running on a different team member's task subset:

| Source | Model | Machine | Tasks | Notes |
|---|---|---|---|---|
| Yeshita | Qwen2.5-7B (local Ollama) | RTX 4050 6GB | 179 | Primary local teacher |
| Faiza | Qwen2.5-14B (local Ollama) | RTX 5070 8GB | 179 | Larger local teacher |
| Sakshi | Groq/Gemini API + local 7B fallback | RTX 4050 6GB | 180 | See note below |

Note on Sakshi's batch: started on Groq/Gemini API teachers but hit
real rate limits partway (Groq daily token cap, Gemini free tier not
enabled on the account) -- the remaining ~90 tasks used the local Ollama
7B model instead. Pass rate was noticeably higher on the API-teacher
portion, a useful data point on teacher quality (see
`docs/merged_dataset_report.md` and `docs/teacher_comparison_notes.md`).

## 3. Dataset pipeline
Raw task sets (HumanEval + MBPP subsets, split 3 ways)
|
v
Trajectory collection (3 sources, run independently)
|
v
datasets/collector.py  ---- merges + dedupes all 3 sources
|
v
datasets/schema.py  ---- validates every record
|
v
Filter to passing-only  ---- (Faiza, faiza/dataset-filter)
|
v
Train/val/test split (80/10/10)  ---- datasets/splits/
|
v
Compression (trace -> single CoT)  ---- datasets/compressor.py (Yeshita)
|
v
Final SFT dataset

Current numbers:
- Merged (all sources): 538 unique tasks, 279 passing (51.9%) --
  `docs/merged_dataset_report.md`
- Canonical training split (used for Week 5 training): built from an
  earlier 358-task merge, 174 passing examples, split 139/17/18 --
  `docs/dataset_split_final.md`
- DPO preference pairs: 97 pairs across 49 tasks --
  `datasets/raw/dpo_pairs_faiza_14b.json`

Known gap: 174 training examples is below the original 800-1500
target. Team decision was to proceed with Week 5 training on this size
first and treat "collect more data" as a Week 6 decision based on real
checkpoint results, rather than delaying training to guess upfront.
Sakshi's expanded 538-task merge (from the trajectory run above) is
available if more data is needed later.

## 4. Evaluation harness

`evaluation/benchmark.py` + `evaluation/run_eval.py` provide a reusable
harness for scoring any pipeline configuration against a task set:
- pass@1 -- fraction of tasks passing on first full pipeline run
- latency -- wall-clock time per task (planner+coder+tester+debugger loop)
- cost -- per-call cost estimate (flat rate for now, to be refined with
  real token counts once available)
- VRAM -- GPU memory usage via `torch.cuda.memory_allocated()`, 0 if no
  CUDA device present (e.g. pure API-based runs)

Results are saved as JSON reports (`evaluation/results/`, gitignored) for
comparison across teacher configurations and training checkpoints.

## 5. Training plan (Week 5)

Two parallel QLoRA runs on the same base model
(Qwen2.5-Coder-7B-Instruct), for comparison:

| Run | Owner | Rank | Batch | Grad accum | Hardware |
|---|---|---|---|---|---|
| Primary | Faiza | 32 | 2 | 8 | RTX 5070 8GB |
| Ablation | Yeshita | 8-16 | 1 | 16 | RTX 4050 6GB |

Ablation run results (already completed as of this doc, see
`docs/training_guide.md` for full detail):
- 3 epochs, clean loss decrease (0.869 -> 0.538 eval loss), no overfitting
  despite small dataset.
- Real bugs hit and fixed along the way: a Windows-specific Unsloth memory
  bug, an `SFTConfig`/`completion_only_loss` conflict, and a device-map
  issue with 4-bit quantization.
- Qualitative spot-check on 3 HumanEval samples: 2/3 solid, 1/3 missing an
  import, 1/3 with a subtle logic gap on edge cases.

## 6. Team tracking

A shared W&B project will track all training runs going forward (Sakshi,
Week 4 Thu task) so Faiza's and Yeshita's runs are comparable side by side.

## 7. Open items
- CI-only sandbox test failure on GitHub Actions (not reproducible
  locally) -- flagged, not blocking, see project notes.
- Gemini API free tier needs manual Google Cloud Console setup before it
  can be used reliably for future trajectory collection.
- Dataset size (174 training examples) may need expansion depending on
  Week 5 checkpoint eval results.