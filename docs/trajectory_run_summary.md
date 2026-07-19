# 14B Teacher Trajectory Collection — Final Summary

## Run details
- Model: `qwen2.5:14b-instruct-q4_K_M`
- Task subset: `datasets/raw/tasks_faiza_14b.json` (179 tasks — 164 HumanEval, 15 MBPP)
- Output: `datasets/raw/trajectories_faiza_14b.json`

## Results
- **143/179 passed (79.9%)**
- HumanEval: ~139/164 passed (~85%)
- MBPP: 4/15 passed (~27%) — notably lower, flagged for investigation

## Issues found and fixed during this run
1. Debugger agent wasn't stripping code fences from output — fixed in
   `agents/debugger.py` (merged).
2. HumanEval-style tests define `check(candidate)` but never call it —
   sandbox was silently passing broken code. Fixed in `agents/tester.py`
   to auto-invoke `check()` (merged).
3. Intermittent Ollama 500 errors during long batch runs (roughly every
   5-10 requests) — added retry logic to `agents/planner.py` and
   `agents/coder.py` (merged).

## Follow-up
- MBPP pass rate is much lower than HumanEval — worth a closer look at
  whether this is a dataset-format edge case (bare asserts) or genuinely
  harder problems, before Week 4 filtering/split work.
- DPO preference pairs (53 pairs from tasks that needed retries) collected
  separately in `datasets/raw/dpo_pairs_faiza_14b.json`.