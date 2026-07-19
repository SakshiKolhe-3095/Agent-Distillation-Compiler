# Dataset Split — Finalized (Week 4)

## Source
Built from `datasets/raw/trajectories_merged.json` (Sakshi's merged/deduped
combined dataset — all 3 team members' trajectory sources).

## Numbers
- Merged total: 358 examples
- Passing-only: 174 examples
- Split: Train 139 (79.9%) / Val 17 (9.8%) / Test 18 (10.3%)

## Balance validation
Checked across source (HumanEval vs MBPP) and difficulty (retry count proxy).
Overall reasonably balanced; test set skews slightly more MBPP-heavy (38.9%
vs 19.4% in train) — attributable to small sample size at this split level,
not a systematic issue.

## Known gap
174 passing examples is below the plan's 800-1500 target range. Flagged to
team (Yeshita, Sakshi) — may need additional trajectory collection before
Week 5 training kickoff, or proceed with current size and revisit if
training results are weak.

## DPO pairs
97 preference pairs (49 tasks) collected separately in
`datasets/raw/dpo_pairs_faiza_14b.json`, built from every consecutive retry
attempt across tasks that needed debugging.

## Status
Split finalized and canonical at `datasets/splits/{train,val,test}.json`.
Confirmed by Yeshita as the single source of truth (resolved earlier
duplicate `datasets/sft/` folder — her training script now points here).