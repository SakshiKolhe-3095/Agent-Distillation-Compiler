# Teacher Model Comparison: 7B vs 14B

## Setup
Compared `qwen2.5:7b-instruct-q4_K_M` vs `qwen2.5:14b-instruct-q4_K_M` as teacher
models through the full 4-agent pipeline (Planner → Coder → Tester → Debugger)
on 5 sample coding problems. Full results in
`benchmarks/teacher_comparison_7b_vs_14b.json`.

## Results

| Metric | 7B | 14B |
|---|---|---|
| Pass rate | 4/5 | 4/5 |
| Total time | 63.4s | 57.2s |
| Avg time/problem | 12.7s | 11.5s |

## Observations

- Both models reached the same pass rate on this sample. Quality difference
  wasn't obviously visible at this problem difficulty — likely need harder
  problems to see 14B's advantage show up.
- 14B was slightly faster overall despite larger size — fewer wasted
  retry cycles seems to be the reason, not raw inference speed.
- **Shared failure:** both models failed the same problem ("find max value in
  list") after 3 retries each. Since both teachers hit the identical failure,
  this points to a bug in the Debugger agent's retry logic rather than a
  model capability gap — worth investigating separately from teacher choice.

## Tradeoffs

- 7B: lower VRAM footprint (fits comfortably on 8GB cards), faster to pull/load.
- 14B: no clear quality win yet on simple problems, but expected to matter
  more on harder/multi-step tasks — needs a follow-up test with a tougher
  problem set before making a final call on which becomes the primary
  teacher for trajectory generation (Week 3).

## Recommendation
Keep both available. Use 7B for fast iteration/testing, 14B for final
trajectory generation batches once the harder-problem comparison confirms
a quality gap.