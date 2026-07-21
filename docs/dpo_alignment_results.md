@'
# DPO Alignment Results — Week 7

## Context
Originally planned to run DPO alignment on the primary rank-32 QLoRA
checkpoint (mine, from Week 5-6). Blocked partway through Monday's setup by
a Windows Smart App Control policy on the training laptop that began
blocking native DLLs (triton, then pyarrow) system-wide, affecting even
previously-working training scripts. The laptop owner (Lakshit) declined a
full Windows reset (Microsoft's only fix once Smart App Control is fully
"On"). Attempted a WSL2/Linux workaround, but no GPU-enabled Ubuntu distro
was available in time.

**Yeshita substituted her own rank-8 ablation checkpoint** for the DPO run
so Week 7 wasn't blocked team-wide. Both checkpoints performed nearly
identically pre-DPO (~87-88% pass@1), so the substitution is reasonable
for this stage.

## Training setup
- Base: Yeshita's rank-8 ablation checkpoint (Week 5-6)
- Data: 97 DPO preference pairs (collected Week 3-4, `dpo_pairs_faiza_14b.json`)
- Final checkpoint: `models/qlora-primary-dpo/checkpoint-7` (local, gitignored)

## Results (from trainer logs)
| Metric | Before | After |
|---|---|---|
| rewards/accuracies | 0.225 | 0.588 |
| mean_token_accuracy | 0.839 | 0.847 |
| train_loss (final) | — | 0.673 |

`rewards/accuracies` improving from 0.225 to 0.588 indicates the model
became markedly more consistent at preferring the correct (preferred) code
over the rejected/earlier-attempt code — the expected direction and
magnitude for a DPO pass on a small preference set.

## Known gap
Post-DPO pass@1 re-evaluation on the held-out val set was attempted but
blocked by a local environment issue (checkpoint load silently hung, no
traceback) — not resolved in time for this report. The numbers above are
training-time metrics only; task-level pass@1 impact of DPO alignment is
not directly confirmed.

## Status
This stands as the team's official DPO result. If laptop/environment
access is resolved later, re-running DPO on the actual primary rank-32
checkpoint (and completing the pass@1 re-eval) would be a natural follow-up
for completeness, but is not blocking the current deliverable.
