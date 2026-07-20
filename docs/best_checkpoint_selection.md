@'
# Best Checkpoint Selection — Primary QLoRA (rank 32)

## Candidates evaluated
| Checkpoint | Epoch | eval_loss | val pass@1 |
|---|---|---|---|
| checkpoint-9  | 1 | 0.6163 | 88.2% (15/17) |
| checkpoint-18 | 2 | 0.5461 | — |
| checkpoint-27 | 3 | 0.505  | 87.5% (14/16) |
| checkpoint-36 | 4 | 0.495  | 87.5% (14/16) |

## Decision
**Selected: checkpoint-36 (epoch 4)**

pass@1 plateaued at 87.5% from epoch 3 onward — confirmed as a data-size
ceiling (174 training examples), not a training config issue. Since loss
kept improving through epoch 4 with no pass@1 regression, epoch 4 is
selected as the best checkpoint: same task performance, better-calibrated
loss, no overfitting signal observed.

## Context
Team decision (Yeshita, Sakshi, Faiza): accept current performance
(~87-88% pass@1, converged across three separate configs: this rank-32
primary run, Yeshita's rank-8 ablation, Sakshi's Llama3.1-8B experiment)
rather than pursue further data collection. Moving to Week 8 packaging.
'@ | Set-Content -Encoding utf8 docs/best_checkpoint_selection.md