# Reproducible experiment results

This directory contains the small tabular artifacts used to produce
the quantitative results in the project README.

## Training

`training/baseline_v1_seed0_5000_metrics.csv` contains training metrics
recorded every 50 optimization steps and full validation metrics every
250 steps for the seed-0 ACT baseline.

## Action-chunk ablation

The files under `action_chunk_ablation/` contain paired closed-loop
evaluations on seeds 1000–1049:

- `action_steps_1.csv`
- `action_steps_5.csv`
- `action_steps_20.csv`

All three runs use the same checkpoint and environment configuration.
Only the number of actions executed before replanning changes.

Large datasets, model checkpoints, videos, logs, and caches remain
outside Git.
