# ACT Action-Chunk Execution Ablation

Paired evaluation on 50 seeds (1000–1049).

## Main results

| Action steps | Success | 95% Wilson CI | Mean steps | Requests/episode | Estimated inference ms/episode |
|---:|---:|---:|---:|---:|---:|
| 1 | 28/50 (56.0%) | [42.3%, 68.8%] | 188.06 | 188.06 | 2420.76 |
| 5 | 39/50 (78.0%) | [64.8%, 87.2%] | 135.92 | 27.58 | 117.12 |
| 20 | 44/50 (88.0%) | [76.2%, 94.4%] | 103.88 | 5.48 | 13.46 |

## Paired McNemar comparisons

| Comparison | Both pass | Both fail | Left only | Right only | Exact p-value |
|---:|---:|---:|---:|---:|---:|
| 1 vs 5 | 24 | 7 | 4 | 15 | 0.019211 |
| 1 vs 20 | 22 | 0 | 6 | 22 | 0.003719 |
| 5 vs 20 | 37 | 4 | 2 | 7 | 0.179688 |

## Baseline failure crossovers

Baseline failures: 1011, 1016, 1034, 1037, 1040, 1042

- `1` steps rescued: 1011, 1016, 1034, 1037, 1040, 1042
- `1` steps regressed: 1000, 1001, 1002, 1004, 1005, 1006, 1007, 1008, 1010, 1014, 1015, 1021, 1023, 1024, 1025, 1028, 1030, 1032, 1036, 1041, 1045, 1046
- `5` steps rescued: 1040, 1042
- `5` steps regressed: 1000, 1001, 1005, 1021, 1025, 1030, 1045

Failed under every setting: none

## Experimental design

All three evaluations used the same ACT checkpoint, environment,
observation and action definitions, success criterion, and 50 test
seeds. The only changed variable was the number of predicted actions
executed before requesting a new action chunk:

- `1`: replan every control step (0.05 seconds at 20 Hz).
- `5`: replan every five control steps (0.25 seconds).
- `20`: execute the full trained action chunk (1.0 second).

The estimated inference cost per episode is computed as
`ceil(episode_steps / action_steps) * mean_request_latency`. It is an
online-cost estimate rather than a separately timed end-to-end
benchmark. The three servers ran concurrently on separate GPUs, so
per-request latency should not be treated as a pure model-throughput
comparison.

## Action clipping diagnostic

| Action steps | Clipped-step fraction | Clipped-element fraction | XYZ clipping | Gripper clipping | Mean maximum excess |
|---:|---:|---:|---:|---:|---:|
| 1 | 98.84% | 24.71% | 0.00% | 98.84% | 0.1029 |
| 5 | 98.98% | 24.74% | 0.00% | 98.98% | 0.1043 |
| 20 | 96.72% | 24.18% | 0.00% | 96.72% | 0.1011 |

Clipping occurred only in the approximately discrete gripper
dimension. Cartesian delta actions were never clipped. Similar
gripper saturation across all settings makes it unlikely to explain
the success-rate differences between execution horizons.

## Interpretation

Executing the complete 20-step action chunk produced the highest
success rate, the shortest mean episode, and the fewest policy
requests. Replanning every step reduced success from 88% to 56%,
consistent with disrupting the temporal continuity of approach,
grasp, and lift motions.

The paired McNemar tests show significant differences for `1` versus
`5` (`p=0.019211`) and `1` versus `20` (`p=0.003719`). The observed
difference between `5` and `20` did not reach the 0.05 significance
threshold (`p=0.179688`) with 50 paired seeds, so it should be reported
as a higher point estimate rather than a statistically established
advantage.

Shorter execution horizons can still correct particular failures:
one-step replanning rescued all six baseline failures, but introduced
22 regressions on baseline successes. No seed failed under every
setting. This crossover suggests a future adaptive controller could
retain long coherent chunks normally and trigger early replanning
only when execution deviates from the expected state.

## Decision

Use `n_action_steps=20` as the current default. It achieved `44/50`
successes (88%) and required an estimated 5.48 policy requests per
episode, compared with 188.06 requests for one-step replanning.
