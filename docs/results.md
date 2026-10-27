# Result provenance

## Historical measurements

The JSON files in `benchmarks/` were generated during the October 26, 2026
performance work. Their scripts, methods and configuration metadata are retained.
The native headline measurement is approximately 0.51 s for four seconds of audio
with N=128, 44.1 kHz internal rate, CPU float64, one thread on an M4 Max. It includes
forward sensitivities and the backward contraction, after extension warmup. It is
not a benchmark of the current default box (N=176, oversampling ×3).

`experiments/results/material_coordinates_cel.json` is the prior four-second
material-coordinate Adam/L-BFGS experiment, preserved from commit `2d717c5`.
Its original generating and plotting scripts are recoverable from that commit.
It used the unchanged default CeL, eight material coordinates and the earlier
feasibility guard. It is historical evidence, not a result for the new bounded
six-coordinate formulation.

## Current experiment

`pixi run recovery` writes `experiments/results/recovery_adam.json` and
`recovery_lbfgs.json`, with ranges, selected grid, full duration, excitation,
parameter order, requested/effective starts, loss configuration, histories,
evaluation counts, timing, stop reason and final parameter errors.
Each optimiser starts independently from the same perturbed initial values.
Timing includes evaluation and fitting but excludes target synthesis and extension
warmup; it is end-to-end fitting time, not a kernel-only benchmark.

The new numerical validation checks run from `pixi run numerics-check`; the
independent reference comparisons remain available via `pixi run forward-check`.
Claude's reported staged recovery and fractional-step results were not supplied
as executable files and are not represented as reproduced evidence here.

## Recorded six-parameter run


4-second experiment: N=176, oversampling ×3. Both methods start independently; loss and duration stay unchanged.

| Parameter | Target | Initial | Adam | Error | L-BFGS | Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| f0 | 76.5713 | 84.22843 | 76.64997 | +0.103% | 76.57223 | +0.001% |
| B | 0.0002875574 | 0.0002614159 | 0.0001764258 | -38.647% | 0.00023915 | -16.834% |
| amplitude | 0.002 | 0.0022 | 0.002000209 | +0.010% | 0.001999996 | -0.000% |
| radius | 0.0003556 | 0.0003232727 | 0.000324671 | -8.698% | 0.0003232782 | -9.089% |
| sigma_0 | 0.921034 | 1.013137 | 0.9213118 | +0.030% | 0.9210654 | +0.003% |
| sigma_1 | 0.000286393 | 0.0002603572 | 0.000246228 | -14.024% | 0.0002841462 | -0.785% |

| Method | Initial loss | Best loss | Steps | Evaluations | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| adam | 0.0196504 | 3.350667e-05 | 135 | 136 | 298.7 |
| lbfgs | 0.0196504 | 3.36666e-06 | 55 | 179 | 393.7 |

adam: Less than 10% best-loss improvement over the last 20 optimizer steps.

lbfgs: Less than 10% best-loss improvement over the last 20 optimizer steps.

These are recovery results, not proof of parameter identifiability or nonlinear accuracy. Times are observed fitting wall time; concurrent validation may affect them.

Recreate the comparison table and loss plot with
`pixi run python experiments/summarize_recovery.py`. Outputs go to ignored
`output/recovery/`.
