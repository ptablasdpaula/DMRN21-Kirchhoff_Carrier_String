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
