# Cached scalar reconstruction experiment

The experimental CUDA path in `experiments/cached_reconstruction` stores three
forward scalars per internal step and string. During the adjoint it reuses
those scalars to simplify state reconstruction and reduce the width of the
cross-mode reduction. It retains all 175 modes and float64 for the same
discrete model. It is first-order-only and is not a public/default backend.
Periodic checkpoint, nonfinite and singularity guards are retained; passing a
boundary check alone does not guarantee accurate interior states or gradients.

## A100 40 GB, four-second full loss and six gradients

Job 34505298 measured synthesis, filtering, independent per-item CeL and
backward, with 256-step checkpoints and three warmed repetitions. All
methods used the same input vectors, node and loss in this job:

| Method | Batch 1 | Batch 128 | Batch 128 peak PyTorch allocation |
|---|---:|---:|---:|
| Forward sensitivity | 2.160 s | 3.295 s | 10.70 GB |
| Replay adjoint | 2.198 s | 3.262 s | 8.19 GB |
| Sequential reconstruction | 1.947 s | 2.882 s | 8.19 GB |
| Cached reconstruction | 1.408 s | 2.219 s | 9.82 GB |

The batch-128 cached result reduced wall time by about 23% compared with
ordinary reconstruction, while increasing peak allocation by 1.63 GB. It is
not directly comparable to the highlighted 1.589-second two-gradient H100
measurement: this job used an A100 and calculated all six gradients.

Cached and seeded waveforms, individual losses and all six coordinate gradients
were checked against replay before timing. For cached reconstruction at four
seconds and batch 128, waveform and loss errors were zero in that check and the
scaled gradient relative L2 error was 1.12e-9. The source hash remained unchanged
during the run. [Raw measurements](../benchmarks/reconstruction_results/a100-cache-compare.json).
The independent full-duration arbitrary-VJP and numerical-guard checks passed
in job 34504688. Additional A100 job 34505340 passed four-second filtered-loss
gradient checks in perceptual and material coordinates against replay and an
independent CPU forward Jacobian, plus short finite-difference and parameter
corner cases. [Coordinate validation](../benchmarks/cache_reconstruction_results/a100-coordinate-stress.json).
H100 job 34505297 completed the same paired comparison on `xdg1`: for four
seconds, batch 128 and all six gradients, forward sensitivity took 2.110 s,
replay took 2.158 s, ordinary reconstruction 1.916 s, and cached reconstruction
**1.521 s**. The respective cached and ordinary reconstruction peaks were
9.866 and 8.240 GB of PyTorch allocation. The cached waveform and loss checks
were zero relative error against replay and scaled gradient relative L2 error
was 1.34e-9. The existing source hash remained unchanged. [H100 raw result](../benchmarks/reconstruction_results/h100-cache-compare.json).
The grouped comparison is job 34505774, reported below.

This path stores approximately 12.7 MB per four-second string (1.63 GB at
batch 128) for the three scalar tapes, in addition to existing allocations.
The checks are empirical and cannot guarantee stability for every parameter
setting; in particular, finite cache corruption may not be caught by the
boundary guard. Further validation is required before wider use.

## Combining cached reconstruction and grouped loss

A100 job 34505775 combined cached reconstruction with the experimental
[grouped CeL loss](loss_batch_experiments.md), using groups of 16. All rows
below calculate all six gradients for 128 four-second strings and include the
same synthesis, filtering and loss. Peak figures are PyTorch allocated memory.

| Method | Full evaluation | Peak allocation |
|---|---:|---:|
| Forward sensitivities, independent losses | 3.290 s | 10.70 GB |
| Forward sensitivities, grouped losses | 3.102 s | 5.83 GB |
| Replay adjoint, independent losses | 3.267 s | 8.19 GB |
| Sequential reconstruction, independent losses | 2.879 s | 8.19 GB |
| Cached reconstruction, independent losses | 2.218 s | 9.82 GB |
| Cached reconstruction, grouped losses | **2.027 s** | **4.95 GB** |

Three warmed synchronized repetitions were measured. The combined result is
about 38% faster and uses 54% less peak PyTorch allocation than the forward
sensitivity/independent-loss baseline in this job. Both grouped methods and
cached reconstruction were independently compared with replay for the full
four-second batch before timing. Cached+grouped had zero waveform error,
2.95e-16 relative loss error and 1.24e-9 scaled gradient relative L2 error in
that test. This remains experimental and does not establish H100 performance.
[Raw combined measurements](../benchmarks/reconstruction_results/a100-cache-grouped.json).

## H100 combined result

H100 80 GB HBM3 job 34505774 measured the same four-second, batch-128,
all-six-coordinate full loss/gradient workload. Three warmed runs were
recorded on the same node, and both grouped methods and cached reconstruction
passed waveform, loss and gradient checks against replay before timing.

| Method | Full evaluation | Peak PyTorch allocation |
|---|---:|---:|
| Forward sensitivities, independent losses | 2.109 s | 10.745 GB |
| Forward sensitivities, grouped losses | 1.963 s | 5.880 GB |
| Replay adjoint, independent losses | 2.159 s | 8.240 GB |
| Sequential reconstruction, independent losses | 1.914 s | 8.240 GB |
| Cached reconstruction, independent losses | 1.519 s | 9.865 GB |
| Cached reconstruction, grouped losses | **1.373 s** | **5.000 GB** |

The combined result reduces time by 34.9% and peak PyTorch allocation by
53.5% against the matched forward-sensitivity/independent-loss baseline.
Its waveform relative L2 error was zero, relative summed-loss error was
2.95e-16 and scaled gradient relative L2 error was 1.25e-9 against replay.
The cached solver and grouped loss remain experimental; the grouped helper
supports one common bound target, not different targets per item. [Raw result](../benchmarks/reconstruction_results/h100-cache-grouped.json).
