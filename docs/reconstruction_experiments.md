# Sequential checkpoint reconstruction

This is an experimental exact-recurrence CUDA backward path. It reconstructs
previous modal states backward from each saved end checkpoint, applies the
adjoint and resets the reconstructed state at each segment boundary. It does
**not** parallelise independent segments in time; that separate user proposal is
recorded in [parallel checkpoint segments](ideas/parallel_checkpoint_segments.md).

The inverse of this particular recurrence is a diagonal-plus-rank-one solve.
Inverse and adjoint reductions are fused. The path avoids forward replay and the
per-step temporary tape but incurs inverse calculations, boundary-drift checks
and one device synchronization to check errors. It is float64, first derivative
only, CUDA, stride 1 and at most 256 intervals; it is not the public default.

## H100 paired results

Job 34307843, node xdg2, NVIDIA H100 80GB HBM3, 30 October 2026. The same
176-interval string, 132.3 kHz internal rate, 44.1 kHz output, 4-second signal,
256-step checkpoints, independent CeL loss per item, and varied parameters were
used for both algorithms. Each cell is a median of three warmed, synchronized
measurements. Compilation and warmup are excluded.

| Batch | Existing replay, loss + backward | Reconstruction, loss + backward | Reduction |
|---:|---:|---:|---:|
| 1 | 1.558 s | 1.366 s | 12.3% |
| 8 | 1.601 s | 1.408 s | 12.1% |
| 64 | 1.841 s | 1.646 s | 10.6% |
| 128 | 2.162 s | 1.916 s | 11.4% |

At batch 128 synthesis alone is about 0.496 s for each method; arbitrary
VJP including synthesis is 1.960 s replay vs 1.713 s reconstruction. Peak GPU
allocation for the full loss/backward was 8.243 GB for both in this run:
removing the small replay tape did not reduce the workload peak, which includes
loss and its intermediates. Different hardware and checkpoint sizes require
separate measurements. Three repetitions do not prove performance stability
across machines or different scheduler contention.

The full test passed 83 checks on H100, including four-second outputs and
arbitrary VJPs for batch 128, parameter-box corners, amplitudes, damping,
multiple grids, excitations, and deliberately singular/nonfinite/corrupted
checkpoints. For the four-second batch-128 check, waveform relative L2 error
against replay was zero and gradient relative L2 error was 2.04e-10. The
benchmark separately checked filtered losses and physical-coordinate gradients
for batch 1/8/64/128; their four-second scaled-gradient relative L2 errors were
1.15e-9 to 1.36e-9. All errors are relative to the discrete implementation,
not an accuracy guarantee for the continuous model.

Every reverse segment checks its reconstructed start pair against the saved
forward boundary; singular, nonfinite or excessive boundary drift sets a flag
and the Python wrapper raises. This guard detects some invalid cases, but passing
it does not prove that all interior states or gradients are accurate. The method
remains research-only until further validation, especially near ill-conditioned
regions, large checkpoints and different configurations.

[Raw H100 measurements](../benchmarks/reconstruction_results/h100-k256.json) ·
[H100 numerical validation](../benchmarks/reconstruction_results/h100-validation.json) ·
[A100 coordinate validation](../benchmarks/reconstruction_results/a100-coordinates.json)

A100 validation job 34307838 passed the full suite. Coordinate job 34307952
passed filtered perceptual f0/B and material gradient checks, including finite
differences; full-duration relative gradient errors vs replay were 7.19e-10 and
2.66e-9 respectively. The H100 checkpoint-length sweep 34307844 and A100 paired benchmark 34308171
completed. The separately queued A100 checkpoint sweep is job 34308172.
Full raw outputs are on scratch in `reconstruction-benchmark-*` and
`reconstruction-validation-*` with Slurm job IDs. Run `hpc/reconstruction-check.json`
and `hpc/reconstruction-benchmark.json` through `hpc_sonic` with fresh state
paths. The benchmark checks source and test hashes against validation evidence
before timing and checks every benchmark case against the replay loss/gradients.

## A100 paired results

Job 34308171 completed on sbg19, NVIDIA A100 PCIe 40GB, under the same
four-second, 256-step conditions with three warmed repetitions:

| Batch | Replay: loss + backward | Reconstruction: loss + backward | Reduction |
|---:|---:|---:|---:|
| 1 | 2.199 s | 1.947 s | 11.4% |
| 8 | 2.240 s | 1.991 s | 11.1% |
| 64 | 2.409 s | 2.145 s | 11.0% |
| 128 | 3.275 s | 2.891 s | 11.7% |

Batch-128 synthesis alone took 0.942 s, and the full loss/backward peak was
8.19 GB for both paths. [Raw matched A100 results](../benchmarks/reconstruction_results/a100-k256.json).
The reconstruction path is experimental and explicitly fails its backward pass
if its checkpoint consistency or numerical guards are violated.

## Checkpoint-length sweep

Job 34307844 completed on the same H100, using the full four-second batch of
128 and three warmed repeats. All waveform, loss and coordinate-gradient checks
passed. Peak numbers include audio/loss allocations.

| Checkpoint length | Replay: loss + backward | Reconstruction: loss + backward | Reconstruction peak GPU allocation |
|---:|---:|---:|---:|
| 32 | 2.201 s | 2.032 s | 13.43 GB |
| 64 | 2.146 s | 1.966 s | 10.46 GB |
| 128 | 2.105 s | 1.922 s | 8.98 GB |
| 256 | 2.148 s | 1.905 s | 8.24 GB |
| 512 | 2.183 s | 1.896 s | 7.87 GB |
| 1024 | 2.192 s | 1.891 s | 7.68 GB |

Larger segments reduce checkpoint storage and launches for these parameters,
but their safety and accuracy outside the tested box remain unproven; these
results do not justify dropping the reconstruction boundary guard. [Raw data](../benchmarks/reconstruction_results/h100-checkpoints.json).

## Initial time-parallel investigation

The independent toy segment-map / suffix-scan proof of concept passed on CPU and
CUDA; it is not a production-scale performance result. See the
[parallel-segments idea](ideas/parallel_checkpoint_segments.md) and its toy report.
A separate full-grid prototype reconstructs groups of checkpointed segments in
parallel, **but its adjoint propagation between groups remains sequential**.
It passed the A100 full four-second batch-128 arbitrary-VJP test with relative
L2 error 2.35e-9 versus replay, plus sampled interior state, GPU-stream, guard,
finite-difference and odd-segment tests. [Validation](../benchmarks/reconstruction_results/a100-parallel-validation.json).

H100 job 34308237 measured short 0.25-second signals, three warmed repeats, 16
concurrently reconstructed segments, and loss/backward including the independent
CeL loss loop:

| Internal steps/segment | Batch | Replay | Sequential reconstruction | Parallel state reconstruction |
|---:|---:|---:|---:|---:|
| 10 | 1 | 0.11888 s | 0.11117 s | 0.10107 s |
| 10 | 8 | 0.13201 s | 0.12421 s | 0.11370 s |
| 256 | 1 | 0.10224 s | 0.09062 s | 0.08390 s |
| 256 | 8 | 0.11241 s | 0.10082 s | 0.09900 s |

The parallel method keeps more reconstructed states in GPU memory than the
fused sequential reconstruction path. [Raw smoke data](../benchmarks/reconstruction_results/h100-parallel-smoke.json).
Full four-second, batch-1/128 tests at 10/64/256 internal steps and 1/16/64
concurrent segments are submitted as H100 jobs 34308297–34308299. They record
unsafe reconstruction and OOM cases; no production-size parallel-gradient scan
has been demonstrated yet.

Job 34308297 then measured four seconds on H100 with the user's 10-step
segments, including full loss and backward, three warmed repetitions:

| Batch | Replay | Sequential reconstruction | Parallel state reconstruction (group 16) | Parallel state reconstruction (group 64) |
|---:|---:|---:|---:|---:|
| 1 | 1.851 s | 1.716 s | 1.543 s | 1.517 s |
| 128 | 2.453 s | 2.319 s | 2.360 s | 2.376 s |

At batch 128 these runs peaked at about 26.5 GB because the 10-step checkpoints
are frequent. The parallel state-only prototype helps the single string relative
to the same 10-step sequential method, but does not help this batch and its
single-string result is slower than the sequential 256-step configuration
measured on a different H100 node. Job 34308297 passed its numerical gates and
recorded an unchanged source hash. [Raw measurements](../benchmarks/reconstruction_results/h100-parallel-k10.json).
H100 job 34308298 tested the same four-second workload with 256 internal
steps per segment. Full loss and backward medians, three warmed repetitions:

| Batch | Replay | Sequential reconstruction | Parallel state reconstruction (group 64) |
|---:|---:|---:|---:|
| 1 | 1.561 s | 1.369 s | 1.242 s |
| 128 | 2.166 s | 1.922 s | 2.018 s |

The batch-128 parallel result is about 5% slower than sequential reconstruction;
the measured full-loss peak for both was about 8.24 GB. Numerical checks passed
for every group, and the batch-128 group-64 scaled gradient relative L2 error
versus replay was 1.48e-9. [Raw measurements](../benchmarks/reconstruction_results/h100-parallel-k256.json).
The 64-step H100 experiment, job 34308299, is running.

A100 80 GB PCIe job 34309733 tested the user's all-segments arrangement with
one four-second string and 10-step segments: all 52,920 segment blocks were
submitted in a single reconstruction grid (the GPU executes them in waves).
Full loss and backward took 2.499 s with replay, 2.345 s with sequential
reconstruction, and 2.176 s with parallel state reconstruction. Peak CUDA
allocation was approximately 0.26 GB for the first two and 1.08 GB for the
parallel version. The benchmark passed its waveform, loss, and gradient checks
against replay, with scaled gradient relative L2 error 4.10e-10. The boundary
adjoint still runs sequentially, so this is not an all-parallel gradient.
[Raw measurements](../benchmarks/reconstruction_results/a100-parallel-all-segments.json) ·
[validation](../benchmarks/reconstruction_results/a100-all-segments-validation.json).

The independent synthetic exact gradient-map/associative-scan prototype was also
scaled to 175 modes and one item for 256/1024/4096 steps on A100 and H100. It
passed comparison to PyTorch autograd for all tested configurations. This is
an eager PyTorch toy, not the physical full-string kernel, and its absolute
runtime is not a comparison against the optimized solvers. In the 4096-step,
10-step-segment test it used roughly 2.1 GB peak GPU memory. Extrapolating only
map storage—not runtime—to four seconds yields 53.2 GB for batch 1 at 10-step
segments, or 266 GB for batch 128 at 256-step segments, excluding scan
workspace. [A100 toy data](../benchmarks/parallel_scan_results/a100-175mode.json) ·
[H100 toy data](../benchmarks/parallel_scan_results/h100-175mode.json).

An optional 16,384-step, 175-mode synthetic scan test (A100 job 34308686)
passed for checkpoint lengths 10, 64 and 256 against eager PyTorch autograd.
At 10 steps it used 8.20 GB peak allocation; that setup is a highly damped
synthetic toy, and neither its runtime nor numerical conditioning is evidence
for a four-second physical string. [Raw result](../benchmarks/parallel_scan_results/a100-175mode-16384steps.json).
