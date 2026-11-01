# Recorded-alpha parallel state recovery

This is an experimental, first-order-only CUDA path for the existing float64
modal recurrence. It is separate from the default solver. The forward pass is
unchanged and records the nonlinear coupling scalar `alpha` at each internal
step. Conditional on the values recorded from that **same forward trajectory**,
each mode's two consecutive states satisfy an independent linear map:

```
[u_(t+1)]   [m * (A - Q * alpha_t)   m * P] [u_t    ]
[u_t    ] = [1                       0    ] [u_(t-1)]
```

Ordered associative products of these maps recover the primal states within
bounded checkpoint intervals in parallel. They use forward checkpoints, so
the recovery does not invert the damped dynamics. The original full nonlinear
discrete adjoint runs afterwards from the recovered states and differentiates
the coupling; it does **not** treat `alpha` as constant. Its propagation
between checkpoint intervals remains sequential: this is a state-recovery scan,
not a fully time-parallel gradient.

The CUDA prototype assigns one block per item, segment and mode and parallelises
up to 256 steps per segment. This can create many GPU blocks at large batch
sizes. Each bounded group retains a local state tape; the implementation
currently reuses the existing three-scalar forward cache, although the scan
needs only alpha. Its full memory and runtime are empirical questions.
Changing multiplication order changes FP64 roundoff and can amplify error;
every scanned segment checks its end state against the corresponding forward
checkpoint, and nonfinite values or excessive drift raise an error. Boundary
checks are not a proof that all interior states or gradients are accurate.

The independent CPU physical-recurrence check `test/alpha_scan_math.py` passed
for 4, 32 and 176 intervals, all three excitation types, and segment lengths 1,
7, 64 and 256. An initial stricter test observed an absolute discrepancy of
3.8e-13 in a state component, illustrating that the scan is not bitwise
identical. A100 validation job 34508178 passed 21 cases with source unchanged,
including sampled intermediate states, arbitrary VJPs against replay, four
seconds for one string, and nonfinite/boundary guards. In the four-second case,
waveform relative L2 error was zero and gradient relative L2 error was 2.33e-10
versus replay. [Validation results](../benchmarks/alpha_scan_results/a100-validation.json).
H100 validation job 34508539 also passed with source unchanged. [H100 validation](../benchmarks/alpha_scan_results/h100-validation.json).
Duplicate A100 validation jobs
34508540 and 34508910 were cancelled after A100 validation succeeded. The full
four-second batch-128 arbitrary-VJP A100 test, job 34509890, passed with waveform
relative L2 error zero and gradient relative L2 error 2.79e-9 against replay.
[Long-batch validation](../benchmarks/alpha_scan_results/a100-long-validation.json).
The A100 coordinate stress job 34510571 passed four-second perceptual and
material-coordinate filtered-loss gradient checks against replay and an
independent native forward Jacobian, plus short finite differences and
amplitude, damping and stability corner cases. [Stress results](../benchmarks/alpha_scan_results/a100-coordinate-stress.json).
The H100 full-batch VJP test, job 34509891, passed with unchanged source and
gradient relative L2 error 2.79e-9 against replay. [H100 long-batch validation](../benchmarks/alpha_scan_results/h100-long-validation.json).
The matched full-loss benchmark harness
`benchmarks/alpha_scan_compare.py` additionally gates each measured case against
the replay waveform, loss and six-coordinate gradients, bounds predicted
long-case runtime, and reports skipped or failed cases explicitly. Benchmark
jobs 34509878 (A100) and 34509877 (H100) were submitted after validation.

## Full four-second batch-128 result

SAE A100 80 GB PCIe job 34510679 compared the full forward simulation,
anti-aliasing, grouped CeL loss (groups of 16) and all six coordinate gradients,
with three warmed repetitions. Segment size was 256 and each scan group held
eight segments. All measurements used the same inputs and GPU:

| Method | 0.25 s audio | 4 s audio | 4 s peak PyTorch allocation |
|---|---:|---:|---:|
| Replay adjoint | 0.204 s | 3.058 s | 3.33 GB |
| Cached reconstruction | 0.138 s | 2.007 s | 4.95 GB |
| Recorded-alpha scan | 0.273 s | 4.180 s | 4.95 GB |

The scan was slower than cached reconstruction in this configuration. Its
measured full-loss allocation matched cached reconstruction because other
allocations determined the peak; this does not mean its state tape is free.
The 4 s scan's waveform and loss relative L2 errors versus replay were zero,
and its scaled gradient relative L2 error was 1.48e-9. Both the 0.25 s and
4 s numerical gates passed, and the source fingerprint remained unchanged.
[Raw A100 measurements](../benchmarks/alpha_scan_results/a100-fast-b128.json).
This result concerns six coordinates and the current state-only scan, not a
fully time-parallel gradient. H100 and other group sizes were measured separately.

A separate A100 40 GB job 34509878 completed with source unchanged and all 12
numerical gates passing across 0.25/4 s, batches 1/8/128 and groups 8/16.
For 4 s and group 16, scan versus cached reconstruction took 1.866 versus
1.410 s (B1), 2.039 versus 1.443 s (B8), and 4.661 versus 2.027 s (B128).
These are full grouped-loss/six-gradient times, not synthesis-only figures.
[Raw sweep](../benchmarks/alpha_scan_results/a100-sweep.json).

H100 job 34509877 completed all 12 checks with source unchanged. For 4 s and
group 8, scan versus cached reconstruction took 1.327 versus 0.968 s (B1),
1.378 versus 1.003 s (B8), and 2.432 versus 1.374 s (B128). B128 peak
allocation was 5.001 GB for each; replay took 2.011 s / 3.375 GB. The group-16
scan took 2.773 s at B128. [H100 full sweep](../benchmarks/alpha_scan_results/h100-sweep.json).

For the selected f0/B case, A100 40GB job 34510937 passed all 12 waveform,
loss and selected-gradient checks, including verifying zero gradients for
unselected coordinates. At four seconds and batch 128 with grouped loss it
measured 2.162 s / 3.665 GB for genuine two-direction forward sensitivities,
2.027 s / 4.953 GB for cached reconstruction, 4.251 s / 4.953 GB for the
three-scalar alpha scan, and 3.072 s / 3.327 GB for replay. The first-order
adjoint kernels internally form their usual coefficient gradients even when
the outer coordinates are masked. [Raw selected-coordinate results](../benchmarks/alpha_scan_results/a100-two-directions.json).
H100 selected-coordinate job 34510936 also passed all 12 checks with unchanged
source. At four seconds and batch 128, genuine seeded-two took 1.776 s / 3.715
GB, cached reconstruction 1.373 s / 5.003 GB, alpha scan 2.438 s / 5.003 GB,
and replay 2.011 s / 3.377 GB in the same grouped-loss job. This differs from
the older 1.589-second seeded measurement, which used a separate job with
ungrouped losses; do not treat it as a same-job comparison.
[H100 selected-coordinate data](../benchmarks/alpha_scan_results/h100-two-directions.json).

A separate alpha-only prototype in `experiments/parallel_alpha_scan_one`
uses a one-scalar forward cache to reduce tape storage. A100 validation job
34512081 passed 21 cases with source unchanged, including four-second B1
waveform and arbitrary-VJP checks against replay (gradient relative L2 error
2.33e-10), sampled intermediate states and corrupted/nonfinite guards.
[Validation](../benchmarks/alpha_scan_results/a100-one-validation.json).
The matched all-six-parameter full-loss H100 benchmark (job 34513052) completed
with unchanged source fingerprints. All four waveform, grouped-loss and
gradient checks passed. Four seconds, batch 128, group 8 gave 2.417 s / 3.919 GB
peak PyTorch allocation for the alpha-only scan, compared with 1.373 s / 5.003 GB
for the three-scalar cached reconstruction and 2.009 s / 3.377 GB for replay.
The scan's waveform and loss relative L2 errors were zero and scaled gradient
relative L2 error 1.36e-9 against replay. The scan trades speed for lower
allocation here; it is not a fully parallel adjoint. Batch 1 took 1.321 s for
the scan versus 0.967 s for cached reconstruction. All timings include the
grouped loss, all six gradients, and three warmed repetitions; compilation and
optimizer updates are excluded. [Raw H100 data](../benchmarks/alpha_scan_results/h100-one-benchmark.json).
A100 80 GB job 34513051 also completed with unchanged source and all four
numerical checks passing. For 4 s / B128, alpha-only scan took 4.084 s / 3.869
GB, cached reconstruction 2.006 s / 4.953 GB, and replay 3.058 s / 3.327 GB.
Thus the one-scalar scan reduced peak allocation by about 1.08 GB relative to
the three-scalar cached method, while taking approximately twice as long.
[A100 data](../benchmarks/alpha_scan_results/a100-one-benchmark.json). Separate
A100 40 GB repeat 34516000 passed all four checks with unchanged source: the
4 s / B128 alpha-only scan took 4.123 s / 3.869 GB versus cached 2.026 s /
4.953 GB and replay 3.071 s / 3.327 GB. [A100 40 GB data](../benchmarks/alpha_scan_results/a10040-one-benchmark.json).
Duplicate pending job 34513388 was cancelled.
