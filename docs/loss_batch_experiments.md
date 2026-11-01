# Grouped CeL loss experiment

The existing batched benchmark calculates and retains 128 independent CeL loss
graphs, then calls backward once. This experiment processes independent items in
small groups, computes each group's waveform gradient immediately, and calls
the FDTD backward once with the concatenated waveform gradients. The existing
bound CeL uses mean reduction; multiplying each group by its number of items
preserves the sum of the original per-item losses, including a short final
group. The nonlinear FDTD, 3× anti-alias filter, CeL and first-order parameter
gradients are unchanged up to floating-point summation order. It is a research
helper, not yet integrated into a public batched optimizer; per-item losses
would need to be returned if an optimizer needs them. It explicitly accepts
only a single common bound target and mean reduction; batches with different
targets must be sliced correctly before that restriction can be relaxed.

## A100 40GB PCIe, 4 seconds, batch 128

Slurm job 34504610 ran three warmed, synchronized repetitions per case, FP64,
175 modes, 132.3 kHz internal / 44.1 kHz output, varied inputs, a common
reference target and individual STFT cumulative-energy losses summed for
backward. `peak` means PyTorch CUDA allocated bytes / 1e9; it is not total
device memory. Same GPU, process, solver, data and loss for each comparison.
Compilation, target creation and optimizer updates are excluded.

| Fitting coordinates | Loss calculation | Time | Peak PyTorch allocation |
|---|---|---:|---:|
| f0, B | 128 individual CeL graphs | 2.362 s | 8.527 GB |
| f0, B | Groups of 8 | 2.185 s | 3.463 GB |
| f0, B | Groups of 16 | 2.164 s | 3.663 GB |
| All six | 128 individual CeL graphs | 3.299 s | 10.694 GB |
| All six | Groups of 8 | 3.124 s | 5.630 GB |
| All six | Groups of 16 | 3.102 s | 5.830 GB |

For each group/direction case, loss and all parameter gradients were directly
compared against the ungrouped implementation on all 128 items before timing.
Maximum observed scaled parameter-gradient relative L2 difference was 4.59e-15;
maximum relative loss difference was 1.48e-16. This checks the regrouping
against the same seeded simulator, not the accuracy of the simulator itself.
The job completed with exit code 0. [Raw data](../benchmarks/loss_batch_results/a100-fast-34504610.json).

An additional CPU comparison covered group sizes 1, 2, 3, 7 and 8 for seven
independent items; loss matched and maximum absolute waveform gradient
difference was 1.09e-19. A second A100 wider group/batch sweep, job 34504561,
completed with 48 checks passed. For the four-second batch of 128, its two-
coordinate baseline was 2.357 s / 8.529 GB; groups of 8, 16 and 32 took
2.184 / 2.163 / 2.153 s at 3.463 / 3.665 / 4.769 GB respectively. For all
six coordinates, the baseline was 3.296 s / 10.697 GB; groups of 8, 16 and 32
took 3.122 / 3.101 / 3.089 s at 5.630 / 5.833 / 6.936 GB. [Raw sweep](../benchmarks/loss_batch_results/a100-sweep-34504561.json).
H100 job 34504552 subsequently completed its full sweep; all 48 grouped-loss
checks passed. At batch 128, four seconds and group 16, f0/B took 1.776 s and
3.715 GB versus 1.924 s and 8.579 GB with independent losses. All six took
1.963 s and 5.883 GB versus 2.111 s and 10.747 GB. Each comparison used the
same GPU job and three warmed repetitions. [H100 raw data](../benchmarks/loss_batch_results/h100-sweep-34504552.json).
The faster earlier 1.589 s H100 measurement was obtained in a different job
and harness, and is not the baseline of these grouped comparisons. Historical
implementation SHA256:
`b8c3c5b6b30ee6a428c286461f33129113cd2ebdb7a600b4f70a4d5308b5b5b5`.
Benchmark script SHA256:
`205aa05724034eeb480c29ba45c36e47b748ec4f46ae637bffab73cd6cbc50b4`.

## Explicit directional loss VJP

The experimental `manual_ce_loss.py` computes the exact first-order default
CeL waveform VJP explicitly, one cumulative direction at a time. It supports
only the unweighted diagonal, non-decaying, uncentered default CeL and is not
a public replacement for general CeL. It explicitly requires a single common
bound target; per-item targets are rejected. The STFT is still differentiated by
PyTorch, and the existing FDTD backward runs once after all loss groups.

In A100 40 GB job 34508132, batch 128, four seconds, group 16 and three warmed
repetitions, the cached-adjoint path took 2.0095 s with the explicit loss VJP
versus 2.0254 s with grouped autograd. Peak PyTorch allocations were 5.068 GB
versus 4.953 GB respectively: the explicit version was slightly faster but
used more memory. For seeded f0/B, the corresponding times were 2.1442 s vs
2.1625 s and peaks were 3.781 vs 3.666 GB. These results do not demonstrate an
improvement on both axes or any H100 speedup. [Measurements](../benchmarks/loss_batch_results/a100-manual-ce-34508132.json).

The A100 run checked full 4 s batch-128 losses and scaled parameter gradients
against grouped autograd for f0/B and all-six seeded, and all-six cached
adjoints, at group sizes 4, 8 and 16. The largest observed relative scaled
parameter-gradient difference was 1.46e-9. Independent CPU/GPU waveform-VJP
tests included random, near-target, exact-target and tiny signals and uneven
group sizes. [GPU tests](../benchmarks/loss_batch_results/a100-manual-ce-validation-34508132.json).
An additional A100 job 34510230 passed nine direct four-second waveform-VJP
comparisons for B1/B8, exact/near-target and varied amplitudes, with maximum
relative L2 difference 1.73e-16. [Long GPU checks](../benchmarks/loss_batch_results/a100-manual-ce-long-34510230.json).
The implementation is first-order-only and retained as an experiment.

On H100 job 34510183, the same four-second B128 cached-adjoint group-16
workload took 1.3664 s / 5.117 GB with the explicit VJP versus 1.3745 s /
5.002 GB with grouped autograd. Three warmed repetitions and all nine
candidate loss/gradient checks passed; the worst scaled-gradient relative L2
difference was 1.29e-9. GPU contract tests included rejecting distinct
per-item bound targets, which the experimental helper does not support. The
small time gain costs approximately 0.115 GB and does not justify claiming an
improvement on both axes. [H100 result](../benchmarks/loss_batch_results/h100-manual-ce-34510183.json) ·
[validation](../benchmarks/loss_batch_results/h100-manual-ce-validation-34510183.json).

## A100 profiling

Job 34508281 profiled the grouped-loss workload at four seconds and batches 1
and 128. On batch 128, the cached simulator's forward chunks accounted for
approximately 0.95 s of GPU kernel time and reconstruction chunks approximately
0.92 s; 128 cumulative-sum operations accounted for approximately 0.026 s.
These are profiler-attributed kernel times, not independent end-to-end timing
measurements, and grouped/nested event totals must not be summed indiscriminately.
They indicate that, after loss grouping, the FDTD recurrence remains the major
cost. [Profile data](../benchmarks/loss_batch_results/a100-profile-grouped-34508281.json).

H100 job 34508280 profiled the same four-second batch-128 cached workload.
Forward and reconstruction chunks accounted for approximately 0.52 s and
0.46 s of GPU kernel time, respectively, and convolution backward accounted
for approximately 0.33 s. These are overlapping profiler attributions, rather
than separate end-to-end measurements. They suggest that both the string
recurrence and the output filter deserve attention in future optimization.
[H100 profile data](../benchmarks/loss_batch_results/h100-profile-grouped-34508280.json).
