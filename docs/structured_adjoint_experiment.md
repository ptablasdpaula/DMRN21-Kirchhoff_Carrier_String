# Structured time-segment adjoint experiment

This is an isolated first-order FP64 research prototype of the user's proposed
parallel time-segment gradients. It retains the original nonlinear recurrence,
all modes and the boundary-adjoint dependency. It performs the forward pass with
the existing CUDA replay solver and processes a bounded window of segments at a
time in reverse order; it is not a production backend. Segment state recovery
uses the guarded end-anchored inverse and is not the separate alpha-scan state
recovery experiment.

## Algebra

For one mode vector let `p,u` be the two prior states, `q=Q*u`,
`l=A*u+P*p`, `alpha=(q·p + m*q·l)/(1+m*q·q)`, and `v=m*(l-q*alpha)`.
The state update maps `[p,u]` to `[u,v]`. Its state Jacobian is a 2×2
block-diagonal matrix over modes plus one outer product: the mode-local terms
are `dp=m*P` and `du=m*(A-alpha*Q)`, and the global update is
`[0,-m*q] [dalpha/dp,dalpha/du]^T`. Taking the transpose gives the matching
adjoint block-plus-rank-one form.

An exact `K`-step segment map and the loss-induced affine bias can therefore
be represented as `M=D+U V^T`, without any rank truncation. When composing
maps `(D1,U1,V1,b1)` and `(D2,U2,V2,b2)` in that order, the block-diagonal
part is `D1 D2`, the new left factors are `[D1 U2, U1]`, the new right factors
are `[V2, D2^T V1 + V2(U2^T V1)]`, and the new bias is `b1+M1*b2`.
No block inverse is used. An ordered associative suffix scan computes each
segment's correct ending adjoint, including the arbitrary per-sample loss
forcing. Parameter VJPs are then evaluated concurrently over segments in the
window; their initial-condition gradients are chained to material parameters.
Windows remain sequential, and the initial nonlinear forward pass is sequential.

## Storage and numerical limits

For a window of four 64-step segments and batch 128, just the final pair of
untruncated `U,V` factors has about 0.734 GB of FP64 values; intermediate
scan tensors, coefficients, states, checkpoints, output and loss add to this.
Rank grows under composition and may exceed the state dimension: the prototype
does not truncate it. Materialising the full four-second history this way is
not feasible; windowing bounds the working set. This eager-PyTorch prototype
launches many operations and synchronizes to perform safety checks, so neither
its timings nor a short-run extrapolation are a CUDA kernel performance claim.

No inverse-reconstruction boundary check establishes correctness of every
interior state or parameter gradient. The implementation rejects singular,
nonfinite or drifting segment states and tests segment-map adjoints against a
second per-step reverse calculation. Both checks test every segment and batch
item separately, so a loud item cannot hide a quiet item's failure. A partial
final segment is handled by
identity maps and zero loss forcing, including the 32- and 64-step cases for
529,200 internal steps.

## Tests and scheduled GPU work

The independent CPU suite compares structured maps and gradients to PyTorch
autograd, including singular diagonal block compositions, K=16/32/64, windows
1/2/3/4, unequal final windows, partial final segments, synthetic and physical
coefficients, and a 175-mode production-parameter one-step Jacobian. The
175-mode local Jacobian relative L2 error was 1.03e-16; the 175-mode 32-step
parameter VJP relative L2 error was 2.54e-13. A separate dense segment-map
scan agreed on the initial-state and parameter VJPs to 8.92e-16 and 7.66e-16
relative L2 respectively. Mixed-loudness corruptions that pass a global norm
were rejected by both per-item guards. A separate CPU test of the physical
175-mode, 0.03-second recurrence, 3× anti-alias filter and CeL loss matched
full PyTorch autograd with material-gradient relative L2 error 8.97e-10.
These numerical results do not
establish full-duration or GPU accuracy.

The GPU jobs compare the physical recurrence and the full material-parameter
VJP directly against the existing CUDA replay adjoint, including initial-state
derivatives and selected nonzero initial-output gradients. They time only the
prototype's simulator VJP (not filtering or CeL) on 4-segment windows, measure
peak PyTorch allocation, and attempt a four-second run only if the measured
short-case runtime predicts at most 180 seconds. The follow-up four-second worker
selects the fastest eligible chunk for each batch and measures the complete
synthesis/filter/CeL/gradient evaluation for both two and six coordinates. A
skipped case is explicitly labelled; any linear extrapolation is a scheduler
guard, not a measured result.
Jobs: A100 `34510044` (gpushort), `34510043` (andrena), H100 `34510042`; A100
single-item fast backfill candidates `34510045` (gpushort) and `34513378` (sae).
Earlier pending jobs were
cancelled before the per-item guard fix. The first A100 full worker
`34510044` completed, after which its duplicate A100 jobs were cancelled; H100
`34510042` also completed. Both recorded an unchanged source fingerprint.

## A100 short physical-grid results

Job 34510044 used an A100 40 GB with 175 modes, 132.3 kHz and float64. All
CPU-derived and CUDA physical tests, singular/nonfinite and mixed-loudness
guard checks passed. For each case below, a window of four segments was checked
against the CUDA replay adjoint, then the structured simulator VJP was timed
twice. These timings **exclude synthesis, filtering and loss** and are only
for the stated number of internal steps.

| Batch | K | Steps | Two VJP timings | Peak allocation | Gradient relative L2 vs replay |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 64 | 0.0452, 0.0446 s | 0.022 GB | 4.25e-13 |
| 1 | 32 | 128 | 0.0873, 0.0855 s | 0.026 GB | 5.59e-12 |
| 1 | 64 | 256 | 0.1716, 0.1675 s | 0.035 GB | 7.61e-12 |
| 128 | 16 | 64 | 0.0473, 0.0458 s | 0.582 GB | 1.11e-12 |
| 128 | 32 | 128 | 0.0885, 0.0888 s | 1.101 GB | 4.50e-12 |
| 128 | 64 | 256 | 0.1849, 0.1822 s | 2.135 GB | 1.43e-11 |

An extrapolation of these short eager-PyTorch times is approximately 355–391
seconds per four-second VJP, which exceeds the predeclared 180-second guard.
Consequently the four-second full-loss tests for both batches were **skipped**,
not measured. Extrapolation is only a reason for the skip: it is not a measured
duration or a prediction for an optimized CUDA implementation. The source hash
was unchanged. [Short physical data](../benchmarks/structured_adjoint_results/a100-physical-short.json)
· [full-loss skip record](../benchmarks/structured_adjoint_results/a100-full-loss-gate.json).

## H100 short physical-grid results

Job 34510042 ran the same short VJP tests on an H100 80 GB; all waveform and
gradient gates and mixed-loudness, nonfinite and singular-mass checks passed.
The short structured VJP timings exclude synthesis, filtering and loss:

| Batch | K | Steps | Two VJP timings | Peak allocation | Gradient relative L2 vs replay |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 64 | 0.0250, 0.0243 s | 0.072 GB | 2.24e-12 |
| 1 | 32 | 128 | 0.0483, 0.0461 s | 0.076 GB | 4.53e-12 |
| 1 | 64 | 256 | 0.0897, 0.0894 s | 0.085 GB | 1.09e-11 |
| 128 | 16 | 64 | 0.0336, 0.0335 s | 0.631 GB | 9.40e-13 |
| 128 | 32 | 128 | 0.0671, 0.0671 s | 1.150 GB | 3.45e-12 |
| 128 | 64 | 256 | 0.1476, 0.1476 s | 2.185 GB | 1.07e-11 |

The predeclared runtime guard skipped full-loss tests based on the 185–305 s
short-run extrapolations; these are not measured four-second durations. Source
hash was unchanged. [Short physical data](../benchmarks/structured_adjoint_results/h100-physical-short.json)
· [full-loss skip record](../benchmarks/structured_adjoint_results/h100-full-loss-gate.json).

After the user requested actual four-second runs despite the short-window
estimate, K32/window4 full-loss workers were submitted: A100 batch 1 `34515404`
and batch 128 `34515406` (gpushort) / `34516015` (sae backfill); H100 batch 1
`34515405` and batch 128 `34515407`.
Each runs one timed full evaluation after a short GPU warmup, checks waveform,
per-item losses and all six gradients against replay, records partial stage
information and has a 40-minute process timeout. A timeout, failed check or
incomplete JSON is not a valid result. These are one-sample tests, not warmed
multi-repetition medians.

The A100 batch-1 worker `34515404` completed: the actual four-second full
synthesis/filter/CeL/all-six-gradient evaluation took **345.209 s** with the
eager structured candidate versus **2.433 s** for replay. These are single
measurements after short warmup, not repeated warmed medians. Waveform and
loss errors were zero in this check, scaled gradient relative L2 error was
1.12e-7, and peak PyTorch allocation for the candidate was 0.157 GB. Source
was unchanged. The large slowdown describes this eager prototype and its many
synchronizations, not a fundamental limit or an optimized CUDA implementation.
[Raw full-duration measurement](../benchmarks/structured_adjoint_results/a100-full-k32-b1.json).
The A100 80 GB batch-128 worker `34515406` also completed: structured took
**366.656 s** versus replay **4.112 s**, with 13.375 GB peak allocation;
waveform and loss errors were zero and the scaled six-gradient relative L2
error was 3.10e-7. [Raw batch-128 result](../benchmarks/structured_adjoint_results/a100-full-k32-b128.json).

H100 batch-1 worker `34515405` completed: structured took **190.117 s**
versus replay **1.755 s**, with 0.207 GB peak allocation; waveform and loss
errors were zero and the scaled six-gradient relative L2 error was 3.57e-8.
[Raw H100 result](../benchmarks/structured_adjoint_results/h100-full-k32-b1.json).
The H100 batch-128 worker `34515407` completed: structured took **277.137 s**
versus replay **2.348 s**, with 13.425 GB peak allocation. Waveform and loss
errors were zero and the scaled six-gradient relative L2 error was 1.61e-7.
[Raw H100 batch-128 result](../benchmarks/structured_adjoint_results/h100-full-k32-b128.json).
All four are single evaluations after short warmup, and their source
fingerprints were unchanged. The duplicate SAE A100 batch-128 backfill
`34516015` was cancelled after the gpushort A100 worker started.
