# Scalar-tape reconstruction sweep

The isolated `experiments/scalar_cache` CUDA experiment implements the same
FP64 discrete update and analytical adjoint with 0, 1, 2 or 3 stored scalars
per internal step. Zero uses the previously validated reconstruction backend.
One stores the forward coupling alpha; two store alpha and the squared coupling
norm; three also store the coupling/linear cross term. Missing terms are
reduced from reconstructed states. No modes or loss terms are removed.

For four seconds and batch 128, the ideal tape sizes are 0, 0.542, 1.084 and
1.626 GB respectively, before checkpoints, loss and other allocations. The
three-scalar variant minimizes per-step reduction width but is not assumed to
be fastest overall. The alternatives retain inverse/boundary guards; passing a
boundary check is not sufficient to guarantee accurate interior gradients.

## GPU checks

The independent CPU [one-step algebra test](../test/scalar_cache_algebra.py)
passed 96 random FP64 cases for all three nonzero cache widths against PyTorch
autograd, including the complete state and coefficient derivatives.

A100 40 GB smoke job 34508079 completed with exit code 0 and passed 0/1/2/3 variants on zero/short signals, multiple
grid sizes and batches, and 4-second B1/B128 arbitrary VJPs. On B128 at 4 s,
one/two/three-scalar waveform relative L2 error was zero, and gradient relative
L2 error against replay was 2.42e-10. The source checksums were unchanged. [Log](../benchmarks/scalar_cache_results/a100-smoke-34508079.log).

Additional A100 coordinate and safety checks in job 34508377 passed for all
three nonzero cache widths. They covered four-second filtered CeL gradients in
both perceptual and material coordinates against replay and an independent CPU
forward Jacobian; short directional finite differences; low/high amplitude,
undamped, high-damping and stability-corner cases. Results for cache widths
[1](../benchmarks/scalar_cache_results/a100-coordinates-1.json),
[2](../benchmarks/scalar_cache_results/a100-coordinates-2.json) and
[3](../benchmarks/scalar_cache_results/a100-coordinates-3.json).

The optional whole-batch fallback regenerates checkpoints from the initial
state after a detected inverse safety error, rejects inconsistent checkpoints
and nonfinite data, then applies the exact replay VJP. Forced fallback and
injected nonfinite-cache tests passed, as did rejection of corrupted checkpoints
and nonfinite upstream gradients. It is not a guarantee against undetected
interior error or finite scalar-cache corruption.

For four-second batch-128 all-six-coordinate full loss on A100 40 GB, the
same job measured replay at 3.265 s, normal three-scalar reconstruction with
the fallback wrapper at 2.265 s, and **forced fallback** at 4.248 s. The extra
forward pass makes fallback a robustness option rather than a speedup. All
three warmed repetitions and the gradient checks are in the
[fallback measurements](../benchmarks/scalar_cache_results/a100-fallback-compare.json).

The main A100/H100 full-loss sweeps are 34508378/34508379. A grouped-loss
helper was edited during the A100 sweep, after its process imported the prior
implementation. The supported common-target semantics did not change, but its
helper source was not fingerprinted at job start, so grouped timings from that
job require a repeat with frozen, fingerprinted dependencies. The H100 worker
started after the edit; a helper checksum was saved during its run and matched
on completion. Inspect Slurm status and output before claiming
timings; queued or failed jobs are not measurements. The experimental helpers
are not production backends.

### A100 preliminary full sweep

Job 34508378 completed with exit 0 and 144/144 numerical checks passed on an
A100 40 GB. For batch 128, four seconds, all six coordinate gradients and
grouped loss (group 16), full synthesis/loss/backward medians and peak PyTorch
allocations were:

| Saved scalars | Time | Peak allocated |
|---:|---:|---:|
| 0 | 2.687 s | 3.325 GB |
| 1 | 2.235 s | 3.866 GB |
| 2 | 2.151 s | 4.409 GB |
| 3 | 2.064 s | 4.950 GB |

The matching forward-sensitivity baseline was 3.100 s / 5.830 GB. Waveform,
individual loss and scaled coordinate gradients for each candidate were
compared with replay; all batch-128/four-second cache cases had scaled-gradient
relative L2 error at most 3.15e-8 and per-item relative error at most 4.04e-7.
The grouped-source provenance caveat above applies to these timing numbers.
An isolated repeat that fingerprints all helpers, job 34508690, timed out while
compiling on a cold node after four checks and produced no timings. Its
[partial JSON](../benchmarks/scalar_cache_results/a100-fast-partial-timeout-34508690.json)
is preserved; the repeat was resubmitted with a longer allocation as 34516507.
[Raw A100 data](../benchmarks/scalar_cache_results/a100-sweep-34508378.json).

### H100 full sweep

Job 34508379 completed with exit 0 on H100 HBM3 and passed all 144 checks.
For 128 four-second strings, all six coordinate gradients and grouped losses
(group 16), the same-job results were:

| Method | Time | Peak allocated |
|---|---:|---:|
| Forward sensitivities | 1.963 s | 5.880 GB |
| Replay adjoint | 2.016 s | 3.375 GB |
| Reconstruction, no scalar tape | 1.768 s | 3.375 GB |
| One scalar | 1.520 s | 3.917 GB |
| Two scalars | 1.485 s | 4.459 GB |
| Three scalars | **1.418 s** | 5.000 GB |

The two-coordinate grouped three-scalar result was 1.418 s / 5.000 GB. The
reconstruction path computes internal material adjoints before returning only
the selected coordinates, so its time is nearly the same for two or six. The
one-scalar variant is 22.6% faster and 33.4% smaller in peak PyTorch allocation
than the same-job six-coordinate grouped forward-sensitivity baseline; the
three-scalar variant is 27.7% faster and 15.0% smaller. The previously saved
1.589-second f0/B H100 result came from a different job and node, so the
cross-job difference is not a paired comparison.

Across the four-second batch-128 cache candidates, the maximum scaled gradient
relative L2 error against replay was 3.77e-8 and the maximum per-item relative
error was 2.82e-7. The candidate and benchmark source hashes were unchanged.
The grouped helper was fingerprinted shortly after this job started, and its
checksum remained unchanged through completion. [Raw H100 measurements](../benchmarks/scalar_cache_results/h100-sweep-34508379.json).

The stricter repeat, job 34508689, completed with exit 0 and fingerprints for
all eight candidate, helper, benchmark and script files unchanged from start to
finish. All 12 full-duration batch-128 comparisons passed. Grouped all-six
results were 1.5195 s / 3.917 GB for one scalar, 1.4847 s / 4.459 GB for two,
and 1.4181 s / 5.000 GB for three; the matching seeded baseline was 1.9634 s /
5.880 GB. The two-coordinate three-scalar result was 1.4179 s. Across all cache
variants in the repeat, the maximum scaled-gradient relative L2 error was
3.37e-8 and maximum per-item relative L2 was 3.38e-7 against replay. These
are observed discrepancies, not universal bounds. [Raw strict repeat](../benchmarks/scalar_cache_results/h100-fast-34508689.json).

To reproduce after allocating through the existing cluster helper, use a fresh
scratch submission state, for example:

```sh
python ../hpc_sonic/hpc_sonic.py hpc/scalar-cache-sweep.json \
  --directory "$PWD" --pools gpushort --constraint ampere --max-gpus 1 \
  --submit --state /data/scratch/acw794/kc-string-gpu/scalar-cache-sweep-new.json
```

The worker saves incremental JSON under scratch; the harness records source
hashes, individual synchronized timings, peak PyTorch allocated bytes and
per-case comparisons against replay. Group size 0 denotes the original
individual-loss path and group size 16 uses the experimental grouped CeL.
