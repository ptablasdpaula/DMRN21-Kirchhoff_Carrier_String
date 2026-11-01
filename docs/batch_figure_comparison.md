# Batch-size time and GPU-memory figure

This is an experimental benchmark, not a production solver. It measures batch
sizes 1, 2, 4, 8, 16, 32, 64 and 128 in one A100 80 GB allocation, with each case
isolated in a fresh process. The output prefix contains incremental `.json`,
`.csv`, `.md`, `.png` and `.pdf` files; the final JSON says `complete` only
after all cases have been attempted. A partially populated figure is not a
completed comparison. If source fingerprints change during the run the JSON
is marked invalid and the figure and tables are removed.

* Direct FDTD is uncheckpointed spatial PyTorch reverse-mode autodiff. It is
  intentionally the simple implementation, including Python time-step and
  GPU-kernel launch overhead.
* The proposed path is the exact-discrete CUDA reconstruction with three
  saved scalars; one scalar is measured and included in the table only.
* The modal comparator calls Díaz and Sandler's actual `jaxdiffmodal` release
  at revision `6980979dfb8a07b8664bddf7cf3e3d2f097d5ff0`, including the
  upstream transfer-function solver and its full modal trajectory. It uses
  continuous modal eigenvalues and a different time integrator, so it is not
  the same discrete FDTD calculation. Per-case waveform and loss differences
  against FDTD are diagnostics, not equality tests.

> **Superseded for the DMRN+21 abstract.** The one-page draft now uses the
> one-second raised-cosine native-rate sweep (`hpc/native-rate-sweep.sh`,
> Modal at 100 modes and 44.1 kHz) and the accuracy study in
> `docs/accuracy_convergence_study.md`. The description below records the
> earlier four-second, eight-mode campaign.

All methods use four seconds, FP64, 44.1 kHz output, 3× oversampling, the same
175-mode/grid resolution, multimode excitation, physical parameters, pickup,
common target and gradients with respect to all eight material coordinates:
**density, tension, radius, Young's modulus, length, amplitude, sigma0 and
sigma1**. The test strings initially have density 8050 kg/m³ and length 1 m,
but both are independent differentiable inputs. Case values are constructed
once from the existing deterministic strings before timing, then all solvers
consume the same physical coordinates directly; no f0/B chain rule is part of
the benchmark.
Time is the median of three
synchronized warmed evaluations of synthesis, anti-alias filtering, CeL and
backward; the target, compilation and setup are outside the timer. Per-case
preparation, validation, warm-up and repetitions together have a 30-minute
limit. The default loss uses batches of up to 16 items and is summed per item.

GPU memory is the maximum NVML per-process memory sampled every 10 ms during
the measured repetitions, including context and allocator reservations. JAX
preallocation is disabled. It is the same metric for all methods, but can miss
short transient peaks. Framework-native memory metrics are recorded separately
and should not be substituted for the cross-framework values. Host RSS has a
32 GB per-case safety limit to keep a massive autodiff graph from killing the
entire job; this is reported separately from actual GPU out-of-memory. OOM,
host-limit, timeout and errors appear in the table, never as plotted values.

Before the full campaign, the direct solver was checked on CPU against the
independent dual-number spatial implementation for all eight physical
gradients. The Díaz adapter was checked against a separately assembled
upstream solve, finite differences in all eight physical parameters, and
grouped-loss values and gradients. Earlier A100 0.03-second batch-1/2 GPU
pilots used six perceptual gradients; they are not used in the full figure.
Subsequent [A100 preflight job 34543406](../benchmarks/figure_comparison_results/a100-eight-smoke.json)
passed the direct CUDA replay check and completed all four methods at batch 1,
0.03 seconds and eight physical gradients. These short preflight timings also
do not belong in the four-second figure.
The [final Díaz diagnostic preflight](../benchmarks/figure_comparison_results/a100-diaz-eight-smoke.json)
also completed at batches 1 and 2 and records waveform, loss and physical
gradient differences against FDTD.

The `diaz-deps` scratch directory contains the pinned upstream package plus
its pure-Python dependencies; its `direct_url.json` and module hashes are
recorded. To repeat on the cluster after provisioning the existing CUDA/JAX
environments and those dependencies, submit through `hpc_sonic`:

```sh
python ../hpc_sonic/hpc_sonic.py hpc/figure-comparison.json \
  --directory "$PWD" --pools sae --constraint 'a100&80G' --memory 64G --max-gpus 1 \
  --submit --state /data/scratch/acw794/kc-string-gpu/figure-comparison-new-submission.json
```

`benchmarks/figure_comparison_driver.py` stores a log for each case, source
fingerprints, measured repetitions, validation data and failures. Re-render a
valid completed or partial JSON with `benchmarks/figure_comparison_plot.py`.

## Current submission

The single-device eight-physical-gradient A100 80 GB campaign is Slurm job
`34545711`, submitted through `hpc_sonic`. Its files will be written under
`/data/scratch/acw794/kc-string-gpu/figure-comparison-34545711` with the
extensions listed above. A short-queue A100 40 GB job `34545856` separately
measures both cached variants at all sizes on a single card. The A100 GPU
preflight is job `34543406`. Do not mix the 40 GB and 80 GB results in one
figure: available memory changes allocation and out-of-memory behavior.
Check Slurm and each JSON's `metadata.status` and row statuses before using
the figure; submission and partial output do not imply it has finished.

For the proposed short-queue campaign, `hpc/figure-comparison-shard.sh` runs
one method/batch case per job, except that the two cached variants share one
allocation across all sizes. `benchmarks/figure_comparison_tasks.py` creates a
candidate `hpc_sonic` task manifest. Each case records the NVML GPU model and
UUID when observable. `benchmarks/figure_comparison_merge.py` checks the
workload, source hashes and GPU model before combining shards, and reports all
device UUIDs. Use `--require-same-uuid` if the comparison requires the same
physical card; the default allows different cards of the same model and makes
that fact explicit. A 40 GB run may have more OOM cases than an 80 GB run.
