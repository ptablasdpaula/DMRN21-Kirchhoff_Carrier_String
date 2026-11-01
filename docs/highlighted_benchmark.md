# Highlighted GPU result: 128 four-second strings

On an NVIDIA H100 80 GB HBM3 GPU, the experimental CUDA forward-sensitivity
implementation evaluated **128 four-second strings in 1.589 seconds**, including
synthesis, anti-alias filtering, the independent loss for each string, and
gradients for the selected **f0 and inharmonicity B** coordinates. The other
fitting coordinates were held fixed for the gradient calculation. This is a
highlighted result for future comparisons, not a claim that all conceivable
workloads or methods have been compared.

| Gradient coordinates | Median time | Individual warmed runs |
|---|---:|---|
| f0 and B | 1.589 s | 1.5904, 1.5890, 1.5892 s |
| All six perceptual coordinates | 1.786 s | 1.7877, 1.7860, 1.7865 s |

Both rows cover 512 seconds of output audio per evaluation. The simulation used
float64, all 175 modes (176 intervals), multimode excitation, 44,100 Hz output
and 132,300 Hz internal sampling (3× oversampling). Inputs used varied parameter
vectors; each item had an independent STFT cumulative-energy loss against the
same reference target, and the losses were summed for backward. Timings include
parameter-to-material conversion and exclude compilation, target setup and
optimizer updates. These are synchronized wall-time medians of three warmed
evaluations, not measurements of a complete fitting run. The batched path is
an experimental benchmark interface, not a public batched optimizer.

The measurement was recorded in Slurm job **34328829**, host `xdg2`, using
PyTorch 2.14.1+cu130 and CUDA runtime 13.0. The timing rows and source hash are
preserved in [the raw benchmark result](../benchmarks/jax_results/h100-pure-b128.json).
The timing worker also compared the separate JAX implementation against the
CUDA replay reference; those checks should not be mistaken for an independent
per-case check of the timed CUDA seeded rows. Existing seeded tests and
large-batch isolation checks are described in the
[exact-solver experiment report](exact_solver_experiments.md).

A later direct validation on A100 (job 34504531) checked every item in a
four-second batch of 128 against replay. The scaled f0/B gradient relative L2
error was 5.06e-8, and the all-six-coordinate error was 1.77e-9; waveform and
per-item loss checks also passed. This validates the same experimental code
path on A100, not the original H100 timing directly. [Raw validation and paired
measurements](../benchmarks/jax_results/a100-seeded-direct-b128-4s.json).
The equivalent H100 job 34504524 subsequently passed all-item waveform, loss,
and gradient checks for four seconds and batch 128: scaled gradient relative
L2 errors were 4.97e-8 for f0/B and 1.75e-9 for all six. [H100 direct
validation](../benchmarks/jax_results/h100-seeded-direct-b128-4s.json). This
separate job measured 1.923 s for f0/B and 2.109 s for all six. It used a
different node and validation/timing harness, so these measurements should be
reported alongside, not substituted for, the original 1.589 s result.

The [comparison report](jax_comparison.md) records the single-item and other
batch results, numerical checks, methodology and remaining limitations. The
same benchmark can be submitted through `hpc_sonic` using
[`hpc/jax-pure-b128.json`](../hpc/jax-pure-b128.json) and a fresh submission
state; its [job script](../hpc/jax-pure-b128.sh) also requires the isolated
JAX environment prepared for that experiment.

For subsequent experiments combining cached state reconstruction with grouped
loss evaluation, see [cached reconstruction results](cached_reconstruction_experiments.md).
H100 job 34505297 subsequently measured **1.521 s** for the same batch and
duration with cached reconstruction and all six gradients, versus 1.916 s for
ordinary reconstruction and 2.110 s for forward sensitivity in that same job.
This is a separate run and harness from the 1.589 s result above. Grouped-loss
H100 job 34505774 then measured **1.373 s and 5.00 GB** peak PyTorch allocation
for cached reconstruction plus grouped loss, including all six gradients; the
matched forward-sensitivity baseline was 2.109 s and 10.745 GB. These are
experimental full-loss evaluations, not complete optimizer runs, and use one
common target. See the linked report for validation and caveats.
