# JAX versus fused CUDA: initial exact-model comparison

30 October 2026. This is a research benchmark, not a new production backend.
The JAX implementation performs the same full modal nonlinear recurrence and
physical setup using float64, `jax.lax.scan`, all 175 modes, multimode excitation,
132,300 Hz internal rate and 44,100 Hz output. It uses either JAX forward-mode
JVPs for the chosen physical-coordinate directions or JAX's reverse-mode
through the whole scan. The independent JAX implementation of the same default
anti-alias filter and bound STFT cumulative-energy loss is used in the pure-JAX
runs. The precomputed target is shared; target setup, compilation, and optimizer
updates are excluded. Each timing includes synthesis, filtering, loss and
parameter gradients and is the median of three warmed, synchronized calls.

## A100 40 GB PCIe, four-second audio

Job 34328725 ran pure JAX forward mode, compared with the existing fused CUDA
coordinate-seeded forward sensitivities and CUDA replay adjoint on the same
allocated GPU. Two directions are f0 and B; six include every perceptual
coordinate. The seeded paths return gradients only for the selected coordinates.

| Batch | Fitted directions | Pure JAX forward mode | Fused CUDA forward mode | CUDA replay adjoint (all 6) |
|---:|---|---:|---:|---:|
| 1 | f0, B | 10.616 s | 1.805 s | 2.200 s |
| 1 | all six | 12.501 s | 2.158 s | 2.200 s |
| 8 | f0, B | 15.243 s | 1.835 s | 2.247 s |
| 8 | all six | 15.375 s | 2.188 s | 2.247 s |

This JAX `scan` implementation is substantially slower than our current fused
CUDA implementation for these cases; this result does not rule out more
specialized JAX/XLA implementations or other hardware. Forward-mode sensitivity
is already present in our CUDA path and is not specific to JAX. Results are not
complete optimizer-run timings. JAX compilation for the four-second cases took
2.09–5.06 s per compiled batch/direction shape in this run; compilation and
first-run time are recorded separately. JAX peak GPU memory was not isolated in
these runs. [Raw pure JAX / CUDA measurements](../benchmarks/jax_results/a100-pure.json).

As a cross-check, hybrid job 34327840 used the same JAX simulator/JVPs with the
existing PyTorch filter and CeL, sharing buffers through DLPack. For four seconds,
batch 1, it measured 10.937 s (two directions) and 12.852 s (six); batch 8 was
14.859 s and 14.986 s respectively. These are hybrid timings, separate from the
pure-JAX table. [Raw hybrid data](../benchmarks/jax_results/a100-hybrid.json).

## JAX reverse mode

Job 34329070 used `jax.value_and_grad` for the entire JAX synthesis and loss. At
batch 1 and four seconds, plain reverse-mode `scan` took 17.412 s, and a variant
checkpointing each scan step took 16.966 s. The corresponding CUDA replay
adjoint took 2.200 s, on the same GPU allocation in this job. Their JAX compile
times were 43.0 s and 12.3 s, respectively. They compute all six coordinate
gradients; batch 8 at four seconds was intentionally skipped because the
reverse-tape memory requirements had not been measured and checked. Both
variants passed the benchmark gates. Device memory stats in the raw file are
cumulative across cases and exclude PyTorch allocations, and do not establish
per-case total peak memory. [Raw reverse-mode results](../benchmarks/jax_results/a100-reverse.json).

## Numerical checks and limitations

Each JAX result was checked against the existing CUDA replay waveform, bound
loss and physical-coordinate gradients before timing. Waveform comparisons use
`rtol=1e-5, atol=1e-9`; loss and scaled gradient comparisons use `rtol=1e-3`,
`atol=1e-10` and `atol=1e-8`, respectively. Across the four pure-JAX 4-second
batch/direction cases, waveform relative L2 error was 7.16e-10–1.13e-9, loss
relative error 3.55e-11–6.80e-11, and scaled gradient relative L2 error
1.03e-9–4.73e-8. Full reverse-mode B1 gradient relative L2 errors were
1.40e-9–2.27e-9. These are observed errors on the tested parameters, not a
proof over the whole parameter box.

The JAX filter and loss were separately checked on A100 against the installed
PyTorch CeL implementation, including four-second audio, shorter batches, and a
near-target case. The waveform gradient relative L2 errors were at most
1.21e-13. [Validation](../benchmarks/jax_results/a100-loss-validation.json).
The initial 20-step JAX CPU waveform and six-coordinate tangent checks against
the original spatial dual reference had relative L2 errors 2.94e-14 and
1.72e-13, respectively.

All A100 jobs above completed with exit code 0. Raw files contain package/GPU
metadata, source snapshots and individual repetitions. The source hash at job
start for the existing project was
`7b2310bfe033d40c7cc3dcfdc330e87db003e28ee5a3f91a08301cb03e91bf45`.
The reverse harness SHA-256 was
`8ce1e737e723eca8e9d501a79f3d22e2dc409830e30b42a71c7a8647c735c80d`,
and the checkpointed JAX recurrence SHA-256 was
`4b9a3aaa1377c0cd73d2f72225ac07ebc2ca0f8e9900cd7f63f64a0cfcb79808`.

## H100 80 GB HBM3, four-second audio

The previously submitted H100 pure-JAX batch-1/8 and batch-128 jobs 34328734
and 34328829 completed successfully. Using the same settings and timing
conventions as the A100 tests, the following measurements include synthesis,
filtering, loss and parameter gradients; compilation is excluded.

| Batch | Fitted directions | Pure JAX forward mode | Fused CUDA forward mode | CUDA replay adjoint (all 6) |
|---:|---|---:|---:|---:|
| 1 | f0, B | 7.946 s | 1.385 s | 1.555 s |
| 1 | all six | 8.361 s | 1.569 s | 1.555 s |
| 8 | f0, B | 10.370 s | 1.398 s | 1.577 s |
| 8 | all six | 10.570 s | 1.586 s | 1.577 s |
| 128 | f0, B | 11.119 s | 1.589 s | 1.839 s |
| 128 | all six | 12.444 s | 1.786 s | 1.839 s |

For the 4-second batch-128 cases, the JAX waveform relative L2 error versus
CUDA replay was about 1.01e-9, and scaled gradient relative L2 errors were
6.04e-8 for two directions and 2.15e-9 for six. All reported cases passed
their numerical gates. [H100 batch 1/8 raw data](../benchmarks/jax_results/h100-pure.json) ·
[batch 128 raw data](../benchmarks/jax_results/h100-pure-b128.json).

H100 reverse-mode job 34329079 also completed successfully. For batch 1 and
all six coordinates, plain JAX reverse mode took 13.568 s and per-step
rematerialization took 13.800 s, compared with 1.546 s for CUDA replay.
Four-second batch-8 JAX reverse was skipped because its tape memory had not
been measured; the CUDA baseline was measured. These measurements do not
isolate total JAX peak GPU memory. [H100 reverse-mode results](../benchmarks/jax_results/h100-reverse.json).
The separate [H100 filter/loss validation](../benchmarks/jax_results/h100-loss-validation.json)
and [hybrid JAX/PyTorch measurements](../benchmarks/jax_results/h100-hybrid.json)
also completed successfully.
