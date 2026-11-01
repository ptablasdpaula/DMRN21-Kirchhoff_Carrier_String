# Exact-recurrence performance experiments

Status on 30 October 2026: **experimental; benchmarking in progress**. After
an initial Slurm outage, jobs 34299731 and 34299738 passed CUDA/CPU variant,
seeded and checkpointed-adjoint checks, including a four-second adjoint VJP.
Job 34299742 passed short-batch and four-second waveform/loss/gradient checks
and recorded initial timings. Full A100/Hopper and CPU campaigns are queued.
Existing measurements in `docs/gpu_results.md` are from the earlier
implementation and must not be attributed to these variants.

The experiments keep float64, the same grid, integration rate, nonlinear
recurrence and all discrete modes. Variants are opt-in and the existing solver
remains the default:

- A primal register kernel for synthesis and a modal kernel that recomputes
  intermediate terms to reduce shared-memory usage and synchronisation.
- Directional forward sensitivities for selected physical fitting coordinates;
  `fit(..., sensitivity='seeded')` is experimental. The original material
  Jacobian interface and default remain unchanged.
- A checkpointed modal discrete adjoint, `src.adjoint.simulate`: CUDA float64,
  no output subsampling, up to 256 intervals. It replays chunks and computes a
  scalar-loss VJP, rather than materialising the complete audio Jacobian.

The analytic single-step and multistep adjoint equations passed independent CPU
PyTorch autograd checks. Seed chain-rule and fitting plumbing were checked with
CPU mocks. GPU jobs 34299731 and 34299738 subsequently validated the compiled kernels;
performance results are being collected.

## Preliminary A100 result

Job 34299742 completed on an A100 80GB PCIe, using 0.25-second items and two
warm repetitions. Four-second single-item and short batch-4 waveform, loss and
gradient checks passed before timing; the source hash was unchanged. These are
smoke-test timings, not the full benchmark campaign.

| Workload | Batch | Existing CPU modal | Existing CUDA modal | Best experimental CUDA |
|---|---:|---:|---:|---:|
| Synthesis | 1 | 0.01253 s | 0.09533 s | 0.05281 s, register |
| Synthesis | 4 | 0.04736 s | 0.09663 s | 0.05407 s, register |
| Loss and backward | 1 | 0.17927 s | 0.15574 s | 0.14394 s, seeded |
| Loss and backward | 4 | 0.71738 s | 0.16108 s | 0.14868 s, seeded |

The recompute kernel was slower on loss and backward in this short test. The
adjoint passed a four-second arbitrary VJP check but has not yet been timed.
Raw timings and checks: [smoke results](../benchmarks/exact_results/a100-smoke.json)
and [kernel validation](../benchmarks/exact_results/a100-kernel-check.json).
The full A100 and Hopper campaigns (jobs 34300730–34300739), CPU sweep
(34300726–34300727), adjoint smoke (34300765) and seeded full-fit comparison
(34300771) are queued. Inspect their states and logs before interpreting them.

## Even batch sizes through 128

The requested dense sweep covers batch 1 and every even size from 2 to 128,
for 0.25-second and four-second signals. Separate A100 and Hopper campaigns
were submitted as jobs 34305815–34305822 and 34305832–34305839. Each range
validates candidate output and gradients before timing and records peak GPU
memory; it stops a method at the first out-of-memory result. The separate
large-batch isolation check is job 34305803.

Preliminary quick job 34305680 finished on an A100 80GB PCIe, with two warmed
repetitions for 0.25-second signals. **Loss and backward includes independently
computed CeL losses for every item**, with the existing per-item Python dispatch:

| Batch | Existing CUDA modal | Seeded CUDA | Seeded items/s | Seeded peak allocated GPU memory |
|---:|---:|---:|---:|---:|
| 1 | 0.1618 s | 0.1476 s | 6.8 | 23.4 MiB |
| 64 | 0.2793 s | 0.2718 s | 235.5 | 328.2 MiB |
| 128 | 0.5434 s | 0.4385 s | 291.9 | 644.5 MiB |

For synthesis alone, the register path took 0.0536, 0.0561 and 0.0677 s for
batches 1, 64 and 128 respectively. These are preliminary results for short
signals, not full-duration optimisations. The comprehensive 128-item isolation check passed for all four differentiable
methods, comparing four selected items with independent CPU waveforms, losses
and gradients; register synthesis passed the same selected-item check. The dense
2–128 sweeps are running or queued. Focused four-second job 34305969 measures
batches 1, 64 and 128.
[Raw quick measurements](../benchmarks/exact_results/even-batches/a100-quick.json)
include individual repetitions, device metadata, hashes and validation evidence.

A separate four-second A100 40GB PCIe quick run (job 34305969, two warmed
repetitions) measured complete independent loss and backward:

| CUDA method | Batch 1 | Batch 64 | Batch 128 | Batch 128 peak GPU allocation |
|---|---:|---:|---:|---:|
| Existing modal | 2.399 s | 2.596 s | 5.185 s | 12.02 GB |
| Seeded | 2.162 s | 2.371 s | 3.378 s | 10.94 GB |
| Recompute | 2.505 s | 2.704 s | 4.621 s | 12.02 GB |
| Checkpointed adjoint | 2.202 s | 2.437 s | 3.351 s | 8.43 GB |

The batch-128 adjoint and seeded times are close relative to the two-repetition
sample; the adjoint uses less GPU memory. Batch-128 isolation was checked for
four sampled items on a 0.25-second signal, and four-second correctness was
checked for one item. The dense sweep and further repetitions remain in flight.
[Raw four-second measurements](../benchmarks/exact_results/even-batches/a100-long-quick.json)
include timings and source-hashed validation evidence.

To assemble completed shard files without mixing hardware, hosts or source hashes:

```sh
python benchmarks/summarize_even_batches.py --help
```

## Reproducing the jobs

From the repository root, use fresh state paths for every submission:

```sh
python ../hpc_sonic/hpc_sonic.py hpc/experimental-kernels.json --constraint ampere --max-gpus 1 --submit --state /data/scratch/acw794/kc-string-gpu/jobs/experimental-kernels-new.json
python ../hpc_sonic/hpc_sonic.py hpc/adjoint-check.json --constraint ampere --max-gpus 1 --submit --state /data/scratch/acw794/kc-string-gpu/jobs/adjoint-check-new.json
python ../hpc_sonic/hpc_sonic.py hpc/exact-smoke.json --constraint ampere --max-gpus 1 --submit --state /data/scratch/acw794/kc-string-gpu/jobs/exact-smoke-new.json
```

Inspect completed logs and require all correctness checks to pass before the
full benchmarks. Do not assume that a submitted job has succeeded. Repeat on
Hopper with fresh state paths. Then run `hpc/exact-gpu.json` and
`hpc/exact-cpu.json` with appropriate pool, hardware and wall-time constraints;
CPU sweeps have long wall-time estimates. The helper's max-GPU limit applies per
invocation, so account for concurrent campaigns. GPU outputs and detailed
validation are written to `/data/scratch/acw794/kc-string-gpu/` with job IDs.

The benchmark uses distinct in-box parameters and an identical CPU target. It
measures synthesis, material or perceptual tangents, arbitrary VJPs, and
independent filtered CeL losses and gradients for batches of 1–64 and durations
0.25 and 4 seconds. It records synchronized wall/CUDA-event times, warmup,
memory, source hashes and validated baselines; OOM and failed stages are kept.
Adjoint has no explicit full-Jacobian timing. The single-item public optimiser
is unchanged by the batch harness. Compare full fits only after selecting numerically validated candidates from
these results. The corresponding `hpc/exact-fit-{recompute,seeded,adjoint}.json`
workers compare each candidate with modal CUDA using the same target and node.

The separate future idea of parallelising backward work across checkpointed time
segments is saved in [parallel checkpoint segments](ideas/parallel_checkpoint_segments.md).
