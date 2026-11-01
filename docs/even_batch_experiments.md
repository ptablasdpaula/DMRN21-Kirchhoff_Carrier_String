# Even-size batch experiments

The benchmark runs batch 1 and every even batch size 2, 4, 6, …, 128,
for 0.25- and 4-second strings on the same 176-interval, 132.3 kHz internal
rate, float64 recurrence. No modal truncation, approximate gradients or lower
internal rate are used. Each independent string uses a different deterministic
parameter vector and is compared against the same CPU-generated target.

Methods are original CUDA modal, selected-coordinate seeded, modal recompute,
checkpointed adjoint (independent loss/VJP), and register-based primal-only
synthesis. Loss and backward are timed end to end; the existing independent CeL
losses are evaluated in a Python loop, whose overhead is part of the reported
time. Absolute latency, items/second, per-repeat and CUDA-event timings, peak
allocated CUDA memory, warmup and source/hardware metadata are recorded.

A quick 0.25-second, two-repeat 1/64/128 comparison is job 34305680. An
independent 128-item isolation test passed in job 34305803: it compares selected
items' audio, loss and gradients against separate native CPU runs. Every full
shard also repeats this check plus short-batch and four-second single-item
preflight tests before running.

A self-gated 4-second, two-repeat 1/64/128 quick comparison is job 34305969.

The A100 full campaign is jobs 34305815–34305822; Hopper is jobs
34305832–34305839. Each campaign is a single chained lane, split into four
ranges (2–32, 34–64, 66–96, 98–128) and two durations. Each range also measures
batch 1 on its node. Three warmed timed repetitions are used per method, size and
duration. Slurm dependencies and estimates do not guarantee completion: inspect
`sacct`, job logs and result JSON. Results live in
`/data/scratch/acw794/kc-string-gpu/even-sweep-*` and
`/data/scratch/acw794/kc-string-gpu/even-batch-quick-*`.

Each method/size is run in its own process. The aggregate JSON is saved after
every case, including child files, exit codes, memory and timing rows. On a
recorded CUDA out-of-memory event, the driver stops larger sizes for that method
and continues with the remaining methods. Unexpected nonzero child exit marks
the shard failed and retains its prior results. GPU memory use grows with both
batch size and duration; a 4-second batch of 128 needs about 4.33 GB just for
the original eight-column internal-rate Jacobian.

To repeat, submit `hpc/even-batch-quick.json`,
`hpc/even-batch-isolation.json` and `hpc/even-batch-sweep.json` using
`hpc_sonic.py`, a fresh state path and the desired `ampere` or `hopper`
constraint. The implementation and benchmark files are:
`test/even_batch_isolation.py`, `benchmarks/even_batch_driver.py`, and
`hpc/even-batch-sweep.sh`. Do not pass `--quick` to the underlying
`benchmarks/exact_methods.py`; it overrides the requested batch list.

To summarise completed or partial aggregate JSON files without mixing GPUs,
hosts or source versions, use e.g.:

```sh
/data/scratch/acw794/kc-string-gpu/venv/bin/python benchmarks/summarize_even_batches.py \
  /data/scratch/acw794/kc-string-gpu/even-sweep-*.json \
  /data/scratch/acw794/kc-string-gpu/even-batch-quick-*.json \
  --output-dir /data/scratch/acw794/kc-string-gpu/even-summary
```

The summary excludes duplicate per-case child JSON and changed-source runs,
marks running shards provisional, and reports batch coverage and out-of-memory
cases. `timings.csv` includes peak allocated memory; figures compare latency and
throughput separately for each host, GPU, source hash, duration and workload.

## Initial short-duration measurements

Job 34305680 completed on `sbg4` (A100 80 GB PCIe), measuring 0.25-second
signals with two warm repetitions. Its short-batch and 4-second single-item
preflights passed; the source hash remained unchanged. The independent
128-item sample-isolation job 34305803 also passed for all five methods.

| Loss and backward | Batch 1 | Batch 64 | Batch 128 |
|---|---:|---:|---:|
| CUDA modal | 0.16185 s | 0.27934 s | 0.54337 s |
| CUDA seeded | 0.14758 s | 0.27181 s | 0.43848 s |

At batch 128, seeded processes approximately 292 strings/s versus 236 for the
baseline, and peak allocated CUDA memory was 676 MB versus 744 MB. These are
single-machine, two-repeat measurements, and include the independent Python
loss loop. The dense even-size sweep is still in progress; the four-second quick results follow below.
For primal-only register synthesis, elapsed time was 0.05360 / 0.05614 /
0.06772 s at batches 1 / 64 / 128. The CPU is not measured in this quick job.

[Raw result](../benchmarks/even_results/a100-quick-34305680.json) ·
[Loss-and-backward plot](../benchmarks/even_results/a100-quick-summary/NVIDIA-A100-80GB-PCIe-sbg4-7b2310bfe033-0.25-loss_backward.png) ·
[CSV and coverage](../benchmarks/even_results/a100-quick-summary/summary.json)

Job 34305969 subsequently completed the four-second 1/64/128 quick comparison
on `sbg19` (A100 PCIe 40GB). It passed both numerical preflights and repeated the
128-item isolation checks before timing, with two warm repetitions. The source
hash remained unchanged.

| Loss and backward | Batch 1 | Batch 64 | Batch 128 | Peak at batch 128 |
|---|---:|---:|---:|---:|
| CUDA modal | 2.3988 s | 2.5961 s | 5.1854 s | 12.02 GB |
| CUDA seeded | 2.1616 s | 2.3710 s | 3.3784 s | 10.94 GB |
| CUDA recompute | 2.5048 s | 2.7045 s | 4.6208 s | 12.02 GB |
| CUDA checkpointed adjoint | 2.2023 s | 2.4373 s | 3.3506 s | 8.43 GB |

At B=128 the adjoint achieved 38.2 strings/s, the seeded method 37.9/s and the
baseline 24.7/s. The adjoint/seeded latency difference is small relative to the
limited two-repeat sample; their memory difference is substantial. Register
primal-only synthesis at B=128 took 0.9979 s. These results include the Python
per-item loss loop and should not be read as complete batched optimiser timings.

[Raw four-second result](../benchmarks/even_results/a100-long-quick-34305969.json) ·
[Latency and throughput plot](../benchmarks/even_results/quick-summary/NVIDIA-A100-PCIE-40GB-sbg19-7b2310bfe033-4.0-loss_backward.png) ·
[Reproducible summary and coverage](../benchmarks/even_results/quick-summary/summary.json)
