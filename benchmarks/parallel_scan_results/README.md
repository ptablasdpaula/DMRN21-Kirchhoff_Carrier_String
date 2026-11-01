# Single-item dense segment-scan scaling

The experiment uses the same algebraic diagonal-plus-rank-one recurrence as the
modal solver but **synthetic, strongly damped coefficients**, not the prepared
physical parameter box. All operations are eager PyTorch (including a naive
autograd baseline); there is no optimized CUDA implementation, audio filtering,
CeL, or full four-second signal. Values cannot be used as estimates of production
runtime, and the strong synthetic damping can cause late states to underflow.

All cases use 175 modes, one string, float64, eight arbitrary coefficient
directions, and a weighted audio-space VJP. Each segment reconstruction and
boundary state/parameter-gradient scan was checked against sequential PyTorch
autograd. Non-divisible durations are padded with zero-weight loss samples.
The candidate total comprises forward checkpoint generation and segment-scan
backward; the synthetic autograd baseline includes forward and backward. Both
are measured after a separate warmup with two timed repetitions.

| GPU | Steps | Chunk | Naive eager forward + autograd | Eager checkpoint forward + segment scan | Scan only | Peak allocated |
|---|---:|---:|---:|---:|---:|---:|
| A100 80GB PCIe | 4,096 | 10 | 2.858 s | 0.728 s | 0.0685 s | 2.07 GB |
| A100 80GB PCIe | 4,096 | 64 | 2.858 s | 0.754 s | 0.1049 s | 0.34 GB |
| A100 80GB PCIe | 4,096 | 256 | 2.858 s | 1.023 s | 0.3701 s | 0.10 GB |
| H100 80GB HBM3 | 4,096 | 10 | 1.756 s | 0.439 s | 0.0349 s | 2.12 GB |
| H100 80GB HBM3 | 4,096 | 64 | 1.756 s | 0.460 s | 0.0593 s | 0.39 GB |
| H100 80GB HBM3 | 4,096 | 256 | 1.756 s | 0.611 s | 0.2078 s | 0.15 GB |
| A100 80GB PCIe | 16,384 | 10 | 11.839 s | 2.866 s | 0.2693 s | 8.20 GB |
| A100 80GB PCIe | 16,384 | 64 | 11.839 s | 2.811 s | 0.1932 s | 1.29 GB |
| A100 80GB PCIe | 16,384 | 256 | 11.839 s | 3.016 s | 0.4063 s | 0.34 GB |

16,384 steps would be about 0.124 seconds at 132.3 kHz; it is still a toy,
not a four-second production run. Do not extrapolate runtime linearly. Results
come from jobs 34308609 (A100), 34308555 (H100), and 34308686 (long A100).
All individual trial/configuration checks passed and both source files remained
unchanged. The [A100](a100-175mode-4096.json), [H100](h100-175mode-4096.json),
and [long A100](a100-175mode-16384.json) records include all rows, warmups,
individual repetitions, projected full-run map storage and source metadata.

For an actual 4-second sequence (529,200 steps) on a single string, dense segment
state/parameter maps alone would require an estimated 53.2 GB at chunk 10,
8.31 GB at chunk 64, and 2.08 GB at chunk 256. For batch 128, chunk 256 maps
alone require ~266 GB. A scan additionally allocates copies and workspace; the
16,384-step chunk-10 toy allocated 8.20 GB for 1.61 GB of state maps plus
parameter maps. Estimates are not measured full-run peaks.
