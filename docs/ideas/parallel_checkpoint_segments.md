# Idea: parallel time segments from saved string checkpoints

Status: **under investigation**. The sequential reconstruction experiment is
a separate baseline; the prototypes below investigate time-segment parallelism.

Save the two consecutive string states needed to restart the recurrence at
regular time boundaries during the forward synthesis. For an illustrative
100-step signal, save the boundary state pairs at steps 10, 20, …, 100. Assign
one independent GPU block or work unit to each segment, and simultaneously
reconstruct its ten preceding time steps from the saved end boundary. This
parallelises work across **time segments**, including for a single string,
rather than merely processing checkpoints sequentially or batching strings.

All counts refer to internal simulation steps. The current default is 529,200
internal steps for four output seconds at 132.3 kHz; a boundary must retain both
consecutive modal states. Periodic anchors limit accumulated reconstruction
error, but the accuracy and conditioning for nonlinear, damped strings must be
measured, not assumed.

State reconstruction for segments can be independent, while the exact reverse
adjoint has a dependency between segment boundaries. Investigate whether each
segment's map from its ending adjoint to its starting adjoint can be represented
compactly and composed with a parallel prefix/reverse scan, including local
loss contributions and parameter gradients. The full state has about 350
components for 175 modes; dense segment matrices and their composition may be
prohibitively costly. Explore whether diagonal-plus-low-rank structure survives
composition without making an unsupported approximation. Do not treat segments
as independent gradients or discard boundary contributions.

A future evaluation should compare one string and batches, four-second output,
FP64 trajectory and gradient accuracy, compilation and wall/CUDA-event times,
peak GPU memory, conditioning and throughput against the existing exact
replay-based adjoint. Validate a range of checkpoint lengths, nonlinear
amplitudes and damping values, including near-singular inverse cases. Preserve
the current forward model and loss; label any approximation explicitly.

## Initial investigation, 30 October 2026

A synthetic FP64 toy composes per-segment affine adjoints using an associative
suffix scan and agrees with PyTorch autograd on CPU and GPU for small problems.
It also computes eight parameter gradients; GPU timings are eager PyTorch on a
different, synthetic problem and must not be compared with our fused production
kernel. See [toy details](../../experiments/parallel_segments/README-toy.md).

For the default 175-mode state, a dense segment map has 350×350 entries. At
four seconds, 256-step segments and batch 128, storing dense state maps and
8-parameter maps alone would require approximately 266 GB, excluding scans and
workspace. Increasing segment length saves maps but increases internal
reconstruction. Short segment Jacobians have diagonal/block plus low-rank
structure, but rank can grow as segments are combined. An exact efficient scan
for the full problem remains an open implementation question.

A second full-grid prototype parallelises only the **state reconstruction** of
bounded groups of segments. It still applies the adjoint sequentially between
segment boundaries; this does not claim to implement the full parallel-gradient
proposal. GPU correctness and paired performance measurements are documented
in the [reconstruction report](../reconstruction_experiments.md). It is
experimental, retains all modes and float64, and is not a public backend.

## One-second outer segments

At 132,300 internal steps per second, a four-second signal has four one-second
segments. Four dense 350-by-350 FP64 boundary maps plus eight-parameter maps
would use about 4.02 MB per string, or 515 MB at batch 128, before scan buffers
and other workspace. This is a storage estimate, not a benchmark of forming
the maps: propagating all 350 state directions through the nonlinear trajectory
is much more work than propagating a single adjoint, even though directions can
potentially run in parallel.

One way to retain these long outer segments without inverting an entire second
is to use short inner checkpoints (for example every 256 internal steps). Each
outer segment could obtain its local loss gradients using short replay or
reconstruction intervals, form its terminal-state and parameter sensitivity
maps, and then join the four outer segments with an exact suffix scan. The
estimates for batch 128 are about 0.74 GB for inner state checkpoints, 0.52 GB
for outer dense maps and 0.19 GB for four 256-step reconstruction tapes, before
outputs, loss, scan workspace and other intermediates. This nested algorithm
has not been implemented or timed; map construction and numerical accuracy
need evaluation.

For context, the nominal linearization around zero displacement predicts an
inverse amplification of approximately 6.5e15 for the highest of 175 modes
over one second, based on the repository's default physical coefficients. This
is a conditioning estimate, not a measurement of nonlinear trajectory error.

A100 40 GB job 34322229 directly tested four 132,300-step segments with the
state-parallel, sequential-adjoint prototype. For the four-second full-loss
and backward workload, three warmed repetitions gave:

| Batch | Replay | Sequential reconstruction | Parallel state reconstruction | Parallel peak CUDA allocation |
|---:|---:|---:|---:|
| 1 | 2.326 s | 1.930 s | 2.014 s | 0.778 GB |
| 8 | 2.385 s | 1.984 s | 2.048 s | 6.004 GB |

Waveform, loss and parameter-gradient gates and the boundary guard passed on
these cases. Scaled gradient relative L2 errors against replay were 3.57e-7
and 1.88e-7 respectively, larger than seen with short checkpoints. This tests
only the state-parallel prototype, not a four-map gradient scan or the nested
algorithm. [Raw results](../../benchmarks/reconstruction_results/a100-parallel-one-second.json) ·
[validation](../../benchmarks/reconstruction_results/a100-one-second-validation.json).
