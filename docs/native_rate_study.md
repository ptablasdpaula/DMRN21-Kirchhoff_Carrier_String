# JAX modal synthesis at 44.1 kHz

This study uses the genuine pinned `jaxdiffmodal` solver, all eight physical
parameter gradients, and exactly the same waveform loss as the CUDA study. The
native-rate version integrates and outputs at 44.1 kHz without a downsampling
filter. The original FDTD comparison retains its 132.3-kHz internal rate and
anti-alias filter. The modes are continuous modal basis functions, not the
discrete finite-difference eigenvalues.

All modal counts below project the same reference excitation, generated and
normalized on the original 176-interval grid. This shape contains only the
first eight sine modes; there is no subsequent external force. For this
specific Kirchhoff–Carrier problem, nonlinear tension modulation couples modes
through a shared scalar and does not create a mode whose initial value is
exactly zero. This is not a claim that eight modes suffice for arbitrary
excitations, forcing, boundary conditions, or other nonlinear models.

## Four-second, batch-one pilot

Job `34555111` used the **same A100 PCIe 40 GB GPU** for all five rows. Three
warmed synchronized trials include synthesis, filtering where applicable,
loss and all eight physical parameter gradients. Compilation, target setup,
cross-discretization diagnostics and optimizer updates are excluded. GPU
memory is sampled NVML per-process memory, including CUDA context and allocator
reservations; it can miss short transient peaks. The underlying source hashes
did not change during the run. [Raw data](../benchmarks/native_rate_results/a100-pilot-34555111.json).

| Method | Time | Sampled GPU memory |
|---|---:|---:|
| Cached FDTD, 132.3 kHz | 1.453 s | 0.726 GB |
| Díaz JAX, 175 modes, 132.3 kHz | 4.866 s | 18.195 GB |
| Díaz JAX, 175 modes, 44.1 kHz | 1.630 s | 5.981 GB |
| Díaz JAX, 123 modes, 44.1 kHz | 1.524 s | 5.847 GB |
| Díaz JAX, 8 modes, 44.1 kHz | 1.319 s | 4.081 GB |

For this one physical parameter vector and excitation, 44.1-vs-132.3-kHz JAX
at 175 modes had waveform relative L2 difference 4.64e-7 and scaled eight-
gradient relative L2 difference 3.95e-7. At 44.1 kHz, 8-vs-175 modes had
waveform relative L2 2.70e-13 and scaled eight-gradient relative L2 4.85e-10;
123-vs-175 modes had 1.20e-12 and 5.24e-10. These are measured differences on
this input, not guarantees of stability, full-band accuracy, or convergence
over the parameter box.

Compared to the oversampled FDTD for this case, the 44.1-kHz/175-mode JAX
waveform relative L2 was 0.0328 and its scaled gradient relative L2 was
1.66e-5. Different time and spatial discretizations mean this comparison is
diagnostic, not an assertion of equality. First-item waveform and gradient
artifacts remain on scratch in `native-rate-pilot-34555111-logs/`.

The runner `benchmarks/native_rate_compare.py` supports every power-of-two
batch from 1 to 128, recording OOM and timeout separately, plus modes 8, 123,
and 175. `hpc/native-rate-pilot.json` and `hpc/native-rate-sweep.json` are
designed for `hpc_sonic`; each process is isolated and records the physical
GPU identity and sampled process memory. The sweep configuration retains all
cases on one allocated GPU.

## Completed batch sweep

The 4-second single-GPU A100 80 GB campaign, job `34557214`, completed with
unchanged source fingerprints. Its output prefix is
`/data/scratch/acw794/kc-string-gpu/native-rate-sweep-34557214`: `.png`,
`.pdf`, `.md`, `.csv` and `.json` contain the main comparison, and
`-modes.md` records excitation-specific truncation tests. The 3× JAX pilot
data is not included in the main figure.

At batch 128, cached FDTD with three saved scalars took 2.111 s and 6.793 GB
sampled process memory; one scalar took 2.276 s and 5.708 GB. JAX at 175 modes
ran out of GPU memory for batches 32, 64 and 128. The simple direct-autodiff
FDTD baseline hit the 32 GB host memory limit at batches 1–8 and GPU OOM at
batches 16–128, so no direct-baseline speedup is established by this sweep.

This workload's initial displacement is explicitly a sum of eight sine modes.
For that particular unforced input, JAX's eight-mode and 175-mode waveforms
agree to 2.70e-13 relative L2 in the batch-one check; the eight-mode JAX run
at batch 128 took 1.451 s and 35.377 GB. It is a valid excitation-specific
comparison, not evidence that eight modes suffice for raised-cosine plucks.
The 175-mode JAX and FDTD waveforms differ by about 3.28% relative L2 for the
first tested item, so these costs are not an established equal-accuracy
comparison. All measurements are for the stated physical cases, grid, loss,
precision and hardware, not a guarantee over the whole guitar range.

## One-second raised-cosine sweep (perceptual batch, CeL loss; superseded on 2026-11-01)

The DMRN+21 abstract now uses `hpc/native-rate-sweep.sh` with
`--duration 1 --excitation raised_cosine --modes 100 --methods cached3,fdtd,diaz_44100
--timeout 3600`: the reference code's raised-cosine pluck, one second, Cached
adjoint, Autodiff (direct PyTorch autodiff) and Modal at 100 continuous modes and
44.1 kHz. A raised cosine excites every mode, so the modal count is a real
truncation choice: 123 continuous modes lie below 22.05 kHz for the nominal
string (120–127 across the perturbed batch items), and 175 modes at 44.1 kHz
would alias modes 124–175 into the audio band. One second lets Autodiff finish
for small batches (about 78 kB of autograd graph per step on the host, 2.4 ms
per step), so all three methods appear in one figure
(`paper_dmrn2026/figures/build_benchmark_figure.py`) and the table is gone.

The 0.03 s smoke (job 34600869,
`benchmarks/native_rate_results/a100-smoke-1s-raised-cosine-34600869.json`)
passed for every method; Modal(100) vs the benchmark FDTD grid was 0.98%
waveform relative L2 at that length, 123 vs 100 modes 2.9e-4 and 44.1 vs
132.3 kHz 6.9e-6. The accuracy of both configurations against a
grid-converged reference is in `docs/accuracy_convergence_study.md`.

The order-2 sweep (job 34601165) was cancelled once the fourth-order stencil
was adopted. With the five-point stencil (`SPATIAL_ORDER = 4`) the FDTD runs on
176 intervals at **2×** (88.2 kHz) on the benchmark's narrow `BenchmarkBox`; the
0.03 s smoke at that configuration (job 34612294,
`benchmarks/native_rate_results/a100-smoke-1s-order4-34612294.json`) passed for
every method, with Modal(100) vs the benchmark grid 0.63 % waveform relative L2
at that length (1.03 % over a full second per the accuracy study), 123 vs 100
modes 2.9e-4 and 44.1 vs 88.2 kHz 1.3e-5.

### Completed one-second sweep (job 34614181)

The full one-second sweep ran as job 34614181 on `sae` (rdg13, A100 80 GB
PCIe, torch 2.14.1+cu130) and completed on 2026-10-31 with no source change
during the run. Its JSON is
`benchmarks/native_rate_results/a100-sweep-1s-raised-cosine-order4-34614181.json`
and, as the figure's data, `paper_dmrn2026/figures/native-rate-sweep-34614181.json`
(the default input of `paper_dmrn2026/figures/build_benchmark_figure.py`).
Every failure is a CUDA out-of-memory error; no case timed out or reached the
32 GB host limit. Times are the median of three evaluations (forward, loss and
eight gradients) after warm-up; memory is NVML-sampled process memory.

| Method | batch 1 | last completed batch | first failure | host RSS |
|---|---|---|---|---|
| Cached adjoint (`cached3`) | 0.245 s, 0.60 GB | 128: 0.371 s, 1.69 GB | none | 1.5–1.7 GB |
| Modal (`diaz_44100`, 100 modes) | 0.373 s, 1.54 GB | 64: 0.424 s, 39.3 GB | 128 (GPU OOM) | 1.5–2.7 GB |
| Autodiff (`fdtd`) | 162 s, 2.31 GB | 32: 174 s, 45.2 GB | 64 (GPU OOM) | 9.1 GB |

Trends: the cached adjoint's time grows by a factor of 1.5 over the 128-fold
batch range and its memory from 0.6 to 1.7 GB; Modal is 1.4–1.7× slower with
memory linear in batch (about 0.6 GB per item); Autodiff is about 660× slower
with memory about 1.4 GB per item. Outside the timed region each case is
checked against the checkpointed replay (256-step checkpoints): the cached
adjoint's audio and loss are identical and its scaled gradients agree to 6e-10
relative L2 at every batch, Autodiff's audio and gradients to 7.5e-10; Modal's
first item is 1.15e-2 from the FDTD waveform (5.1e-3 in FFT magnitude) with
scaled gradients 7.7e-5 apart, consistent with the accuracy study. The
abstract's figure (`paper_dmrn2026/figures/build_benchmark_dots.py label`)
plots peak memory against batch size with the first and last evaluation time
of each method printed; a curve simply ends at the last batch that fit in
memory.

## Ten-parameter sweep with the mean-absolute-error loss (2026-11-01)

The abstract's experiment was redefined on 2026-11-01 so that nothing in the
paper refers to the perceptual controls (f0, inharmonicity): the batch now
perturbs the physical parameters themselves, two more parameters are
differentiated, and the loss is a plain waveform error. The same
`hpc/native-rate-sweep.sh` runs it with `--loss mae` (1 s, raised cosine,
`cached3,fdtd,diaz_44100`, 100 modes, batches 1–128, per-case timeout 3600 s,
32 GB host limit); `metadata.loss`, `loss_definition` and `parameters` in the
JSON record the change.

**Ten parameters.** The eight material parameters of the Kirchhoff–Carrier
string plus the pluck position x_e and the pickup ("listening") position x_o,
both as fractions of L (`src/kc_string_forward.py:PARAMETER_NAMES`; nominal
0.5 and 0.5, which reproduce the former centred raised cosine and the midpoint
node bit for bit, so every Octave-exactness test is unchanged). The pluck
gradient flows through the initial-state tangents; the pickup is a per-string
sine-interpolant readout vector whose per-mode gradient the backward kernels
accumulate (`gw`) and the wrappers chain through its tangent. Autodiff and
jaxdiffmodal differentiate both positions with their own AD once they are
traced (`experiments/figure_comparison/fdtd.py`, `diaz.py`).

**Target.** The nominal string of Russo et al., rendered by the benchmark
FDTD itself (176 intervals, 2× = 88.2 kHz, five-point stencil, decimated to
44.1 kHz), plucked at its centre with the reference's raised cosine (half-width
⌊0.25(N−1)⌋ = 43 nodes = 0.244 L) and read at its midpoint:

| ρ | T₀ | r | E | L | amplitude | σ₀ | σ₁ | x_e | x_o |
|---|---|---|---|---|---|---|---|---|---|
| 8050 kg/m³ | 75 N | 0.3556 mm | 174 GPa | 1 m | 2 mm | 0.921 s⁻¹ | 2.86×10⁻⁴ m²/s | 0.5 L | 0.5 L |

jaxdiffmodal is scored against the same FDTD-rendered target (its own
discretisation is about 1 % away in waveform), which is timing-neutral.

**Batch.** `benchmarks/exact_methods.py:varied_material(n)`: item i has every
parameter j multiplied by exp(0.03 sin((i+1)(j+1)·0.73)), clamped to the
`BenchmarkMaterialBox` (nominal·e^{±0.03}), so item i is the same string at
every batch size and the largest |ratio − 1| over 128 items is 3.05 %. The
box's worst stability corner is 0.93 of the bound (margin-adjusted
`GridConfig.physical_score`) and the worst batch item 0.90; ±5 % on the
material parameters alone would reach 1.07, hence ±3 % on all ten.

**Loss.** `--loss mae`: for each string the mean over the 44 100 decimated
samples of |y − target|, summed over the batch, so each item's gradient is
independent of the batch size (the convention of the former grouped CeL;
`src/optimization.py:WaveformMAE`, `experiments/jax_loss_benchmark.py:mae`,
shown equal across torch and JAX by `test/jax_loss_reference.py`).

**Validation chain run on the ten-parameter code before the sweep:**

- `test/positions.py`: nominal readouts bit-identical to the former code;
  autograd FDTD versus the dual recurrence on all ten tangents to 1e-10.
- Octave (jobs 34846668–34846671; `test/reference_paper_exact.py` substitutes
  the pluck centre and a position-dependent readout into Russo et al.'s
  script): 4 s waveform 2.45e-9 (order 2) and 5.0e-10 (order 4) relative L2
  against the reference; all ten gradients against central finite differences
  of the Octave output on an off-centre string (x_e = 0.45 L, x_o = 0.55 L):
  pluck 1.3e-7, pickup 8.7e-8 relative; steel and nylon cases pass. Details in
  `docs/paper_reference_validation.md`.
- CPU battery (job 34854292) and GPU check (job 34851912: fused and adjoint
  parity on every grid and both orders, CUDA and CPU).
- 0.03 s smoke, job 34852079
  (`benchmarks/native_rate_results/a100-smoke-1s-mae-si10-34852079.json`):
  every row ok; the cached adjoint's audio and loss are identical to the
  replay and its scaled ten-gradient within 2e-12, Autodiff within 3e-11;
  Modal(100) first item 6.1e-3 from the FDTD waveform at that length, scaled
  gradients 1.4e-2 apart; 123 vs 100 modes 1.5e-4, 44.1 vs 88.2 kHz 2.5e-6.

### Results

Job 34855605 ran on `sae` (rdg11, A100 80 GB PCIe, torch 2.14.1+cu130, JAX
on the same GPU) and completed on 2026-11-01 with `status complete`,
`loss mae` and no source change during the run. Its JSON is
`benchmarks/native_rate_results/a100-sweep-1s-mae-si10-34855605.json` and, as
the figure's data, `paper_dmrn2026/figures/native-rate-sweep-34855605.json`
(now the default input of `paper_dmrn2026/figures/build_benchmark_dots.py`).
Every failure is a CUDA/JAX out-of-memory error at the same batches as
before; no case timed out or reached the 32 GB host limit. Times are the
median of three evaluations (forward, MAE loss and ten gradients) after
warm-up; memory is NVML-sampled process memory.

| Method | batch 1 | last completed batch | first failure | host RSS |
|---|---|---|---|---|
| Cached adjoint (`cached3`) | 0.240 s, 0.57 GB | 128: 0.348 s, 1.51 GB | none | 1.3–1.5 GB |
| Modal (`diaz_44100`, 100 modes) | 0.377 s, 1.53 GB | 64: 0.419 s, 39.3 GB | 128 (GPU OOM) | 1.4–2.7 GB |
| Autodiff (`fdtd`) | 169 s, 2.33 GB | 32: 177 s, 44.9 GB | 64 (GPU OOM) | 9.0–9.1 GB |

Trends: the cached adjoint's time grows by a factor of 1.45 over the 128-fold
batch range (0.240, 0.243, 0.247, 0.254, 0.257, 0.261, 0.266, 0.348 s) and
its memory from 0.57 to 1.51 GB; Modal is 1.5–1.7× slower at every batch with
memory linear in batch (about 0.6 GB per item); Autodiff is about 700× slower
(169–177 s) with memory about 1.4 GB per item. Outside the timed region each
case is checked against the checkpointed replay (256-step checkpoints): the
cached adjoint's audio and loss are identical and its scaled ten-gradient
agrees to 3e-11 relative L2 at every batch; Autodiff's audio to 4e-10 and its
gradients to 1.5e-9; Modal's first item is 9.6e-3 from the FDTD waveform
(5.6e-3 in FFT magnitude) with scaled gradients 1.3e-4 apart. The per-item
pickup weights and the extra per-mode gradient accumulation cost the cached
adjoint nothing measurable (0.240 s against 0.245 s at batch 1 in the
previous sweep, 0.348 s against 0.371 s at batch 128). The abstract's figure
(`build_benchmark_dots.py label`) and its sentence "nearly three orders of
magnitude faster than plain autodifferentiation and faster than
jaxdiffmodal" are unchanged in substance by the new definition.

## Checkpointed baselines (2026-11-02, review response)

A reviewer of the abstract pointed out that both baselines run as
distributed, without rematerialisation, so their linear memory growth mixes
an implementation choice with the comparison of methods. Two variants isolate
that choice; everything else (workload, grid, rates, loss, batch, timing and
memory measurement) is the one-second ten-parameter MAE sweep above.

- `fdtd_ckpt` (`experiments/figure_comparison/fdtd.py`, `checkpoint_steps=256`):
  the spatial PyTorch loop is cut into 256-step chunks and each chunk goes
  through `torch.utils.checkpoint` (non-reentrant form, every differentiable
  tensor passed explicitly). Only the two states at the chunk boundaries are
  kept; the chunk's graph is rebuilt during the backward pass. Same segments
  as the cached adjoint's checkpoints. `test/figure_fdtd.py` checks that the
  values and all ten gradients equal the plain path bit for bit, including a
  partial final chunk and a chunk longer than the run.
- `diaz_44100_ckpt` (`experiments/native_rate_study/renderer.py`,
  `solve_tf_checkpointed`): the upstream `solve_tf` recurrence (coefficients
  a1, a2, b1, RK4 first step, two-term update, output layout) rewritten as a
  `jax.lax.scan` over 256-step segments whose body, an inner scan with the
  upstream `unroll=8`, is wrapped in `jax.checkpoint`; a shorter final segment
  covers the remainder (44 099 steps = 172 × 256 + 67). The upstream package
  is untouched. `test/native_rate_study.py` checks values (1e-13) and
  gradients (1e-10) against upstream `solve_tf` for segments of 13, 64, 256
  and a segment longer than the run.

Jobs: `hpc/native-rate-checkpointed.sh [methods] [batches] [tag]` runs
`benchmarks/native_rate_compare.py` with the sweep's flags for the methods
given and writes `$KC_SCRATCH/native-rate-ckpt/<tag>-<job>.json`; the live
per-case plot is skipped because it only knows the three original methods.
`hpc/native-rate-checkpointed.json` splits the Autodiff variant into one task
per batch (each about four evaluations of several minutes) so that the tasks
also fit the one-hour `gpushort` partition, and runs the JAX variant in one
task. `hpc/native-rate-checkpointed-smoke.{sh,json}` first runs the two tests
on a GPU node and a 50 ms sweep of both variants at batches 1 and 2 (the
worker validates `fdtd_ckpt` against the replay adjoint like every other
PyTorch method). `paper_dmrn2026/figures/merge_sweeps.py` concatenates the
new rows with job 34855605's into `native-rate-sweep-merged.json`, checking
that duration, loss, excitation, grid, rates and mode counts agree and that
every input ran on the same GPU model; `build_benchmark_dots.py` draws the
checkpointed series as hollow dots (colour = method, fill = checkpointing)
and marks each baseline's first out-of-memory batch with an x on the 80 GB
line.

### Results

Two jobs, both on A100 80 GB PCIe cards: 35062741 (`sae`, rdg7) ran
`fdtd_ckpt` and `diaz_44100_ckpt` over batches 1–128
(`benchmarks/native_rate_results/a100-ckpt-1s-mae-si10-35062741.json`);
35105080 (`gpushort`, sbg4) ran `diaz_44100_ckptro`, the checkpointed
recurrence with the pickup read inside the scan
(`a100-ckptro-1s-mae-si10-35105080.json`). Both completed with every row
`ok` and no source change. The merged figure data is
`a100-sweep-1s-mae-si10-merged-34855605-35062741-35105080.json`, a copy of
`paper_dmrn2026/figures/native-rate-sweep-merged.json`.

| Method | batch 1 | batch 128 | per-item memory | first failure |
|---|---|---|---|---|
| Cached adjoint (34855605) | 0.240 s, 0.57 GB | 0.348 s, 1.51 GB | ~7 MB | none |
| Autodiff, plain (34855605) | 169 s, 2.33 GB | 32: 177 s, 44.9 GB | ~1.4 GB | 64 (OOM) |
| Autodiff, checkpointed (`fdtd_ckpt`) | 256 s, 0.57 GB | 264 s, 1.80 GB | ~10 MB | none |
| Modal, plain (34855605) | 0.377 s, 1.53 GB | 64: 0.419 s, 39.3 GB | ~0.6 GB | 128 (OOM) |
| Modal, checkpointed, trajectory returned (`diaz_44100_ckpt`) | 0.479 s, 0.99 GB | 0.615 s, 26.5 GB | ~0.2 GB | none |
| Modal, checkpointed, readout in scan (`diaz_44100_ckptro`) | 0.497 s, 0.99 GB | 0.606 s, 1.27 GB | ~2 MB | none |

Checkpointed Autodiff: 255.8, 255.9, 258.0, 257.3, 259.3, 263.3, 260.8,
263.8 s over the batches, memory 0.57, 0.60, 0.60, 0.65, 0.72, 0.88, 1.17,
1.80 GB; host RSS 9.5 GB at every batch as for the plain variant (the
autograd graph of one chunk plus the Python objects). Its gradients agree
with the replay adjoint to 0.6–1.5e-9 at every batch, like the plain
variant's. The time factor over plain Autodiff is 1.5: the chunk's forward
pass is recomputed during backward.

Checkpointed Modal with the upstream interface (full modal trajectory and
velocities returned as scan outputs): 0.479–0.615 s, 0.99 → 26.5 GB. The
per-step residuals are gone but the two (steps, modes) float64 outputs and
their cotangents remain, 0.2 GB per item against 0.6 GB for the plain
solver. Reading the pickup inside the scan removes them: 0.497, 0.515,
0.480, 0.518, 0.545, 0.561, 0.588, 0.606 s and 0.99, 0.99, 0.99, 0.72,
0.72, 0.72, 1.02, 1.27 GB (the sampled values fluctuate with the allocator;
the trend is flat). Its first-item waveform and gradients are the plain
modal solver's to the printed digits (9.6e-3 and 1.3e-4 from the FDTD,
exactly as before). The time factor over the plain solver is 1.3–1.5.

Reading: with the same 256-step checkpointing, both baselines reach the
cached adjoint's memory and every batch fits; what remains is the time. The
cached adjoint is 760–1070× faster than checkpointed Autodiff and 1.7–2.2×
faster than the checkpointed modal solver (1.5–1.7× faster than the plain one). The abstract's figure shows the
checkpointed series as hollow dots and prints their time at the last batch;
its text says "nearly three orders of magnitude faster than autodiff and
1.5 to 2.2 times faster than jaxdiffmodal, checkpointed or not; checkpointing
brings both baselines down to our memory at 1.3 to 1.5 times their time".

## Replay adjoint as a timed stage (2026-11-10)

The abstract now compares four FDTD stages and no longer includes
jaxdiffmodal: Autodiff, Autodiff checkpointed, the replay adjoint and the
cached adjoint. The replay adjoint (`src/adjoint.py`, `src/kc_adjoint.cu`)
is the same CUDA kernel, sine basis, 256-step checkpoints and Python wrapper
as the cached adjoint, but stores nothing per step: on the reverse pass it
re-runs each segment forward from its checkpoint into a `(B, 258, n)` state
buffer (three reductions per step) and then reverses through it (five
reductions per step). The cached adjoint stores three scalars per step,
regenerates the states backwards by inverting the step and needs one
reduction per step.

Until now replay was only the untimed validation reference of every PyTorch
case (`reference_seconds`, a single first run: 0.9 to 1.3 s, not a warmed
median). It is now a method of `benchmarks/figure_comparison_worker.py` and
`benchmarks/native_rate_compare.py`; its own validation is against itself
(`validation_reference: self`), and `test/figure_fdtd.py --cuda` (run at the
start of the job, passed) and the `fdtd` rows tie it to autodiff.

Job 35706349 (`hpc/native-rate-replay.sh`, `gpushort`, `a100&80G`, node
sbg1, NVIDIA A100 80GB PCIe; the cached adjoint's rows are from rdg11, the
same GPU model), same one-second MAE workload, result
`a100-replay-1s-mae-si10-35706349.json`, merged into
`a100-sweep-1s-mae-si10-merged-34855605-35062741-35105080-35706349.json`:

| batch | replay | cached adjoint | ratio |
|---|---|---|---|
| 1 | 0.371 s, 0.57 GB | 0.240 s, 0.57 GB | 1.54 |
| 2 | 0.373 s, 0.59 GB | 0.243 s, 0.59 GB | 1.54 |
| 4 | 0.374 s, 0.58 GB | 0.247 s, 0.60 GB | 1.52 |
| 8 | 0.375 s, 0.60 GB | 0.254 s, 0.62 GB | 1.48 |
| 16 | 0.378 s, 0.66 GB | 0.257 s, 0.69 GB | 1.47 |
| 32 | 0.382 s, 0.75 GB | 0.261 s, 0.82 GB | 1.46 |
| 64 | 0.389 s, 0.93 GB | 0.266 s, 1.07 GB | 1.46 |
| 128 | 0.515 s, 1.24 GB | 0.348 s, 1.51 GB | 1.48 |

Reading: the hand-written kernel adjoint accounts for almost all of the
gain over checkpointed Autodiff (510 to 690 times); the three cached
scalars give a further factor of 1.5, for about 2 MB more memory per item
(the scalar cache, 88,200 steps x 3 float64). The abstract's text says "The
replay adjoint is then 510 to 690 times faster, and the cached scalars a
further 1.5 times: 760 to 1070 times faster than checkpointed autodiff".

## Checkpointed adjoint and compiled autodiff baselines (2026-11-10 and 11)

The abstract is reframed. Its method is now the adjoint method with
checkpointing, an established technique in physics simulation (the default
of Diffrax, `RecursiveCheckpointAdjoint`; Kidger's thesis,
arXiv:2202.02435), and that is the `replay` kernel of the previous section.
The text calls it the *checkpointed adjoint* and the figure the *Adjoint
kernel*. The cached adjoint is no longer in the paper or its figure; its
rows stay in the merged data, and the supplement, whose applications ran on
it, says so. The previous source is
`paper_dmrn2026/variants/main-before-checkpointed-adjoint.tex`.

With the kernel as the method, the comparison with `fdtd` and `fdtd_ckpt`
was unfair in one respect: both are eager PyTorch, where every time step
issues about eighty small GPU operations, 88,200 times. Most of the
kernel's 510 to 690 times over `fdtd_ckpt` is that dispatch and not the
method. Compiled autodiff baselines separate the two.

- `fdtd_jax`, `fdtd_jax_ckpt` (`experiments/figure_comparison/fdtd_jax.py`,
  `basis='grid'`): the PyTorch baseline's arithmetic, operation for
  operation (five-point stencil with odd reflection, start-up step, pluck
  profile, pickup interpolant, rank-one solve), written for one string and
  `jax.vmap`ped over the batch. The time loop is a `jax.lax.scan` that reads
  the pickup in its body; loss and gradient are one `jax.jit`. With
  checkpoints it is an outer scan over 256-step segments whose body, the
  inner scan, is wrapped in `jax.checkpoint`, and a shorter checkpointed
  segment covers the remainder: the construction of `diaz_44100_ckptro`.
- `fdtd_jax_sine`, `fdtd_jax_sine_ckpt` (the same file, `basis='sine'`): the
  same scheme stepped in the discrete sine basis. The states are the grid
  states under an orthonormal change of coordinates, the two stencils become
  one multiplication per mode by the eigenvalue table, and the pickup is
  read with modal weights. This is the kernel's formulation, so this row
  compares compiler with hand-written kernel and nothing else.
- `fdtd_compiled`, `fdtd_ckpt_compiled` (`experiments/figure_comparison/fdtd.py`,
  `simulate(compile_block=n, compile_mode=..., checkpoint_reentrant=...)`):
  `_advance` over n steps compiled by `torch.compile` (TorchInductor,
  `dynamic=False`) and called repeatedly, inside each 256-step
  `torch.utils.checkpoint` chunk for the checkpointed variant; a second
  compiled size covers remainders. `torch.compile` has no loop construct: it
  traces the Python loop, so a block of n steps is a graph of n unrolled
  steps and n is limited by compilation time.

Checks. `test/figure_fdtd_jax.py` compares the JAX twin with the PyTorch
baseline in 25 cases per basis (five grids, excitations and orders times
five combinations of steps, checkpoint length and unroll, including no
steps, a remainder only and exact division): waveform to 1e-10 and the ten
gradients to 5e-9 relative, passed for both bases on CPU. `test/figure_fdtd.py
--cuda --compiled` compares the compiled blocks with the eager loop on the
GPU. In the sweep, the worker asserts every JAX case's first item against
the 88.2 kHz FDTD reference (waveform 1e-6, scaled gradient 1e-5) and
validates every `torch.compile` case against the replay adjoint, as for the
other PyTorch methods.

Jobs: `hpc/native-rate-jit.sh [methods] [batches] [tag] [duration]
[driver options]` runs `benchmarks/native_rate_compare.py` on the
one-second MAE workload and writes
`$KC_SCRATCH/native-rate-jit/<tag>-<job>.json`; compiler caches are
node-local and fresh per job, and compilation is outside the medians (the
warm-up call for `torch.compile`, `compile_seconds` for `jax.jit`).
`hpc/native-rate-jit-smoke.sh` runs batch 1 for a quarter of a second to
choose the settings, and `hpc/native-rate-jit-sweep.sh METHOD [BATCHES]`
runs one method of the sweep with them.

### Smoke runs: choosing the settings

Batch 1, 0.25 s, A100 40 GB (`gpushort` and `andrena`), seconds per
evaluation:

| `jax.lax.scan` unroll | 1 | 8 | 32 | 64 |
|---|---|---|---|---|
| `fdtd_jax_ckpt` (grid) | 1.67 | 0.87 | 0.76 | |
| `fdtd_jax` (grid) | 1.41 | 0.72 | 0.62 | |
| `fdtd_jax_sine_ckpt` | 1.47 | 0.61 | 0.54 | 0.51 |
| `fdtd_jax_sine` | 1.25 | 0.53 | 0.47 | 0.45 |

Unrolling pays up to 32 to 64 steps per loop iteration; compilation then
takes 25 to 60 s instead of 2 to 6 s. The sweep uses 32 on the grid and 64
in the sine basis. The sine basis is 1.4 to 1.5 times faster than the grid
at equal unroll.

`torch.compile`, checkpointed (`fdtd_ckpt_compiled`), same workload. Eager
`fdtd_ckpt` takes 104 to 108 s on it on these nodes (sbg21), not the 64 s
that a quarter of the 80 GB card's 256 s would suggest: eager PyTorch is
bound by the host CPU, so eager times are only compared within one node
type.

| block | mode | checkpoint form | result |
|---|---|---|---|
| 256 steps | default | non-reentrant | no compiled block after 25 min (time limit) |
| 256 steps | reduce-overhead | non-reentrant | the same |
| 64 steps | default | non-reentrant | 7.93 s; warm-up 471 s; 2.53 GB |
| 64 steps | default | reentrant | 4.38 s; warm-up 641 s; 2.53 GB |
| 16 steps | default | non-reentrant | 8.10 s; warm-up 80 s; 0.65 GB |
| 16 steps | default | reentrant | 4.93 s; warm-up 105 s; 0.65 GB |
| 16 steps | reduce-overhead | non-reentrant | fails: "accessing gradient tensor output of CUDAGraphs that has been overwritten by a subsequent run" |

Gradients agree with the replay adjoint to 1.4 to 2.6e-11 wherever it runs.
The block length hardly changes the time (8.10 against 7.93 s, 4.93 against
4.38 s) but multiplies compilation by six and memory by four, so the sweep
uses 16-step blocks. The checkpoint form matters more: the reentrant form
runs each chunk's first pass under `no_grad`, where the compiled block is a
plain inference graph, and is 1.6 to 1.8 times faster than the non-reentrant
form, whose first pass still runs the differentiable graph. `reduce-overhead`
replays each block as a CUDA graph with static output buffers, which
checkpointing's recomputation overwrites while the earlier outputs are still
needed for the backward pass; it is not usable here.
`test/figure_fdtd.py --cuda --compiled` passes in the default mode for the
plain loop and for both checkpoint forms (job 35744947), and fails in
`reduce-overhead` with the same error once checkpoints are on.

### Sweep results: the compiled baselines and the kernel

One second, MAE, ten gradients, A100 80 GB PCIe on the `sbg` nodes
(`gpushort`, constraint `a100&80G`), float64. Seconds per evaluation (median
of three after a warm-up) and peak sampled GPU memory in GB:

| batch | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| `replay`, s (job 35706349, sbg1) | 0.371 | 0.373 | 0.374 | 0.375 | 0.378 | 0.382 | 0.389 | 0.515 |
| GB | 0.57 | 0.59 | 0.58 | 0.60 | 0.66 | 0.75 | 0.93 | 1.24 |
| `fdtd_jax_sine_ckpt`, s (job 35743913, sbg4) | 2.01 | 1.86 | 2.34 | 3.05 | 3.22 | 3.32 | 3.53 | 3.65 |
| GB | 0.74 | 0.75 | 0.75 | 0.75 | 0.75 | 1.02 | 1.29 | 2.10 |
| `fdtd_ckpt_compiled`, reentrant, s (job 35744964, sbg4) | 21.7 | 21.5 | 22.0 | 22.0 | 21.8 | 21.7 | 27.0 | 22.1 |
| GB | 0.66 | 1.02 | 1.02 | 1.07 | 1.11 | 1.22 | 1.44 | 1.83 |
| `fdtd_ckpt_compiled`, non-reentrant, s (job 35744004, sbg4) | | | | | 33.9 | 33.5 | 33.8 | 33.9 |
| GB | | | | | 1.10 | 1.22 | 1.43 | 1.82 |

- JAX: compilation takes 36 to 40 s per batch size and is outside the
  medians. The first item of every case agrees with the 88.2 kHz FDTD
  reference to 3.9e-11 in waveform and 1.2 to 1.3e-10 in the scaled
  gradient.
- `torch.compile`: the warm-up call, which compiles the 16-step block and
  the remainder block for the forward and the reverse pass, takes 122 to
  154 s (reentrant) and 101 to 111 s (non-reentrant). Every case agrees with
  the replay adjoint to 2.9 to 4.1e-10 in waveform and 4.9e-10 to 1.4e-9 in
  the scaled gradient. Batch 64 of the reentrant sweep is an outlier in all
  three repetitions (26.8, 28.6 and 27.0 s against 21.5 to 22.1 s at every
  other batch, including 128); the time does not depend on the batch, so
  this is the node at that moment and not the method.
- Job 35744004 ran the non-reentrant form before the smoke runs had shown
  the reentrant one to be faster; it covers batches 16 to 128 and stays as a
  standalone file
  (`benchmarks/native_rate_results/a100-jit-torch-ckpt-nonreentrant-b16_128-1s-mae-si10-35744004.json`),
  not merged into the figure data. The reentrant form is 1.5 to 1.6 times
  faster at batches 16, 32 and 128, and 1.25 times at the outlying batch 64.

### The eager checkpointed baseline: checkpoint form and host (2026-11-11)

Two things about the eager PyTorch rows came out of the compiled work.

**Checkpoint form.** `fdtd_ckpt` uses `torch.utils.checkpoint` with
`use_reentrant=False`, the form PyTorch recommends: each chunk's first pass
runs the differentiable graph with saved-tensor hooks. With
`use_reentrant=True` the first pass runs under `no_grad` and each chunk is
re-run with its own backward pass. A probe on the eager loop (job 35744978,
sbg21, A100 40 GB, batch 1, a quarter of a second) gives 108.0 and 103.6 s
non-reentrant against 73.9 and 73.6 s reentrant, 1.4 times faster, with
gradients equal to 2.7e-11 relative. The reentrant form is therefore the
fairer eager baseline. It is the new method `fdtd_ckpt_reentrant`
(`simulate(checkpoint_steps=256, checkpoint_reentrant=True)`; worker, driver
and `hpc/native-rate-jit.sh`), checked in `test/figure_fdtd.py`
(`test_checkpointed_chunks`: values to 1e-13, gradients to 5e-9, because a
backward pass per chunk sums the parameter gradients in a different order;
this form only supports `.backward()`, not `autograd.grad`).

**Host.** Eager PyTorch issues about eighty small GPU operations per step
and is bound by the host CPU, not the GPU. The eager rows of the figure
(jobs 34855605 and 35062741) ran on the `rdg` nodes (partition `sae`); the
kernel, JAX and `torch.compile` rows ran on the `sbg` nodes (`gpushort`).
Both carry the A100 80 GB PCIe, but the hosts differ:

| eager case, one second of audio | host | seconds |
|---|---|---|
| `fdtd`, batches 1 to 32 (job 34855605) | rdg11 | 169 to 177 |
| `fdtd_ckpt` (non-reentrant), batches 1 to 128 (job 35062741) | rdg7 | 256 to 264 |
| `fdtd`, batches 16 and 32 (job 35745903) | sbg4 | 277.3 and 285.2 |
| `fdtd_ckpt_reentrant`, batch 16 (job 35745148, timed out in batch 32) | sbg4 | 307.8 (320.0, 305.5, 307.8) |
| `fdtd_ckpt_reentrant`, batch 16 (job 35745897) | sbg4 | 350.0 (348.8, 350.0, 350.1) |
| `fdtd_ckpt_reentrant`, batch 32 (job 35745897) | sbg4 | 412.3 (415.0, 411.7, 412.3) |
| `fdtd_ckpt` (non-reentrant), batch 1, four times the 0.25 s probe | sbg21 | 414 to 432 |
| `fdtd_ckpt_reentrant`, batch 1, four times the 0.25 s probe | sbg21 | 294 to 295 |

sbg4 is a Xeon Gold 6242 at 2.8 GHz (`host_cpu`, recorded in the metadata
from job 35745897 on). On it the faster checkpoint form takes longer than the
slower form does on rdg7, and the same case moves by 14 % between two jobs
an hour apart (307.8 and 350.0 s at batch 16) while the three repetitions
inside a case agree to 1 %: the eager time follows the load on a shared
node. The plain loop is 1.6 times slower on sbg4 than on rdg11 (277 and 285 s
against 174 and 177 s at batches 16 and 32, with the same 22.8 and 44.9 GB),
which the earlier smoke runs already showed (`fdtd`, 0.03 s of audio on sbg4
in 7.4 to 7.8 s, jobs 34612294 and 34852079). The two complete sbg4 files
are kept as records and not merged:
`benchmarks/native_rate_results/a100-eager-sbg4-plain-b16_32-1s-mae-si10-35745903.json`
and `a100-eager-sbg4-ckpt-reentrant-b16_32-1s-mae-si10-35745897.json`.

Consequences for the abstract as it stood then (four-series figure; superseded by
the ladder below, whose rows all come from the `rdg` hosts):

- The figure's eager times (169 to 177 s and 256 to 264 s) are the lowest
  eager times measured anywhere, so every ratio of an eager row to a
  compiled row or to the kernel is a lower bound for the hardware on which
  the compiled rows ran.
- The text states the ratios to eager PyTorch only to an order of magnitude
  (about 10 times for `torch.compile`, about 100 times for JAX). Measured:
  256 to 264 s over 21.5 to 22.1 s is 11.7 to 12.2 batch by batch (9.7 at the
  outlying batch 64); over JAX's 1.86 to 3.65 s it is 72 to 138; on sbg4
  alone, 308 to 412 s over the same rows gives about 14 to 19 and 85 to 220.
- The kernel's 5 to 9 times over JAX compares two rows from the `sbg` hosts
  and is unaffected.
- The same sweep of the two eager methods on the `sbg` nodes was started and
  cancelled after jobs 35745897 and 35745903
  (`hpc/native-rate-eager-reentrant.json`, `hpc/native-rate-eager-plain.json`):
  the `sbg` timings are too dependent on load to quote. A re-measurement of
  the other rows on the `rdg` nodes was queued instead; the ladder below
  decides which rows it needs.

### The ladder of the abstract (2026-11-11)

Mixing eager and compiled, grid and sine basis, with and without
checkpoints made the comparison hard to read. The abstract now shows one
ladder of five implementations of the same gradient, each adding exactly one
ingredient to the one before, in a table of memory and time against batch
size (`paper_dmrn2026/figures/build_benchmark_summary_table.py`, the design
of commit 2d5372d):

| row | method | framework | stepping | compiled | checkpoints | adjoint |
|---|---|---|---|---|---|---|
| Autodiff | `fdtd` | PyTorch | grid | no | no | autodiff |
| + compiler | `fdtd_jax` | JAX | grid | yes | no | autodiff |
| + checkpoints | `fdtd_jax_ckpt` | JAX | grid | yes | every 256 steps | autodiff |
| + modes | `fdtd_jax_sine_ckpt` | JAX | sine basis | yes | every 256 steps | autodiff |
| Hand-written | `replay` | CUDA, called from PyTorch | sine basis | yes | every 256 steps | derived by hand |

Reverse-mode autodiff of the scheme is its discrete adjoint, derived by the
framework, so "+ checkpoints" is already adjoint plus checkpointing (what
Diffrax's default does) and "Hand-written" is not a different method: it is
the same adjoint with the same checkpoints, derived by hand and compiled
with the forward pass into one kernel. The discrete sine basis is ours and
not part of the published scheme, which works on the grid (a rank-one solve
per step in `src/kc_string.py`, the port of the Octave reference): with
simply supported ends the discrete sine transform diagonalises both
stencils, the modes interact only through the tension, and the result equals
the grid scheme to rounding. No Diffrax row was added: Diffrax integrates
ODEs with its own time-steppers, and with our scheme as a custom solver it
would reduce to the "+ checkpoints" row with its per-step bookkeeping.

Out of the abstract, kept here: eager PyTorch with checkpoints (two rows
whose times depend on host and checkpoint form), `torch.compile`, the sine
basis without checkpoints and the cached adjoint.

Seconds per evaluation and peak GPU memory in GB; `sbg` nodes except the
Autodiff row (rdg11):

| batch | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| Autodiff `fdtd`, s (job 34855605) | 169 | 169 | 171 | 171 | 174 | 177 | oom | oom |
| GB | 2.3 | 3.8 | 6.3 | 11.8 | 22.8 | 44.9 | | |
| + compiler `fdtd_jax`, s (job 35743473) | 2.46 | 2.27 | 2.48 | 2.99 | 3.18 | oom | oom | oom |
| GB | 5.0 | 9.5 | 18.8 | 36.0 | 64.3 | | | |
| + checkpoints `fdtd_jax_ckpt`, s (job 35743472) | 3.03 | 2.68 | 3.20 | 4.01 | 4.23 | 4.21 | 4.46 | 4.55 |
| GB | 0.74 | 0.61 | 0.74 | 0.61 | 0.74 | 1.01 | 1.28 | 2.09 |
| + modes `fdtd_jax_sine_ckpt`, s (job 35743913) | 2.01 | 1.86 | 2.34 | 3.05 | 3.22 | 3.32 | 3.53 | 3.65 |
| GB | 0.74 | 0.75 | 0.75 | 0.75 | 0.75 | 1.02 | 1.29 | 2.10 |
| Hand-written `replay`, s (job 35706349) | 0.371 | 0.373 | 0.374 | 0.375 | 0.378 | 0.382 | 0.389 | 0.515 |
| GB | 0.57 | 0.59 | 0.58 | 0.60 | 0.66 | 0.75 | 0.93 | 1.24 |
| off the ladder: `fdtd_jax_sine` (no checkpoints), s (job 35743914) | 1.77 | 1.58 | 1.82 | 2.41 | 2.56 | oom | oom | oom |
| GB | 5.3 | 9.6 | 18.3 | 36.0 | 64.3 | | | |

What each step buys:

- Compiler: 55 to 74 times faster than eager PyTorch (with the Autodiff
  row from the faster host, so a lower bound), but 2.2 to 3.1 times the
  memory: 4 GB per string and second against 1.4 GB, so that batch 32, which
  eager PyTorch still fits in 44.9 GB, runs out of the 80 GB (the XLA
  program asks for 95.9 GiB). Compilation fixes time and not memory.
- Checkpoints: memory falls to 0.6 to 2.1 GB (64.3 to 0.74 GB at batch 16),
  for 1.18 to 1.34 times the time.
- Modes: 1.25 to 1.51 times faster at the same memory.
- Hand-written: a further 5.0 to 9.1 times faster, with the lowest memory at
  every batch size (0.57 to 1.24 GB).

Every JAX case's first item agrees with the 88.2 kHz FDTD reference: on the
grid to 5e-13 to 9e-13 in waveform and 1e-12 to 2e-10 in the scaled
gradient, in the sine basis to 3.9e-11 and 1e-10 to 4e-10. Compilation takes
16 to 21 s without checkpoints and 28 to 40 s with them, outside the
medians.

Same-host re-measurement (the abstract's table). The four JAX rows and the
kernel were measured again on the `rdg` nodes, where the Autodiff row ran
(Xeon Gold 6442Y, from `host_cpu`; `hpc/native-rate-rdg-short.json`,
`hpc/native-rate-rdg-jax-grid.json`, `hpc/native-rate-rdg-jax-sine-plain.json`),
and merged with job 34855605 into
`paper_dmrn2026/figures/native-rate-one-host-merged.json`, which the table
script prefers:

| batch | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| Autodiff `fdtd`, s (job 34855605, rdg11) | 169 | 169 | 171 | 171 | 174 | 177 | oom | oom |
| + compiler `fdtd_jax`, s (job 35747154, rdg8) | 2.41 | 2.26 | 2.49 | 2.99 | 3.13 | oom | oom | oom |
| + checkpoints `fdtd_jax_ckpt`, s (job 35747153, rdg13) | 3.04 | 2.68 | 3.18 | 4.01 | 4.25 | 4.23 | 4.45 | 4.56 |
| + modes `fdtd_jax_sine_ckpt`, s (job 35746438, rdg6) | 2.02 | 1.88 | 2.34 | 3.04 | 3.22 | 3.33 | 3.51 | 3.65 |
| Hand-written `replay`, s (job 35746439, rdg6) | 0.369 | 0.371 | 0.371 | 0.372 | 0.375 | 0.379 | 0.384 | 0.511 |
| off the ladder: `fdtd_jax_sine`, s (job 35747074, rdg7) | 1.78 | 1.59 | 1.82 | 2.42 | 2.58 | oom | oom | oom |

Memory is the same as on the `sbg` nodes to the displayed digit. The times
of the JAX rows and of the kernel agree with their `sbg` measurements to 2 %:
unlike eager PyTorch, they do not depend on the host. On this table the
steps read: compiler 55 to 75 times faster (174 to 3.1 s at batch 16) for 2.2
to 3.1 times the memory, with batch 32 out of memory; checkpoints 64.3 to
0.74 GB at batch 16 for 1.19 to 1.36 times the time; modes 1.25 to 1.50 times
faster; hand-written a further 5.1 to 9.1 times faster with the lowest memory
at every batch size. The `rdg` jobs for `torch.compile` and
`fdtd_ckpt_reentrant` (35746441 to 35746445) were cancelled: those rows are
no longer in the abstract.

## The hand-written adjoint without mode stepping (2026-11-11)

Question: how much of the hand-written kernel's lead comes from being hand-written, and how much from mode
stepping? `src/adjoint_grid.py` with `src/kc_adjoint_grid.cu` (method `replay_grid`) is the grid counterpart
of `src/adjoint.py`: the same scheme, the same 256-step checkpoints and the same hand-derived adjoint in one
CUDA kernel, with one thread per grid point. Where the mode kernel multiplies a mode by its eigenvalue, the
grid kernel applies the five-point stencil through shared memory, twice per step in each direction
(`l = 2 c1 u + c4 p - c5 D p + D (c2 u - c3 D u)`, and the transposed form for the adjoint, D being
symmetric). It synchronises five times per step instead of three and tapes three values per point and step
(u, D u, l) instead of one.

Same workload as the paper's table (1 s, 176 intervals at 88.2 kHz, MAE, ten gradients, A100 80 GB). The
worker validates every case against the mode kernel before timing: waveform to 2.9e-10 .. 4.1e-10 and
scaled gradients to 4.9e-10 .. 1.4e-9 (relative l2).

| batch | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| hand-written, grid, rdg3 (job 35757465), s | 0.475 | 0.477 | 0.478 | 0.479 | 0.483 | 0.499 | 0.518 | 0.660 |
| hand-written, modes, rdg6 (job 35746439), s | 0.369 | 0.371 | 0.371 | 0.372 | 0.375 | 0.379 | 0.384 | 0.511 |
| grid kernel, peak GB | 0.57 | 0.59 | 0.58 | 0.60 | 0.66 | 0.78 | 1.00 | 1.38 |
| mode kernel, peak GB | 0.57 | 0.59 | 0.58 | 0.60 | 0.66 | 0.75 | 0.93 | 1.24 |

The same-node pair on sbg4 (job 35757463, both kernels in one job) gives 0.478 .. 0.666 s for the grid
kernel and 0.371 .. 0.517 s for the mode kernel.

- Mode stepping buys 1.29 to 1.35 times inside the hand-written kernel, close to what it buys under JAX
  (1.25 to 1.50 times, `fdtd_jax_ckpt` against `fdtd_jax_sine_ckpt`).
- Hand-writing buys most of the lead: on the same grid, the kernel is 5.6 to 8.8 times faster than the
  compiled, checkpointed autodiff (`fdtd_jax_ckpt`, 2.68 .. 4.56 s).
- Memory is the same up to batch 16 and slightly higher above it (the larger tape).

Results: `benchmarks/native_rate_results/a100-replay-grid-rdg-35757465-1s-mae-si10.json` and
`a100-replay-grid-pair-sbg4-35757463-1s-mae-si10.json`. Job lists: `hpc/native-rate-replay-grid.json`,
`hpc/native-rate-rdg-replay-grid.json`. The paper's table does not include this row.

## PyTorch's compiler on the same host, and a table by framework (2026-11-12)

Question (from a co-author): the ladder's first step changes framework and compiler at once (eager PyTorch to
compiled JAX). What does each ingredient buy inside PyTorch? The three compiled PyTorch rows were measured on
the `rdg` nodes, where the ladder's other rows ran, because `torch.compile` turned out to be bound by the host
CPU like the eager loop (13.5 s here against 21.7 s on sbg4 for the same checkpointed row).

- `fdtd_compiled`, `fdtd_ckpt_compiled`: as before (`torch.compile` blocks of 16 unrolled steps, default mode,
  reentrant `torch.utils.checkpoint` over 256-step chunks).
- `fdtd_sine_compiled`, `fdtd_sine_ckpt_compiled` (new): `experiments/figure_comparison/fdtd.py` gained
  `basis='sine'`, the same change of coordinates as `fdtd_jax`'s (`second(state) = eigenvalues * state`, the
  initial profile transformed once, the pickup read with modal weights). `test/figure_fdtd.py` checks it
  against the grid loop on CPU (`test_sine_basis`: waveform to 1e-9, gradients to 5e-8, plain and chunked) and,
  compiled, against the eager loop on CUDA (job 35837963: all five default-mode cases pass; `reduce-overhead`
  fails under checkpoints as before).

Same workload as the table (1 s, 176 intervals at 88.2 kHz, MAE, ten gradients, A100 80 GB, Xeon Gold 6442Y
hosts). Every case is validated against the hand-written kernel before timing (waveform 2.9e-10 to 4.1e-10,
scaled gradients 4.8e-10 to 1.4e-9). Seconds per evaluation and peak GPU memory in GB:

| batch | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| PyTorch Auto `fdtd`, s (job 34855605, rdg11) | 169 | 169 | 171 | 171 | 174 | 177 | oom | oom |
| GB | 2.3 | 3.8 | 6.3 | 11.8 | 22.8 | 44.9 | | |
| + compiler `fdtd_compiled`, s (job 35837960, rdg4) | 10.7 | 11.0 | 11.2 | 11.3 | 11.8 | 11.8 | 12.1 | 12.4 |
| GB | 1.4 | 2.3 | 3.3 | 5.4 | 9.6 | 18.1 | 34.9 | 71.9 |
| + checkpoints `fdtd_ckpt_compiled`, s (job 35837959, rdg11) | 13.2 | 13.6 | 13.9 | 13.7 | 13.5 | 13.9 | 13.8 | 14.2 |
| GB | 0.66 | 1.02 | 1.02 | 1.07 | 1.11 | 1.22 | 1.44 | 1.83 |
| + modes `fdtd_sine_ckpt_compiled`, s (job 35837961, rdg5) | 4.31 | 4.52 | 4.52 | 4.59 | 4.54 | 4.60 | 4.55 | 4.68 |
| GB | 0.57 | 0.59 | 0.59 | 0.62 | 0.66 | 0.77 | 0.94 | 1.33 |

What each step buys in PyTorch, next to JAX (rows of the same-host table above):

- Compiler: 14.8 to 15.8 times faster than the eager loop, with 0.40 to 0.59 times its memory, and every batch
  size fits (71.9 GB at 128). JAX without checkpoints is 3.8 to 4.9 times faster than this row but uses 3.7 to
  6.7 times its memory and runs out from batch 32.
- Checkpoints: 1.14 to 1.25 times the time (JAX: 1.19 to 1.36). With them JAX is 3.1 to 5.1 times faster.
- Modes: 2.98 to 3.08 times faster (JAX: 1.25 to 1.50), at 0.57 to 1.33 GB, close to the kernel's memory.
  With modes JAX is 1.3 to 2.4 times faster than PyTorch.
- Hand-written kernel: 9.2 to 12.3 times faster than PyTorch with modes; 5.1 to 9.1 times faster than JAX
  with modes, the fastest autodiff row at every batch size.

Block length for the modes row (job 35837962, rdg5, batches 1 and 32): 64-step blocks give 3.92 and 4.17 s
against 4.31 and 4.60 s with 16, 9 to 10 % faster, for 1.89 and 1.48 GB against 0.57 and 0.77 GB and 212 and
111 s of compilation against 69 and 41 s. The table keeps 16, as for the grid rows.

Files: `benchmarks/native_rate_results/a100-jit-torch-{plain,ckpt-reentrant,sine-ckpt-reentrant}-rdg-3583796{0,59,1}-1s-mae-si10.json`
and `a100-jit-torch-sine-ckpt-reentrant-block64-b1_32-rdg-35837962-1s-mae-si10.json`; job list
`hpc/native-rate-rdg-torch-compiled.json`. Merged with the same-host rows into
`paper_dmrn2026/figures/native-rate-two-frameworks-merged.json`.

Test layout, not the submission: `paper_dmrn2026/main-two-frameworks.tex` shows eight rows grouped by
framework (PyTorch: Auto, + compiler, + checkpoints, + modes; JAX: Auto, + checkpoints, + modes; Hand-written),
from `build_benchmark_summary_table.py two-frameworks`. The table is 52 pt taller than the five-row one; the
page holds it with the title gaps at 1.2, 1 and 1 em instead of the template's 2, 1.5 and 1.5 em.

## Autodiff and the adjoint are one computation here (2026-11-12)

Every row of the table computes the discrete adjoint of the scheme (discretise, then differentiate): the
autodiff rows because reverse mode applies the transposed Jacobian of each time step in reverse order, which
is the adjoint recurrence; the kernel because that recurrence is written out by hand. Checkpointing changes
which states are stored, not what is computed. The continuous adjoint (derive an adjoint equation, then
discretise it; `BacksolveAdjoint` in Diffrax, `odeint_adjoint` in torchdiffeq) is not used anywhere: it
returns the gradient of the continuous problem up to discretisation error, not the gradient of the scheme.
Diffrax's default, `RecursiveCheckpointAdjoint`, is the discrete form with checkpoints ("discretise-then-optimise",
"backpropagation through the solver"; https://docs.kidger.site/diffrax/api/adjoints/, read 12 November 2026).

`test/adjoint_states.py` checks this step by step rather than only in the final gradient: the forward and
backward steps of `src/kc_adjoint.cu` transcribed into PyTorch (CPU, float64), against autograd's gradient of
the loss with respect to every intermediate state. For 7, 15 and 175 modes over 1, 40 and 300 steps every
adjoint state equals autograd's state gradient to 2.3e-15 of the largest adjoint, and the accumulated
gradients of A, P, Q and m agree likewise (computeshort job 35844430).
