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
