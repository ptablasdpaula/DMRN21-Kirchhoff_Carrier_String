# Accuracy of the benchmarked FDTD grid and of the modal comparator

`benchmarks/accuracy_study.py` answers the question the DMRN+21 abstract has to
state honestly: the Cached-adjoint FDTD (176 intervals, 3× oversampling) and
the `jaxdiffmodal` Modal comparator (100 continuous modes, 44.1 kHz) are each
run at the resolution their own scheme needs for 44.1 kHz output, and they
are **not** at equal accuracy. This study measures how far each is from a
converged solution, for the benchmark's raised-cosine pluck over one second.
Results: `benchmarks/accuracy_study_results/raised-cosine-1s-{nominal,batch0}.{json,md}`.
Run on CPU in about four minutes with the scratch JAX venv
(`JAX_PLATFORMS=cpu PYTHONPATH=$KC_SCRATCH/diaz-deps`).

## Method

* **String**: the nominal string (f0 = 76.57 Hz, B = 2.876e-4, 123 continuous
  modes below 22.05 kHz) and, separately, the first benchmark item
  (`--parameters batch0`).
* **Excitation**: the reference code's raised cosine centred at L/2 with
  half-width ⌊0.25 (N−1)⌋ nodes. That node convention is grid dependent
  (0.2443 L at N = 176, 0.2496 L at N = 1118), so refined grids use
  `raised_cosine:<fraction>` with the benchmark grid's physical half-width
  43/176 L; on the benchmark grid the two spellings are identical to 1e-15.
  The modal solver projects the same 176-node samples onto its continuous
  modes exactly as the benchmark does.
* **FDTD refinement** follows Russo et al.'s convergence protocol: h is set
  1 % above the stability bound at each rate, h = 1.01 h_min(k) with
  h_min the positive root of h⁴ − (c²k² + 4σ₁k)h² − 4K²k² = 0, and
  N = 2⌊L/2h⌋. Russo's code then recomputes k so that h sits exactly on the
  margin; here the rate stays an integer multiple of 44.1 kHz (as in
  `src/grid.py`) so the project's strided decimation filter applies. For
  oversampling 3, 6, 12, 24, 48 this gives N = 272, 390, 556, 788, 1118
  (nominal string). The CUDA adjoint kernels cap N at 256, so these run on
  the benchmark target's own CPU path (`ForwardString(backend='native')`).
* **Modal**: `solve_tf` with 100 modes (benchmark) and 123 modes (all below
  the output Nyquist), at 44.1 kHz raw (as benchmarked) and band-matched
  through the project's 1× filter (same cutoff and width as the FDTD
  decimation), and at 132.3 kHz decimated exactly like the FDTD.
* **Metrics**: waveform relative L2 and FFT-magnitude relative L2 at
  44.1 kHz over the full second.

## Nominal string

| Candidate | Reference | Waveform rel. L2 | FFT-mag. rel. L2 |
|---|---|---:|---:|
| FDTD 176 / 3× (benchmark) | FDTD 1118 / 48× | 1.140e-1 | 3.98e-2 |
| FDTD 272 / 3× | FDTD 1118 / 48× | 4.80e-2 | 1.70e-2 |
| FDTD 390 / 6× | FDTD 1118 / 48× | 2.39e-2 | 8.79e-3 |
| FDTD 556 / 12× | FDTD 1118 / 48× | 1.07e-2 | 4.26e-3 |
| FDTD 788 / 24× | FDTD 1118 / 48× | 3.72e-3 | 1.52e-3 |
| Modal 123, 44.1 kHz (band-matched) | FDTD 1118 / 48× | 3.72e-3 | 1.56e-3 |
| Modal 100, 44.1 kHz (raw, as benchmarked) | FDTD 1118 / 48× | 3.72e-3 | 1.56e-3 |
| Modal 100, 44.1 kHz (raw) | FDTD 176 / 3× | 1.168e-1 | 4.09e-2 |
| Modal 100 | Modal 123 (44.1 kHz) | 1.43e-5 | 1.01e-5 |
| Modal 100, 44.1 kHz (band) | Modal 100, 132.3 kHz | 3.6e-6 | 2.1e-6 |

Successive-refinement error ratios are 1.87, 1.93, 1.94 against a
second-order prediction of (h_coarse/h_fine)² ≈ 2.0–2.06: on this
stiffness-limited path h shrinks by √2 per level, so the error halves per
level — second-order spatial convergence, as expected for the scheme.

### Reading the table

1. **The FDTD converges to the modal solution.** The finest grid's last
   increment is 3.72e-3; with error ∝ h² and r ≈ 2 its own distance to the
   limit is ≈ 3.7e-3 — the same number as its distance to the 123-mode modal
   waveform. The modal waveform is therefore the limit of the FDTD sequence
   to the resolution of this study (the modal side's remaining errors —
   truncation 1.4e-5, explicit tension term 3.6e-6 — are two to three orders
   smaller). Measured directly, the FDTD error *against the modal waveform*
   is 1.168e-1, 5.12e-2, 2.72e-2, 1.42e-2, 7.31e-3, 3.72e-3 over the six
   grids, i.e. ratios 2.28, 1.88, 1.92, 1.94, 1.97 against the second-order
   prediction 2.39, 2.06, 2.03, 2.01, 2.01.
2. **The benchmark FDTD grid is 11.7 % from that solution** in waveform
   relative L2 (4.1 % in FFT magnitude) over one second. The error is the
   dispersion-induced phase drift of the low modes (on 176 intervals the
   discrete modes are flat by −0.2 ¢ at mode 3, −1.4 ¢ at mode 8, −78 ¢ at
   mode 50) accumulated over a second; the raised cosine puts more energy in
   those modes than the eight-sine excitation of the earlier campaign, whose
   gap was 3.28 % over four seconds. Raising the oversampling alone does not
   reduce it; only N does, and N > 256 is outside the current CUDA adjoint.
3. **The modal comparator's accuracy is not limited by 100 modes or by
   44.1 kHz** for this pluck: 100 vs 123 modes differ by 1.4e-5 (the modes
   between 15.1 kHz and Nyquist carry little energy and decay quickly), and
   44.1 vs 132.3 kHz by 3.6e-6, so the explicit, one-step-lagged tension
   term in `solve_tf` is converged at the audio rate. Running 175 modes at
   44.1 kHz, as the earlier campaign did, would instead alias modes 124–175
   (up to ~42 kHz continuous) into the band — it was harmless only because
   that excitation left those modes at exactly zero.

### What the abstract states

Each solver at the resolution its scheme needs for 44.1 kHz output; the
benchmark grid is 11.7 % from the converged (modal) solution, a 1118-interval
48× grid 0.37 %, and 100 vs 123 modes differ by 1.4e-5. The benchmark
compares backpropagation cost at those stated configurations, not equal
accuracy.

## First benchmark item

`raised-cosine-1s-batch0.md` repeats the study on the first perturbed item
(f0 = 78.90 Hz, B = 3.007e-4, 120 sub-Nyquist modes; refinement grids
264/380/540/768/1088). The picture is the same: benchmark grid 1.209e-1 from
the modal solution (1.178e-1 from the finest grid; 3.51e-2 in FFT magnitude),
finest grid 4.09e-3 from it against a Richardson estimate of 4.04e-3, ratios
2.16, 1.89, 1.90, 1.94, 1.96 against 2.25, 2.07, 2.02, 2.02, 2.01;
truncation 1.19e-5, time stepping 4.7e-6. The sweep's
`validation.first_item_audio_relative_l2_vs_fdtd` for `diaz_44100` at 100
modes must equal this file's "modal-100-44.1k-raw vs fdtd-176-3x" row,
1.209e-1.

## Caveats

* The pluck is symmetric and the pickup is the midpoint, so even modes are
  never exercised.
* One string (plus one perturbed item); the parameter box's corners are not
  covered. The grid is chosen for stability over the box, not accuracy.
* The FDTD reference is itself 0.37 % from the limit; all "vs 1118" numbers
  carry that floor. The modal solution is the better reference for the
  in-band signal, and both agree to within that floor.

## Higher-order spatial stencil (`benchmarks/stencil_study.py`)

The 11.7 % is the three-point stencil's dispersion. Because every symmetric
difference operator with simply supported ends is diagonal in the same discrete
sine basis, the standard five-point (fourth-order) and seven-point (sixth-order)
central differences, closed by odd reflection at the fixed ends, are a change of
eigenvalue table only: −4s² → −4s²(1 + s²/3) → −4s²(1 + s²/3 + 8s⁴/45), with
s = sin(πn/2N). The cost per step, Russo's time step, the rank-one solve and the
adjoint structure are unchanged. Measured with the pure-PyTorch reference
recurrence (its operator patched at runtime; order 2 reproduces the native kernel
to 7e-14), nominal string, raised cosine (physical half-width pinned on every
grid), 1 s, waveform relative L2 against the sub-Nyquist modal solution
(`accuracy_study_results/stencil-1s-nominal{,-rates}.md`):

| Grid | Work vs benchmark | Order 2 | Order 4 | Order 6 | Spectral |
|---|---:|---:|---:|---:|---:|
| 176 / 2× | 0.67 | — | 1.03 % | — | — |
| 152 / 3× (box-stable grid for order 4) | 0.86 | — | 0.43 % | — | — |
| 176 / 3× (benchmark grid) | 1.00 | 11.68 % | **0.45 %** | 0.50 % | 0.50 % |
| 176 / 4× | 1.33 | — | 0.27 % | — | — |
| 256 / 4× | 1.94 | — | 0.30 % | 0.31 % | — |
| 176 / 6× | 2.00 | 11.88 % | **0.18 %** | — | 0.18 % |
| 256 / 6× | 2.91 | — | 0.17 % | — | — |

* With the five-point stencil the grid no longer matters (152, 176 and 256
  intervals agree at a given rate); the remaining error is the second-order
  time step and scales roughly as k²: 1.03 % at 2×, 0.45 % at 3×, 0.27 % at 4×,
  0.18 % at 6×. Oversampling, which did nothing at order 2, now pays.
* Order 6 and the exact eigenvalues are marginally worse than order 4 at the
  same rate: a small residual spatial flattening cancels part of the time
  step's sharpening. Order 4 is the right stop.
* The stencil tightens the stability bound by 4/3 on the wave term and 16/9
  on the bending term, so the full box's largest stable even grid at 3×
  becomes 152 intervals (`GridConfig()` with the new default), and 176
  intervals at 2× is stable only over the benchmark's narrow `BenchmarkBox`
  (corner utilisation 0.87), which is the grid the benchmark now uses.

**Adopted.** The five-point stencil is the project default (`SPATIAL_ORDER = 4`
in `src/kc_string_forward.py`; `order=2` keeps the reference scheme for the
Octave comparisons). It is implemented in the pure-PyTorch recurrence, the
native C++ kernel, the fused CPU/CUDA spatial kernel (nine-point fourth
difference on the odd extension) and, through the shared eigenvalue table
`discrete_eigenvalues`, every sine-basis CUDA kernel. `test/accelerated.py`
checks the direct stencil against the eigenvalue table on CPU and CUDA for
both orders, `test/numerical_validation.py` checks the closed-form linear
response and energy conservation for both, and
`test/reference_paper_exact.py --order 4` compares against the same Octave
code with the five-point operator substituted. Validation on 31 October
2026: the CPU test battery (`hpc/cpu-tests.sh`, job 34612236), the CUDA
fused-kernel check (`hpc/check.json`, job 34612322: spatial vs modal vs
native at both orders, energy drift < 1e-10, four-second renders), the CUDA
adjoint check (job 34612324, four-second VJP), the 0.03 s native-rate smoke
(job 34612294: cached adjoint vs replay vs Autodiff at rtol 1e-6) and the
Octave comparison (job 34612299: nominal string, 4 s at 88.2 kHz, CPU
2.0e-10 and cached CUDA 5.0e-10 relative L2 against the five-point Octave
variant; the order-2 rerun reproduced 2.451e-9 exactly) all passed. The
accuracy study is re-run with `--order 4` (grids 236/338/480/682/968 along
the refinement path); its numbers replace the provisional ones above once the
compute job finishes.

## Fourth-order stencil: measured with the implemented scheme

`benchmarks/accuracy_study.py --order 4` (compute job 34613351; results in
`accuracy_study_results/raised-cosine-1s-{nominal,batch0}-order4.md`), same
protocol as above but with `GridConfig(order=4)`'s bound (h = 1.01·√(4/3)·h_min,
grids 236/338/480/682/968 for the nominal string):

| Candidate | vs modal (123 modes, band-matched) | vs finest FDTD (968 / 48×) |
|---|---:|---:|
| 176 / 2× (benchmark grid) | **1.032e-2** | 1.033e-2 |
| 236 / 3× | 4.87e-3 | 4.81e-3 |
| 338 / 6× | 1.78e-3 | 1.59e-3 |
| 480 / 12× | 9.03e-4 | 6.65e-4 |
| 682 / 24× | 4.80e-4 | 2.33e-4 |
| 968 / 48× | 2.50e-4 | — |
| Modal 100 modes, 44.1 kHz (raw) vs finest FDTD | | 2.48e-4 |

The first benchmark string (`--parameters batch0`, 120 sub-Nyquist modes,
grids 228/…/942) gives 1.145e-2 on the benchmark grid and 2.55e-4 at the
finest, with the same pattern. `stencil-1s-nominal-api.md` reproduces the
earlier monkeypatched study exactly (1.032e-2 at 2×, 4.531e-3 at 3×,
1.782e-3 at 6×), so the `src` implementation and the study's operator agree.

Two observations:

* The error against the modal solution now falls by a factor of about **2
  per level** (ratios 2.12, 2.74, 1.97, 1.88, 1.92 for the nominal string),
  not the 4 that h⁴ (or the k² time step) would give. With the spatial
  dispersion gone, the remaining error is **first order in k**. The
  reference code's startup step (`u¹ = u⁰ + ½k²·h·(…)`, the "extra h
  factors" preserved from the MATLAB code) perturbs the initial velocity by
  O(k), which is the natural source; it is a property of Russo et al.'s
  scheme, not of the stencil, and is the next lever if the remaining 1 %
  ever matters. The Richardson line printed by the script assumes the
  (h_c/h_f)⁴ ratio and therefore underestimates the finest grid's distance
  to the limit; with the observed ratio ≈ 2 the estimate is ≈ the last
  increment (2.3e-4), which matches the measured 2.5e-4 to the modal
  solution.
* The modal floors are unchanged (100 vs 123 modes 1.4e-5; 44.1 vs
  88.2 kHz 2.9e-6), two decades below the finest FDTD grid, so the modal
  solution remains the reference for the in-band signal.

For the abstract: the benchmark FDTD is within 1.2 % (1.03 % nominal,
1.15 % first benchmark string) of the modal solution over one second.
