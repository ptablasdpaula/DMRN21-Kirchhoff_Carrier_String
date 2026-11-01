# Comparison with the submoduled paper implementation

The reference is `nLinStringsConv-SAV/KCstring/KCStringSAV_noSplit.m` at submodule commit `5253d5c8dc7e2645c925fd45f57f6bc32b66f2ec`. Its original default is `SAV = true` and damping off. Our implementation corresponds to its **non-SAV**, damped branch. Comparisons must explicitly select this branch and matching damping values; they do not establish equivalence with the upstream default SAV scheme.

The original repository test `test/kc_string.py` executes the actual MATLAB script in GNU Octave with SAV disabled, damping enabled and its own 512× grid/time-step calculation. It tests only a 0.0001-second prefix at the default parameters and first-mode initial condition, comparing native CPU and PyTorch outputs with `rtol=1e-9, atol=1e-12`. It is not a full benchmark-duration test. The upstream script chooses an even interval count and then recomputes its timestep; its reported `SR` is the floored reciprocal of that timestep. This is different from fixing the benchmark grid at 176 intervals and 132300 Hz.

`test/reference_paper_exact.py` copies the submodule script into a temporary directory and configures physical parameters, damping, `SAV=false`, 176 spatial intervals, `dt=1/132300`, and the requested number of steps. It leaves the upstream initial conditions, recurrence and energy calculation intact. It compares the raw midpoint signal, including both startup samples, before antialias filtering and decimation. Double precision is used throughout. Matching this adjusted-grid upstream program demonstrates agreement of the discrete scheme under shared inputs; it does not claim bitwise identity or identity with the unmodified upstream default grid or SAV branch.

The first pilot, job 34559180 on 31 October 2026, compared 265 steps (about 2 ms) for a non-default combination of all eight physical parameters. The saved output is [`benchmarks/paper_reference_results/octave-pilot-34559180.json`](../benchmarks/paper_reference_results/octave-pilot-34559180.json). The maximum waveform difference was 1.40e-15 m and relative L2 difference was 3.44e-13. The initial total energy was 0.0009031320583064426 J in Octave and 0.0009031320583075776 J in the independent CPU evaluation. The first two samples were identical at displayed precision.

The same pilot measured eight gradients of a normalized mean-square signal loss using central differences of the Octave output and analytical native CPU tangents. All eight estimates were recorded, but the smallest gradient (`sigma1`) was **inconclusive** at the original perturbation: its absolute difference was 2.86e-8 and relative difference 3.55%, consistent with subtractive sensitivity at that small gradient. The pilot did not assert gradient tolerances. The improved harness increases the perturbation for the damping coordinates, sets an explicit `rtol=1e-3, atol=2e-9` in relative-multiplier gradient coordinates, and checks cached CUDA outputs and gradients directly against Octave for the one- and three-scalar variants. It also checks additional short steel-like and nylon-like parameter vectors.

The four-second nominal first-mode CPU/Octave comparison completed and passed its waveform assertion in job 34559393: 529200 steps, maximum absolute error 2.53e-12 m and relative L2 error 2.41e-9. Its [result JSON](../benchmarks/paper_reference_results/octave-four-second-34559393.json) records 58.8 s for Octave (including startup) and 8.1 s for native CPU including extension loading; these are validation timings, not GPU benchmark figures. The first-step total energies differ by 5.9e-18 J. The comparison printed and saved its successful result, but Slurm recorded the job as failed after the comparison because the shell wrapper was edited concurrently while it was running; the final wrapper now passes `bash -n`.

A clean rerun in job 34560485 completed successfully and added full-duration CUDA comparisons. For both cached variants the raw waveform differs from Octave by at most 3.24e-12 m and has relative L2 error 3.07e-9; the CPU waveform errors match the previous run. The full measurements are in [the result JSON](../benchmarks/paper_reference_results/octave-cuda-four-second-34560485.json).

The independent short cached-CUDA/Octave and gradient comparison in job 34560206 completed successfully. Both the one-scalar and three-scalar cached variants agreed with the raw Octave output to 1.56e-15 m maximum absolute error and 3.41e-13 relative L2 error. All eight CPU and cached-CUDA gradients passed `rtol=1e-3, atol=2e-9` against Octave finite differences; the revised sigma1 perturbation reduced its CPU difference to 7.95e-12 absolute (9.53e-6 relative). Additional short steel-like and nylon-like CPU waveforms passed at 1.03e-12 and 1.59e-13 relative L2 respectively. Complete parameters, tolerances and results are in [the result JSON](../benchmarks/paper_reference_results/octave-cuda-gradients-34560206.json). Job 34559179 failed from concurrent Pixi environment creation; job 34559979 failed before comparison because the direct Octave reuse lacked `OCTAVE_HOME`. Neither is validation evidence. The original dependent gradient job 34559394 was cancelled in favour of the independent suite. Results should be added here only after inspecting the job outputs.

## Raised-cosine excitation

The reference harness also accepts `--excitation raised_cosine`; it sets the upstream script's `initType` to 2 without changing the initialization formula, and uses the matching native/CUDA shape. At 176 intervals the reference width parameter is `floor(0.25 * 175) = 43` grid intervals on **each** side of the midpoint, for a total support of 86/176 of the string length. All comparisons below use the damped non-SAV branch, fixed 176-interval grid, 132300 Hz internal sample rate, float64, and the raw midpoint output including startup samples. This reference-code convention should not be confused with the paper's description of a 25%-wide raised cosine.

Job 34563185 ran the nominal parameters for four seconds and completed successfully. Both cached CUDA variants matched the actual Octave reference waveform to a maximum absolute error of 1.83e-12 m and relative L2 error of 2.45e-9; the CPU comparison gave 1.46e-12 m and 1.93e-9, respectively. The [result JSON](../benchmarks/paper_reference_results/octave-cuda-raised-four-second-34563185.json) records the parameters and energy diagnostics. These are direct waveform comparisons, not runtime benchmarks.

Job 34563184 compared 265 steps for a varied eight-parameter vector and checked gradients of a normalized mean-square loss in relative-multiplier coordinates against independent central finite differences of the Octave signal. All eight CPU and one- and three-scalar cached CUDA gradients passed `rtol=1e-3, atol=2e-9`. The CUDA waveform relative L2 error was 1.20e-12, with maximum absolute difference 2.48e-15 m; its largest absolute gradient difference from Octave finite differences was 4.15e-9. Additional short CPU waveform comparisons for steel-like and nylon-like vectors had relative L2 errors 7.36e-13 and 1.74e-13, respectively. The complete numerical results are in [the gradient result JSON](../benchmarks/paper_reference_results/octave-cuda-raised-gradients-34563184.json).

The raised-cosine initial shape and midpoint readout are symmetric, so the midpoint signal does not exercise antisymmetric modes. Benchmark figures use an eight-mode initial shape and filtered 44.1 kHz output, so direct Octave checks of that benchmark excitation require configuring the same initial shape and then applying the same filter to both raw signals. The present checks do not establish coverage across an entire guitar parameter box, all excitations, all durations, or arbitrary losses; gradients are first-order and are tested at the stated fixed discretization.

## Fourth-order stencil (`--order 4`)

Since the project default became the five-point spatial stencil, the harness
takes `--order`: `--order 2` (default) is byte-for-byte the comparison above
against Russo et al.'s unmodified operator at 176 intervals and 132 300 Hz;
`--order 4` runs at the benchmark's 88.2 kHz and replaces the script's single
operator line `Dxx = Dxp*Dxm;` by the pentadiagonal five-point matrix
(−1, 16, −30, 16, −1)/(12h²) with the odd-reflection end rows (−29/12 on the
corner diagonal entries). The script's `Dxxxx = Dxx.'*Dxx`, startup, update,
nonlinear term and energies all read `Dxx`, so the whole reference switches
with it. This is a modified reference, not the paper's code; it validates
that our fourth-order implementation integrates the same discrete scheme the
modified Octave code does. Jobs are submitted through
`hpc/paper-raised-cosine.json` (long and gradient modes at both orders); add
their results here after inspecting the job outputs.

After the stencil refactor, the `--order 2` path was re-run on 31 October 2026
to confirm that the three-point scheme is unchanged: the four-second
raised-cosine long run (job 34612298) reproduced the earlier numbers exactly —
CPU 1.46e-12 m / 1.934e-9 relative L2, both cached CUDA variants 1.83e-12 m /
2.451e-9 — and the gradient run (job 34612296) gave 3.94e-13 relative L2 with
all eight gradients within `rtol=1e-3, atol=2e-9` (largest relative error
4.1e-7), i.e. byte-for-byte the previous behaviour. The `--order 4` long run
uses the nominal string at 88.2 kHz; the `--order 4` gradient run uses the
perturbed test string, which sits at a margin-adjusted utilisation of 1.022 on
176 intervals at 2× with the five-point stencil, so it runs with
`--oversampling 3` (132.3 kHz) instead.

Job 34612299 ran the `--order 4` four-second raised-cosine comparison for the
nominal string at 176 intervals and 88.2 kHz (margin-adjusted utilisation
0.75): the CPU waveform matched the five-point Octave variant to 3.9e-13 m
maximum absolute error and 2.02e-10 relative L2, and both cached CUDA
variants to 3.9e-13 m and 5.05e-10. Result JSON:
[`benchmarks/paper_reference_results/octave-cuda-raised-four-second-order4-34612299.json`](../benchmarks/paper_reference_results/octave-cuda-raised-four-second-order4-34612299.json).
The order-2 reruns are saved alongside it with a `-rerun-` suffix.

Job 34614567 ran the `--order 4` gradient comparison (265 steps, perturbed
eight-parameter vector, `--oversampling 3` so the string sits inside the
margin): CPU waveform 4.4e-15 m / 1.98e-12 relative L2 against the five-point
Octave variant, both cached CUDA variants 4.6e-13 relative L2, and all eight
CPU and cached-CUDA gradients within `rtol=1e-3, atol=2e-9` of the Octave
central differences (largest relative error 6.7e-7). The additional
steel-like and nylon-like strings ran at the smallest stable rate for the
five-point stencil on 176 intervals (4× and 3×) and matched to 7.9e-12 and
2.2e-13 relative L2. Result JSON:
[`benchmarks/paper_reference_results/octave-cuda-raised-gradients-order4-34614567.json`](../benchmarks/paper_reference_results/octave-cuda-raised-gradients-order4-34614567.json).

## Ten parameters: pluck and pickup positions (2026-11-01)

The parameter vector now has ten entries: the eight material parameters plus
the raised-cosine pluck centre and the pickup position, both as fractions of
L with nominal value 0.5 (the reference's centred pluck and midpoint node).
`test/reference_paper_exact.py` carries them into the Octave script by text:
`excitPos = outPos;` becomes the pluck centre in node units (the script's
raised-cosine loop accepts a non-integer centre as it stands), and, whenever
the pickup does not sit on a node, the three `Out(...) = ...(outPos)` readouts
become `wOut.'*u` with `wOut` the sine interpolant of the interior nodes at
the pickup fraction; on a node `outPos` is simply set to that node, so the
nominal string reads the script's own midpoint exactly as before. The
gradient runs use an off-centre string (pluck 0.45 L, pickup 0.55 L), since at
the centre both position derivatives vanish by symmetry.

Jobs 34846668–34846671 (ten-parameter kernels, `hpc/paper-raised-cosine.json`):

| run | waveform vs Octave (CPU) | cached CUDA | gradients |
|---|---|---|---|
| long, order 2, 132.3 kHz, 4 s, nominal | 1.93e-9 relative L2 | 2.4514e-9 (both cache counts) | — |
| long, order 4, 88.2 kHz, 4 s, nominal | 2.02e-10 | 5.047e-10 | — |
| gradients, order 2, 2 ms, off-centre | 4.1e-13 | 1.4e-12; gradient max abs error 9.6e-8 | all ten within `rtol=1e-3, atol=2e-9`; pluck 1.3e-7, pickup 8.7e-8 relative |
| gradients, order 4, 3×, 2 ms, off-centre | 2.2e-12 | 4.1e-13; 9.7e-8 | all ten; pluck 1.3e-7, pickup 8.7e-8 relative; steel 4× 7.9e-12, nylon 3× 2.2e-13 |

The two long runs reproduce the earlier eight-parameter values digit for digit
(2.451e-9 and 5.05e-10), which is the intended guarantee: at the nominal
positions the new readout and pluck are bit-identical to the former node
formulas. Result JSONs: `benchmarks/paper_reference_results/octave-cuda-{long,gradients}-order{2,4}*-ten-<job>.json`.
A pure-torch cross-check, `test/positions.py`, additionally shows the plain
autograd FDTD and the dual recurrence agreeing on all ten tangents to 1e-12
relative at off-centre positions.

Beyond the Octave comparison, the ten-parameter kernels passed the CPU battery
(`hpc/cpu-tests.sh`, job 34854292), the GPU check (`hpc/check.json`, job
34851912: fused and adjoint parity on every grid and both stencil orders, CUDA
and CPU) and, for the ported legacy experiments (structured adjoint, scalar-cache
fallback, parallel reconstruction, alpha scans, cached reconstruction), their
own GPU tests (jobs 34856222–34856224 on gpushort, every check passed; the
cached-reconstruction smoke agrees with the replay to 3e-10 at batch 1 and
1.6e-10 at batch 128 over a full second).
