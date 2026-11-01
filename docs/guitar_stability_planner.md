# Guitar-range stability planning for the non-SAV scheme

`python benchmarks/guitar_stability_planner.py` produces the per-case table in
`benchmarks/guitar_stability_results/cases.csv` and the summary JSON beside it.
The calculation uses the same conservative 1% grid-spacing margin as
`GridConfig.physical_score`; a score strictly below 1 is required. It uses
176 spatial intervals, double-ended fixed/simply supported boundary conditions,
and an output rate of 44,100 Hz. The internal rate is an integer multiple of
44,100 Hz. This is a planning calculation, not an execution-time benchmark.

## Defined domains and results

We take the 12 open strings in [Woodhouse et al., Table I](https://www.lam.jussieu.fr/Publications/Articles/AAA_Woodhouse_2012.pdf):
six nylon and six new steel strings with their measured open-note frequencies
and inharmonicities. As a clearly stated illustrative family, each has a 0.65 m
scale length and is shortened from fret 0 through fret 24, holding tension,
mass per length and bending stiffness fixed. Thus each case is correlated:
`L = .65*2**(-fret/12)`, `f0 = f0_open*2**(fret/12)` and
`B = B_open*2**(fret/6)`, where `f0_open = measured_first_partial / sqrt(1+B_open)`.
Real fretting can change tension; the source is a sample, not every guitar.
For this calculation we assume `0 <= sigma1 <= 0.001 m²/s`. This is a chosen
bound for the project's damping model, not a claim that Woodhouse measured it.

* The 300 correlated string/fret cases require a single global factor **34**,
  or **1,499,400 Hz**, to pass the margin criterion. The worst case is the
  nylon high-E string at fret 24. At 3×, 207 cases fail the margin criterion.
  Failing this conservative criterion alone does not prove a case unstable.
* Per-case factors vary from 1 to 34, with median 5. Per-case factors are only
  suitable if the numerical configuration can be fixed before an optimization
  and the whole optimization domain is covered. Changing a grid or rate during
  differentiation can introduce discontinuities and invalidate comparisons.
* A larger **independent** box covers `f0 = 27.5–1318.4 Hz`,
  `B = 1e-6–0.001984`, `L = 0.1625–0.9 m`, and
  `sigma1 = 0–0.001 m²/s`. It requires a global factor **55**, or
  **2,425,500 Hz**. The worst score is 0.98815 at 55× and 1.02505 at 54×.
  The lower-frequency and longer-scale extension is an assumed sensitivity
  margin for extended-range and bass-like instruments, not a measured survey.
  Some box corners combine material properties that no real guitar string has.
  It is a mathematically bounded domain, not a universal bound on all guitars.

With a fixed 176 intervals the grid spacing is `h=L/176`, ranging from
0.923 mm at 0.1625 m to 5.114 mm at 0.9 m. Holding interval count fixed
allows physically different lengths without resizing the state. A condition
on stability says nothing by itself about the accuracy of high partials or
the cost of four seconds of audio at these rates. An output antialias filter
and decimation also need to support the selected factor.

## Why the criterion covers the nonlinear update

The following is a direct algebraic sufficient condition for the specific
non-SAV damped recurrence in `KCStringSAV_noSplit.m`, in exact arithmetic.
It is not an assertion that every FDTD or SAV scheme has the same bound.
Assume `m = rho*A > 0`, `T > 0`, `E >= 0`, `sigma0 >= 0`, `sigma1 >= 0`,
finite initial states, fixed `h` and `k`, and the code's symmetric
simply-supported spatial matrices. Let `K = -Dxx`,
`A_lin = (T/m) K + (E*I/m) K²`, `w = u[n+1]-u[n]`, and
`v = u[n]-u[n-1]`. The non-SAV cross-potential is a nonnegative square.
The damped recurrence has the modified half-step energy

```
H(u_next,u) = m*h/(2*k²) * ||w||²
              + h/2 * u_nextᵀ (T*K + E*I*K²) u
              + E*A*h²/(8*L) * (u_nextᵀ K u)²
              - m*h*sigma1/(2*k) * wᵀ K w.
```

Substitution of the update gives

```
H_next - H_prev = -m*h/(2*k) * (
    sigma0 * ||w+v||² + sigma1 * (w+v)ᵀ K (w+v)) <= 0.
```

Writing `z=(u_next+u)/2`, its quadratic part is

```
m*h/2 * zᵀ A_lin z
+ m*h/(2*k²) * wᵀ [I - k²*A_lin/4 - k*sigma1*K] w.
```

Therefore the strict matrix condition `k²*A_lin + 4*k*sigma1*K < 4*I`
makes this energy coercive; with its nonnegative nonlinear term and
nonincreasing energy, the finite-dimensional states remain bounded. On this
uniform grid `K` has eigenvalues
`lambda_j = 4/h² * sin²(j*pi/(2*N))`, and it suffices to check the largest.
The repository's score is a more conservative bound using
`lambda_max <= 4*(1.01/h)²`. We independently checked the energy identity
against a numerical update of a small generic system; this is not a substitute
for peer review of the derivation.

This mathematical guarantee is for this bounded model and exact arithmetic:
roundoff/overflow, incorrect inputs, changing parameters mid-trajectory,
unmodelled forcing, gradient reconstruction, and discretization or perceptual
accuracy require separate checks. Positive energy also does not imply an
accurate solution. The code still needs explicit domain validation and a full
four-second run at the planned rate before making a broad implementation claim.

## Short GPU checks

Jobs 34566921 and 34567084 ran through `hpc_sonic` and completed successfully.
`test/guitar_high_rate_smoke.py` compares a nonlinear raised-cosine CPU
forward Jacobian against the one- and three-scalar cached CUDA implementations,
including all eight gradients of a normalised mean-square loss. Each simulation
is only 20 ms. An effective homogeneous material reproduces the measured open
pitch and B for nylon high E; the grid stays fixed at 176 intervals while
the length is shortened to frets 0, 12, and 24.

At 34×, with sigma1=0.0001, all three cases passed; maximum CPU/CUDA
relative L2 differences were `9.82e-10` for waveforms and `1.78e-9` for
gradients. At 55×, with sigma1=0.001, they passed together with the independent
box's worst corner (`f0=1318.4`, `B=.001984`, `L=.1625`). Its stability
score is `0.98815`; its waveform and gradient relative L2 differences were
`1.82e-10` and at most `2.63e-10`. The maximum differences across all four
55× cases were `4.69e-10` for waveforms and `2.60e-9` for gradients. These
compare CUDA to the independently differentiated CPU recurrence, not a
direct Octave run at 55×, and are not speed, memory, or long-duration results.
The JSON files are in `benchmarks/guitar_stability_results/`.

Job 34568141 adds a direct **20 ms** comparison of that conservative box
corner against the actual non-SAV Octave submodule at 55×, using its raised
cosine, 176 intervals and the same physical parameters, including sigma1=.001.
The raw midpoint waveform's relative L2 difference was 8.65e-11 for CPU and
1.50e-10 for both cached CUDA variants; their respective maximum absolute
differences were 1.84e-13 m and 3.35e-13 m. The [result JSON](../benchmarks/guitar_stability_results/octave-box-corner-55x-34568141.json)
records the complete parameters. This is a synthetic independent-box corner,
not necessarily a physical guitar material, and neither a four-second Octave
comparison nor a high-rate Octave gradient check.

## Modal-frequency check at 55×

`python benchmarks/guitar_damped_pole_study.py --oversampling 55` compares
the exact damped linear poles with the damped continuous stiff-string model
for the 300 measured-string/fret scenarios above. It assumes sigma0=1/s and
sigma1=0.0001 m²/s, not a measurement of those damping parameters. All 300
linearized cases are stable at 55×. Among the represented modes below 20 kHz,
the maximum absolute errors across applicable cases are 0.023 cents for mode
1, 0.209 for mode 3, 0.596 for mode 5, 2.64 for mode 10, and 12.87 for mode
20. Far higher partials can be shifted far more, and some modes are absent on
the finite grid. Results are in `benchmarks/guitar_damped_pole_results_55x/`.
This small-amplitude modal comparison is not a nonlinear-audio or audibility
test. Raising the temporal sampling rate establishes the stated stability
bound but does not remove spatial dispersion.

## Four-second single-case timing

Job 34567487 ran the conservative independent box corner at 55×, four seconds,
one string, raised cosine, on an NVIDIA A100 80 GB PCIe. The raw midpoint
output and normalised mean-square loss were used; filtering and decimation were
not included. After a warm-up, the one-scalar version took 13.17 s forward
and 15.79 s backward (28.96 s total), and the three-scalar version took 13.30 s
forward and 13.15 s backward (26.45 s total). The respective peak PyTorch
allocated tensor memory values were 0.549 and 0.694 GiB; these exclude some GPU
process overhead and must not be compared directly with an NVML process-memory
figure. The timing is a single run, not a batch-128 or whole-domain benchmark.
The two waveforms have the same reported loss and their gradients are close;
the full-duration result has not been independently checked against Octave.
