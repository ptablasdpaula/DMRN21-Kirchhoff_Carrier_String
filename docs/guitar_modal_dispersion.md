# Guitar pitch and grid-dispersion survey

This investigates the discrete equations shared by the submodule's **non-SAV**
method and our implementation. It does not compare a new physical scheme
against Russo's. The optimized method was separately compared directly with
the reference at the same parameters and grid; see
[paper_reference_validation.md](paper_reference_validation.md).

## What was calculated

The script [guitar_modal_dispersion.py](../benchmarks/guitar_modal_dispersion.py)
analytically diagonalizes the **undamped, linearized** recurrence at 176
intervals and an internal rate of 132,300 Hz, matching the four-second
reference test. It compares its frequencies with those of a continuous
small-amplitude stiff string with the same tension, bending stiffness, length
and mass per length. The reference frequencies already include real physical
inharmonicity; the table therefore shows only numerical frequency error, not
the physical difference from integer harmonics.

For mode `n`, `f_cont = n f0 sqrt(1+B n²)`, where the tension-only `f0` is
not quite the first partial: `f1=f0 sqrt(1+B)`. Let `lambda_n =
4N² sin²(pi*n/(2N))`. The normalized continuous-length discrete natural
angular frequency is `omega_h=2 f0 sqrt(lambda_n+B lambda_n²/pi²)`;
the leapfrog pole gives `f_disc=(fs/pi) asin(omega_h/(2fs))`. Error in cents
is `1200 log2(f_disc/f_cont)`. A negative value means the discrete partial is
lower than the continuous-model partial. The reference discretizes the same
operators, so this is also the reference FDTD's error at the matched grid.

We checked the **entire** linearized undamped grid for stability, not only
the modes below the desired output band. A case is unstable if the largest
spatial-mode ratio `omega_h/(2fs)` exceeds one; no pitch error is reported
for any mode of an unstable full grid. This condition does not guarantee
stability with damping, nonlinear motion, or parameter variation, and it omits
the project's 1% margin. It is therefore an optimistic screen. Physical modes
below 20 kHz that lie beyond the grid's 175 modes are flagged as missing;
represented modes whose discrete frequency exceeds the 22.05 kHz output
Nyquist are flagged separately.

## Results

Illustrative open nylon low E, using the measured 82.4 Hz first partial and
`B=19e-6` from [Woodhouse et al., Table I](https://www.lam.jussieu.fr/Publications/Articles/AAA_Woodhouse_2012.pdf):

| Mode | Error in cents |
|---:|---:|
| 1 | -0.022 |
| 3 | -0.197 |
| 5 | -0.547 |
| 10 | -2.193 |
| 20 | -8.832 |

Twelve continuous-model modes below 20 kHz are beyond the 175 grid modes.
The raised cosine and midpoint pickup used in the reference test cancel the
even spatial modes, so modes 10 and 20 in this table illustrate general modal
errors but do not contribute to that symmetric midpoint output. Mode frequencies
alone do not determine audibility or gradient relevance: excitation, pickup,
damping, nonlinear signal, loss and filtering matter.

The independent envelope samples first partial frequencies from A0 (27.5 Hz)
to E6 (1318.51 Hz) in quarter-semitone increments at `B=13e-6, 50e-6,
124e-6`. It includes E2–E6, the pitch envelope of a standard 24-fret six-string
guitar, and extends lower as a sensitivity study, not a complete bass or guitar
specification. These B values span measured **open-string examples**, but are
held constant while pitch varies, so the Cartesian combinations are not all
real strings. For the E2–E6 section, 262 of 579 sampled combinations pass this
optimistic undamped stability test. The largest sampled stable first partial
for each B is 344.22, 293.66 and 233.08 Hz, respectively. The exact limiting
first partial values are approximately 348.51, 294.62 and 235.09 Hz.

A second, correlated example takes the twelve measured nylon and new-steel
open strings in Woodhouse Table I and models frets 0–24 by shortening L while
holding tension, stiffness and mass per length fixed. For fret `j`, length
scales by `2^(-j/12)` and `B` by `2^(j/6)`; the first partial is computed
with the changed B, so it is not assumed exactly equal-tempered. Real fretting
also changes tension; these idealized cases are not measured fretted strings.
Only 94 of these 300 illustrative cases pass the undamped stability check at
this fixed grid and rate. These counts describe this illustrative set, not a
population of guitar strings. In particular, the measured nylon open high E
is unstable under this screen while the measured new-steel open high E passes.

The important distinction is between **one chosen discretization** and the
method itself. Instability at this 3× configuration does not mean the
reference or optimized solver cannot represent a note: a different matched
grid and rate may be selected and then validated. Nor does equality to the
reference imply that a grid is physically converged. Further simulation and
reference grid-refinement tests are needed to establish usable real-world
parameter ranges and sound-level impact.

## Reproducibility and outputs

Run `python benchmarks/guitar_modal_dispersion.py` from the repository root.
It requires only Python's standard library. Outputs are:

* [Quarter-semitone envelope summary](../benchmarks/guitar_modal_dispersion_results/envelope_summary.csv)
* [Measured-string idealized fret scenarios](../benchmarks/guitar_modal_dispersion_results/measured_strings_fret_scenarios.csv)
* [All individual modes, gzip-compressed CSV](../benchmarks/guitar_modal_dispersion_results/modes.csv.gz)

These calculations do not assert that a fixed domain covers every guitar,
that the resulting sounds are perceptually mistuned, or that the earlier
one-case GPU validation proves agreement over all parameters and excitations.
