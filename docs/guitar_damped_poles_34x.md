# Damped-pole example at 34×

This is an analytical small-amplitude study of the **same non-SAV FDTD
equations** used by both the upstream reference and our optimized method.
At 176 intervals it uses an internal rate of 34 × 44,100 = **1,499,400 Hz**,
length 0.65 m for the open strings, and illustrative damping parameters
`sigma0 = 1 s⁻¹` and `sigma1 = 0.0001 m²/s`. It does not by itself prescribe
a universal guitar configuration or measure the nonlinear solver's output.

The 300 scenarios start from the twelve measured nylon/new-steel open-string
frequencies and inharmonicities in [Woodhouse et al., Table I](https://www.lam.jussieu.fr/Publications/Articles/AAA_Woodhouse_2012.pdf),
then include frets 0–24 with ideal shortening, unchanged tension, mass per
length and effective bending stiffness. In other words the fretted examples
are estimates, not measured guitars; the first partial changes slightly from
equal temperament when inharmonicity grows. These samples do not cover all
guitars, damping parameters, excitation amplitudes, or optimization paths.

For each case the script checks every discrete mode's exact damped recurrence
roots; all 300 have magnitude below 1. It then compares each underdamped
discrete mode with the *damped continuous stiff-string* frequency for the
same parameters, up to a 20 kHz continuum cutoff. This is numerical error on
top of physical inharmonicity, not a comparison with integer harmonics.

| Mode | Largest absolute frequency error | Applicable scenarios |
|---:|---:|---:|
| 1 | 0.0230 cents | 300 |
| 3 | 0.2085 cents | 300 |
| 5 | 0.5939 cents | 300 |
| 10 | 2.614 cents | 300 |
| 20 | 12.812 cents | 281 |
| 50 | 94.807 cents | 148 |
| 100 | 382.957 cents | 38 |
| 175 | 1001.609 cents | 2 |

Only cases where the physical partial is below 20 kHz, is represented on the
grid, and its numerical frequency is below the 22.05 kHz output Nyquist are
included in the maxima. Modes that are above the physical cutoff or missing
from the grid are not assigned an error; the latter are explicitly marked in
the per-mode data. For the earlier centered raised-cosine excitation and
midpoint pickup, the even modes cancel by symmetry.

The 34× rate addresses the *time-step stability* of these selected scenarios.
It does not add grid modes or eliminate spatial dispersion. For example,
mode 175 for the nylon low E at the first fret is 19,642.68 Hz in the damped
continuum model but 11,013.84 Hz in the FDTD. This numerical discrepancy
is shared with the reference at the same settings; neither the example nor
the pole table alone establishes audibility. Changes to the rate or grid need
new performance benchmarks and matched-reference validation.

Run `python benchmarks/guitar_damped_pole_study.py` from the repository root
to regenerate [case results](../benchmarks/guitar_damped_pole_results/cases.csv)
and [per-mode results](../benchmarks/guitar_damped_pole_results/modes.csv.gz).
The script imports the [previous survey](../benchmarks/guitar_modal_dispersion.py)
for the reference parameters and spatial operator. It uses only the standard
library. Pole analysis assumes a fixed grid, fixed damping, and infinitesimal
excitation; nonlinear stability and practical accuracy remain separate tests.
