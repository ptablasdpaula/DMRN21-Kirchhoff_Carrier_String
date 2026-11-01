# Guitar-string parameter coverage and resolution

This is a preliminary study, not a declaration that the current model covers
all guitar strings. A bounded instrument/tuning/fret range, frequency band,
frequency/phase error and amplitude/forcing range are needed before selecting
one global grid and sample rate. In particular, a mode count by itself is not
an accuracy criterion.

## Evidence and modelling limits

[Woodhouse et al., Table I](https://www.lam.jussieu.fr/Publications/Articles/AAA_Woodhouse_2012.pdf)
measure two six-string guitar sets. Their open-string frequencies are about
82.4–329.6 Hz; tensions are 53.4–73.9 N for the nylon set and 121–185 N for
the new steel set, and measured B values are 13–124 × 10⁻⁶ across those sets.
These are examples, not universal bounds. Extended-range guitars reach lower:
the [Ibanez RG8 specification](https://www.ibanez.com/usa/products/detail/rg8_1p_08.html)
lists a low F tuning and 686 mm scale. If bass guitars are included the lower
pitch and longer-scale range must be specified separately.

Parameters are correlated. In the uniform-string approximation,
`f0 = sqrt(T/mu)/(2L)` and `B = pi² EI/(T L²)`. Fretting changes the vibrating
length and therefore both f0 and B. A rectangular box of independently varied
extremes may include unrealistic strings. A representative family of measured
string constructions, scale lengths, tunings and fret positions is a better
starting domain, with explicit safety margin for optimisation.

Wound strings are a separate physical issue: the current homogeneous
circular model links mass per length, bending rigidity and axial stiffness to
one density, radius and Young's modulus. These effective properties need not
obey the same relationships for wound or multifilament strings. Experiments
show simple stiffness estimates agree much better for monofilaments than wound
strings. [Murray and Whitfield (2022)](https://doi.org/10.1119/5.0064373).
Expanding the current box does not by itself resolve that model mismatch.

## Illustrative 20 kHz calculation

In the small-amplitude continuous stiff-string model,
`f_n = n*f0*sqrt(1+B*n²)`. For an output-band cutoff F, `n_max` is the largest
integer satisfying `f_n <= F`. This is a model-based mode count; it does not
tell us whether a mode is excited, audible, accurately discretised or aliased
by a nonlinear solver. It also depends on pitch, B, and the selected band.

Using 20 kHz only as an illustrative cutoff:

* The published nylon low-E example, first partial about 82.4 Hz and measured
  B = 19 × 10⁻⁶, has approximately 187 modes below 20 kHz. Treating its
  measured first partial as the tension-only f0 is a negligible approximation
  at this B. The corresponding new steel low-E example, B = 106 × 10⁻⁶,
  gives approximately 139.
* Even the **existing** box includes f0 = 60 Hz, B = 10⁻⁵, giving 258 modes.
  The minimum even spatial grid that can merely represent all of them is
  260 intervals (259 interior modes).
* For that 260-interval grid, the existing linear stability check over its
  current box (f0 <= 100 Hz, B <= 0.001, sigma1 <= 0.001 m²/s, L = 1 m and
  a 1% spacing margin) fails at 6× and first passes at **7×**, or 308.7 kHz.
  This is not a bound for a broader guitar domain or nonlinear stability.

But 260 intervals are not enough to resolve those highest modes accurately.
At the lower-pitch corner in the undamped linear discrete recurrence, mode
257 is approximately 19,870 Hz in the continuum and 11,232 Hz on that grid
at 7×: about 43% low. We use odd mode 257 because the midpoint pickup cancels
even modes. This is a deliberate worst-region example, not an assertion about
the energy of that mode in our current excitation.

For illustration, demanding less than 1% spatial-wavenumber error at mode 258
requires 1,654 intervals; the same global current-box stability check then
requires 256×. Applying a 1% criterion to squared wavenumber instead requires
2,336 intervals and 509×. These spatial-only criteria do not constitute an
end-to-end frequency or nonlinear-audio accuracy guarantee; temporal error
can offset or worsen spatial error. The full discrete poles and convergence
should be checked for the actual chosen parameter family and frequencies.
Reproduce the above values with `python benchmarks/guitar_grid_explore.py` in
the project environment.

The current cached CUDA backend permits at most 256 interior modes, so even
the 260-interval minimum would require a kernel redesign. The default grid
selector also caps oversampling at 128×.

## A comparison at 44.1 kHz

The 3× FDTD rate was chosen for its own grid and stability requirements; it
does not automatically apply to `jaxdiffmodal`. A separate native-rate pilot
tests the genuine modal method at 44.1 kHz with 8, 123 and 175 modes and the
same initial displacement, physical values and eight gradients, against an
oversampled reference. The nominal continuous string has about 123 modes
below 22.05 kHz; the exact count varies with the physical parameters.
High-frequency modes and nonlinear sidebands above Nyquist can alias.

The current test's initial displacement contains only the first eight sine
modes. In the ideal uniform unforced Kirchhoff–Carrier modal equations, a
mode initially at zero remains zero: the nonlinear tension is shared but
does not create a new spatial mode from exactly zero. Consequently, an
eight-mode result could be adequate for this specific input; it is not an
adequacy claim for arbitrary plucks, external forces or other nonlinearities.
Use waveform, spectrum, loss and all-eight-gradient checks to establish what
is appropriate for the intended workload.
