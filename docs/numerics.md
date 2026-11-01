# Numerical conventions and scope

The production solver preserves the non-SAV update and initial step of
`nLinStringsConv-SAV/KCstring/KCStringSAV_noSplit.m`, with simply supported ends.
The first-mode profile remains available for the Octave comparison. Multimode
uses the first eight sine modes weighted by 1/n² and scaled to the peak amplitude;
raised cosine uses the reference node-centred quarter-string width convention.
The listener (pickup) defaults to the midpoint, suppressing even spatial modes; it is the tenth
differentiable parameter, a fraction of L read through the sine interpolant of the interior nodes,
and the raised-cosine pluck centre is the ninth (the reference's L/2 by default).

## Parameterisation

Let μ=ρπr² and I=πr⁴/4. The six controls map to T=4μL²f₀² and
E=BTL²/(π²I). The continuous small-amplitude modal frequencies are
fₙ=n f₀√(1+Bn²). Density and length are fixed in perceptual mode. The optional direct-material mode instead exposes all ten native parameters: the eight physical ones, including density and length, plus the pluck and pickup positions, without this mapping. Positive B is required
by the bounded fitting layer; the reference linear-limit checks isolate the linear
update separately. B=0 is not a finite log-coordinate endpoint.

The continuous model admits density/material and length/material scaling
invariances. The preserved MATLAB startup contains additional h factors, however,
and breaks the latter invariance in the discrete output. Do not claim two exact
null directions for this complete implementation. Nor does fixing those scales
prove practical identifiability of the remaining six parameters. Radius is sensed
through nonlinear motion, and a first-mode excitation is particularly restrictive.

The indicator B(a/r)² is the initial fractional tension increase for a continuous
first-mode displacement a sin(πx/L). It is not the corresponding expression for
an arbitrary multimode or raised-cosine initial shape, nor an accuracy certificate.

## Spatial stencil

The spatial second difference is the five-point fourth-order stencil
(−1, 16, −30, 16, −1)/12h² by default (`SPATIAL_ORDER = 4` in
`src/kc_string_forward.py`), closed at the simply supported ends by odd
reflection (u₋₁ = −u₁); the fourth difference is its square. The reference
code's three-point stencil remains available as `order=2` everywhere
(`ForwardString`, `initialize`, `GridConfig`, the adjoint wrappers) and is what
the Octave comparisons use. Both operators are diagonal in the same discrete
sine basis with eigenvalues −4s² (order 2) and −4s²(1 + s²/3) (order 4),
s = sin(πn/2N), from `discrete_eigenvalues`; every kernel reads its operator
from that table, so the time step, rank-one solve and adjoint are unchanged.
The five-point stencil's largest eigenvalue is 4/3 of the three-point value,
so its stability bound is the reference's with h → h/√(4/3). On the benchmark
grid it reduces the waveform error against the converged solution from 11.7 %
to about 1 % at 2× oversampling (`docs/accuracy_convergence_study.md`).

## Fixed-grid selection

For k=1/(fs·OF), c²=T/μ, K²=EI/μ, H=(L/N)/1.01 and the stencil's eigenvalue
factor γ (1 for order 2, 4/3 for order 4), the checked score is

S_margin = γ(c²k² + 4σ₁k)/H² + 4γ²K²k²/H⁴ ≤ 1.

The maximum over a rectangular f₀,B,σ₁ range is its upper corner.
For direct material bounds, use rho_min, T_max, E_max, L_min and sigma1_max.
The radius contribution has the form A/r²+C r²; convexity in r² means the maximum
occurs at one of the two radius endpoints. Both are checked when selecting N and
oversampling. If length is fitted, physical h=L/N varies continuously with length;
N and k remain fixed, and this variation is included in the native derivatives. Other controls
are absent from this score. The corresponding unadjusted score cannot simply be
compared against 1/1.01²: bending carries a fourth-power margin.

The minimum N is determined from nominal reference parameters at 44.1 kHz, ×1,
using the reference damping and even rounding: 148 intervals with the
three-point stencil, 128 with the five-point one. The chosen factor is the smallest
positive integer supporting that N over the box; the chosen N is the largest even
stable value at that factor. The perceptual box therefore selects ×3 with 152
intervals at order 4 (176 at order 2). There is no post-selection continuous
adjustment of k. The MATLAB-comparison path separately preserves the reference's
adjustment and reported floor(1/k) sampling rate.

`GridConfig(box, oversampling=, intervals=)` fixes a grid explicitly; it must
still satisfy the margin-adjusted bound at the box corner. The timing benchmark
uses `BenchmarkBox` (every control within exp(±0.05) of the nominal string) with
176 intervals at ×2 (88.2 kHz), corner utilisation 0.87, a grid the full fitting
box's corner would reject.

Stability selection is not an accuracy guarantee. Bilbao's *Numerical Sound
Synthesis*, §7.2.3 discusses the stiff-string bound; §8.1.2, pp.227–229 explains
why stable nonlinear schemes may still show spurious spatial oscillations.
The notebook's modal diagnostics compare the damped continuous linear poles with
the roots of the actual discrete linear recurrence. A nonoscillatory mode has no
cents comparison. These diagnostics do not measure nonlinear pitch-glide accuracy.

The reference damping fit uses 6 ln(10). For the stated amplitude damping operator,
conventional amplitude T60 uses 3 ln(10), so the reference's nominal 15 s and 10 s
fit corresponds approximately to 7.5 s and 5 s under the small-loss approximation.
The main controls fit σ₀ and σ₁ directly; no T60 label substitutes the nominal values.

## Differentiation and validation

The native derivatives are first-order derivatives of the discrete calculation,
up to floating-point error. They are not exact derivatives of the continuous PDE.
Eight material tangent columns remain in the kernel; PyTorch chains the six-control
mapping and bounded transforms. The API deliberately rejects higher derivatives.

The short linear test disables the nonlinear direction and compares the unchanged
native recurrence against an independent closed-form modal solution, including
its derivatives in c², K², L and amplitude and the reference startup factor.
The nonlinear test uses finite amplitude, central differences and Taylor remainder.
Discrete energy is checked for undamped multimode runs at three amplitudes.
These tests establish numerical identities and derivative correctness, not uniform
physical accuracy throughout every possible parameter box.

The reference paper is [Russo, Ducceschi & Bilbao (2026)](https://link.springer.com/article/10.1007/s11071-026-12708-0).
Its SAV convergence conclusions should not be described as tests performed on this
non-SAV implementation. The tutorial preserves the full derivations separately.
