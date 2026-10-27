# Numerical conventions and scope

The production solver preserves the non-SAV update and initial step of
`nLinStringsConv-SAV/KCstring/KCStringSAV_noSplit.m`, with simply supported ends.
The first-mode profile remains available for the Octave comparison. Multimode
uses the first eight sine modes weighted by 1/n² and scaled to the peak amplitude;
raised cosine uses the reference node-centred quarter-string width convention.
The listener is fixed at the midpoint, suppressing even spatial modes.

## Parameterisation

Let μ=ρπr² and I=πr⁴/4. The six controls map to T=4μL²f₀² and
E=BTL²/(π²I). The continuous small-amplitude modal frequencies are
fₙ=n f₀√(1+Bn²). Density and length are fixed, not optimised. Positive B is required
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

## Fixed-grid selection

For k=1/(fs·OF), c²=T/μ, K²=EI/μ, and H=(L/N)/1.01, the checked score is

S_margin = (c²k² + 4σ₁k)/H² + 4K²k²/H⁴ ≤ 1.

The maximum over a rectangular f₀,B,σ₁ range is its upper corner. Other controls
are absent from this score. The corresponding unadjusted score cannot simply be
compared against 1/1.01²: bending carries a fourth-power margin.

The minimum N is determined from nominal reference parameters at 44.1 kHz, ×1,
using the reference damping and even rounding. The chosen factor is the smallest
positive integer supporting that N over the box; the chosen N is the largest even
stable value at that factor. There is no post-selection continuous adjustment of
k. The MATLAB-comparison path separately preserves the reference's adjustment and
reported floor(1/k) sampling rate.

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
