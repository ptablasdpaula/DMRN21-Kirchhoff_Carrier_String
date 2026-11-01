# Reference fidelity and numerical accuracy

The reference is [Russo, Ducceschi and Bilbao (2026)](https://link.springer.com/article/10.1007/s11071-026-12708-0),
which compares split and non-split scalar auxiliary variable (SAV) schemes for
several nonlinear **stiff** strings. Its Kirchhoff–Carrier reference integrator
is in section 8.3, equation (73). The paper emphasizes that numerical energy
stability alone does not guarantee convergence; its SAV results must not be
attributed automatically to a different method.

Our implementation follows the companion MATLAB file
`KCStringSAV_noSplit.m` with **`SAV = false`**, i.e. its non-SAV reference
branch, plus the project's damping and parameter/gradient infrastructure. The
filename alone should not be taken to mean that our solver uses the non-split
SAV algorithm. Any modified spatial or temporal discretization is a new
numerical method for the same continuum model, not the exact same discrete
scheme as this baseline.

## Stability wording to reconcile

The paper describes its KC reference as unconditionally stable. Our local
fixed-grid code explicitly enforces a CFL-like condition. In the zero-amplitude,
undamped linearization of the local update, each mode follows
`x[n+1] = (2 - (omega_h*k)^2)*x[n] - x[n-1]`, whose linear amplification
condition is `omega_h*k <= 2`. At the nominal parameters, 176 intervals,
44.1 kHz and mode 175, `omega_h*k` is about 2.624, with a linearized root of
approximately -4.673. The original MATLAB branch also sets its grid using a
stability-related bound. The exact assumptions/meaning of the paper's claim
and the implemented damped scheme should be reconciled before transferring
the claim to a new configuration; this observation does not settle all possible
nonlinear stability definitions or prove an error in the paper.

## Accuracy options

1. Keep the paper-based discrete scheme and refine space and time, checking
   convergence at fixed physical parameters and consistent boundary conditions.
   This retains discrete-scheme fidelity but can be expensive; the example in
   `guitar_parameter_grid_study.md` separates mode count, stability and modal
   frequency error.
2. Experiment with a sine-spectral spatial operator for the uniform simply
   supported KC equation. Exact continuous sine eigenvalues avoid the current
   finite-difference spatial dispersion for retained modes; a consistent
   discretization of bending, nonlinear coupling and startup, and a newly
   derived adjoint, must be validated. The existing diagonal/global-coupling
   structure suggests opportunities for an efficient implementation, not a
   guarantee that the current CUDA kernel can be reused unchanged. This is an
   alternative discretization, not identical output to the original FDTD.
3. Higher-order spatial differences improve dispersion in the well-resolved
   band but do not resolve a mode at the grid Nyquist accurately. Their
   stability limits and boundary treatments must be recomputed. Exact or
   implicit linear time integration may address part of the timestep
   restriction, but nonlinear temporal error, stability and aliasing remain.

**Adopted (31 October 2026): option 3.** The five-point fourth-order second
difference, closed by odd reflection at the simply supported ends, is now the
project default (`SPATIAL_ORDER = 4`); see `docs/numerics.md` and
`docs/accuracy_convergence_study.md`. The stability bound is the reference's
with h → h/√(4/3); the three-point scheme stays available as `order=2`, which
is what the statements above about 176 intervals at 44.1 kHz describe.
`test/reference_paper_exact.py --order 4` substitutes the same operator into
the Octave code so the fourth-order variant is checked against a modified
reference rather than left unvalidated.

Validate linear poles and decay rates across the target guitar domain; then
compare nonlinear trajectories, spectra, outputs and all eight physical
gradients under independent spatial and temporal refinement. Include varied
plucks/forcing and pickup positions: the current eight-mode initial condition
and midpoint pickup leave much of the full spatial basis untested. Compare
small cases to an independent high-accuracy ODE integrator in addition to the
paper-based reference; compare physical realism against measured string data.
