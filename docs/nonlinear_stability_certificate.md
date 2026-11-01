# A sufficient nonlinear stability condition for the non-SAV reference

This note concerns the **damped `SAV=false` branch** of the submoduled
[`KCStringSAV_noSplit.m`](../nLinStringsConv-SAV/KCstring/KCStringSAV_noSplit.m).
It derives a sufficient condition for bounded, unforced solutions of its
finite-grid update. It does not establish convergence, perceptual accuracy,
or stability of the separate SAV branch.

Let `u^n` be the interior displacement vector, `k` the timestep, `h=L/N`,
`D=-Dxx` and `Q=T0 D + EI D²`. The pinned-end matrix `D` is symmetric positive
definite. Write `mu=rho A`, `c²=T0/mu`, `K²=EI/mu`, and
`v^n=(u^{n+1}-u^n)/k`. All norms below use the ordinary Euclidean inner
product, and `||x||²_D=xᵀDx`. Take positive physical parameters, finite
initial states, nonnegative damping coefficients `sigma0` and `sigma1`,
and no external forcing.

The branch uses a rank-one implicit nonlinear update. Its discrete energy,
including a correction for the time-discretisation of frequency-dependent
damping, is

```text
E_n = mu h/2 ||v^n||²
    + h/2 (u^{n+1})ᵀ Q u^n
    + EA h²/(8L) [(u^{n+1})ᵀ D u^n]²
    - mu h k sigma1/2 ||v^n||²_D .
```

The update gives the exact-arithmetic balance

```text
E_n - E_{n-1} = -2 mu h k (sigma0 ||w^n||² + sigma1 ||w^n||²_D) <= 0,
w^n = (u^{n+1} - u^{n-1})/(2k).
```

The `sigma1` correction follows from
`(w^n)ᵀD v^{n-1} = ||w^n||²_D - (||v^n||²_D - ||v^{n-1}||²_D)/4`.
The original code's `En` tracks its kinetic, linear and nonlinear terms plus
accumulated loss; the corrected `E_n` above instead has an explicitly
non-increasing energy balance.

Set `m=(u^{n+1}+u^n)/2` and `d=u^{n+1}-u^n`. The quadratic part of `E_n`
becomes

```text
h/2 [mᵀ Q m + dᵀ (mu/k² I - Q/4 - mu sigma1 D/k) d].
```

The nonlinear term is a nonnegative square. Therefore a sufficient *strict*
condition for coercivity is

```text
k²(c² lambda_max + K² lambda_max²) + 4k sigma1 lambda_max < 4,
lambda_max = 4/h² sin²((N-1)pi/(2N)).
```

`Q` and `D` have the same eigenvectors, and the expression increases with
the eigenvalue when the physical parameters and damping are nonnegative. The
condition makes `E_n` nonnegative and bounds both `u^n` and `u^{n+1}` by the
finite initial energy for all steps. It applies at any finite excitation
amplitude: an additional amplitude-dependent effective-tension CFL factor
is not needed for this discrete energy argument. Equality is not covered.

A convenient, slightly more conservative sufficient condition uses
`lambda_max < 4/h²`:

```text
(c² k² + 4 sigma1 k)/h² + 4 K² k²/h⁴ < 1.
```

The repository's `GridConfig.physical_score` uses this bound with an additional
1% grid-spacing margin. To guarantee the condition over a parameter region,
one must actually bound all terms over that region, including length and
`sigma1`; testing a few representative strings is not such a proof. A fixed
grid and timestep should be held over each optimisation domain. The argument
does not cover forcing, negative damping or parameters, changing the timestep
or grid during a trajectory, floating-point overflow, nonlinear-solver changes,
or stability of the separately implemented backward reconstruction.

## Reproducible numerical check

Run `python test/nonlinear_energy_certificate.py`. The standard-library-only
check independently evaluates the spatial update for 256 steps, both with and
without each damping term, and for 2 mm and 20 mm initial amplitudes. On the
test platform, the largest balance residual was `5.24e-15 J` (against an
initial energy `0.236 J`); all eight cases passed. This checks the algebra and
implementation numerically at selected points; it is not itself a proof of
the parameter range or of numerical accuracy.
