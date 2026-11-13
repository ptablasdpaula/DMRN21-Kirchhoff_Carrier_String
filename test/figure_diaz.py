"""Independent small upstream and finite-difference checks for Díaz adapter."""
import os
os.environ.setdefault('JAX_ENABLE_X64', 'true')
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jaxdiffmodal.ftm import StringParameters, damping_term, stiffness_term, string_eigenvalues
from jaxdiffmodal.time_integrators import make_tm_nl_fn, solve_tf, string_tau_with_density
from benchmarks.exact_methods import BOX, varied_parameters
from experiments.figure_comparison.diaz import material_parameters, renderer
from experiments.jax_loss_benchmark import anti_alias, cel, jax_loss, prepare_target
from src.kc_string_forward import PARAMETER_COUNT
from src.parameters import MaterialParameterBox


def upstream_reference(x, steps, intervals, rate):
    rho, tension, rad, young, length, amp, s0, s1, _, pickup = x
    area, inertia = jnp.pi * rad**2, jnp.pi * rad**4 / 4
    mass = rho*area
    params = StringParameters(A=area, I=inertia, rho=rho, E=young,
        d1=2*mass*s0, d3=2*mass*s1, Ts0=tension, length=length)
    k = jnp.arange(1, intervals, dtype=x.dtype)
    grid = length * jnp.arange(1, intervals, dtype=x.dtype) / intervals
    initial = sum(jnp.sin(jnp.pi*grid*m/length)/m**2 for m in range(1, min(8, intervals-1)+1))
    normalizer = jnp.max(jnp.abs(initial))
    # The 8-mode shape has analytic orthogonality on this full discrete grid.
    modal_initial = jnp.where(k <= 8, amp * length/(2*normalizer*k**2), 0.)
    lam = string_eigenvalues(intervals-1, length)
    fn = make_tm_nl_fn(lam, string_tau_with_density(params)*lam/(length/2))
    _, traj, _ = solve_tf(damping_term(params, lam), stiffness_term(params, lam),
        1/rate, n_steps=steps+2, nl_fn=fn, u0=modal_initial, v0=jnp.zeros_like(k))
    # Modal readout at the pickup fraction (the midpoint node when pickup = 1/2).
    return traj @ (jnp.sin(jnp.pi*k*pickup)/(length/2))


def main():
    # Compare the exact function used by the adapter against the native
    # MaterialParameterBox ordering, both values and Jacobians. The f0/B
    # inputs only select matched test strings outside differentiation.
    sampled = torch.cat((BOX.defaults()[None], varied_parameters(8)), dim=0)
    box = MaterialParameterBox()
    for reference in sampled:
        material = BOX.to_material(reference)
        row = material
        jrow = jnp.asarray(row.numpy(), dtype=jnp.float64)
        expected = box.to_material(row)
        expected_jac = torch.func.jacfwd(box.to_material)(row)
        actual = material_parameters(jrow)
        actual_jac = jax.jacfwd(material_parameters)(jrow)
        np.testing.assert_allclose(np.asarray(actual), expected.numpy(), rtol=2e-14, atol=1e-14)
        np.testing.assert_allclose(np.asarray(actual_jac), expected_jac.numpy(), rtol=2e-13, atol=1e-10)
    print('PASS: default and varied physical parameter values and Jacobians match BOX')

    ref = torch.tensor([110., 1e-4, 1e-3, 4e-4, 1., 1e-4], dtype=torch.float64)
    native = BOX.to_material(ref)
    assert native.shape == (PARAMETER_COUNT,)
    x = jnp.asarray(native.numpy(), dtype=jnp.float64)
    steps, intervals, rate = 1024, 24, 132300
    render_one = jax.jit(lambda a: renderer(a[None, :], steps, intervals, rate)[0])
    for candidate in (x, x.at[0].multiply(1.08).at[4].multiply(.92), x.at[9].set(.55)):
        actual = render_one(candidate)
        expected = upstream_reference(candidate, steps, intervals, rate)
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-12, atol=2e-15)
    # Test every differentiable coordinate in dimensionless relative units;
    # centered finite differences independently probe the upstream reverse VJP.
    # An off-centre pluck and pickup keep the two position derivatives nonzero
    # (the multimode profile ignores the pluck; a raised cosine moves with it).
    def objective(v, excitation):
        wave = renderer(v[None, :], steps, intervals, rate, excitation)[0]
        return jnp.mean(wave**2) * 1e6
    probe = x.at[8].set(.45).at[9].set(.55)
    for excitation in ('multimode', 'raised_cosine'):
        value_grad = jax.jit(jax.value_and_grad(lambda v: objective(v, excitation)))
        value, grad = value_grad(probe)
        eps = 1e-4
        num = []
        eval_forward = jax.jit(lambda v: objective(v, excitation))
        for i in range(PARAMETER_COUNT):
            e = jnp.zeros_like(probe).at[i].set(eps*probe[i])
            num.append((eval_forward(probe+e)-eval_forward(probe-e))/(2*eps*probe[i]))
        analytic, numeric = np.asarray(grad*probe), np.asarray(jnp.stack(num)*probe)
        np.testing.assert_allclose(analytic, numeric, rtol=2e-3, atol=1e-7)
        assert np.isfinite(np.asarray(value))
        if excitation == 'raised_cosine':
            assert abs(analytic[8]) > 0 and abs(analytic[9]) > 0
        else:
            assert analytic[8] == 0 and abs(analytic[9]) > 0
    print('PASS: genuine upstream TF reference, batch 1, all ten finite-difference coordinates')
    print('scaled analytic', analytic)
    print('scaled finite differences', numeric)

    # Verify summing a full group plus a partial group reproduces the
    # independent all-item objective and waveform VJP after only one filter.
    rng = np.random.default_rng(41)
    wave = jnp.asarray(rng.normal(size=(17, 3074)))
    target = jnp.asarray(rng.normal(size=1024))
    kernel = jnp.asarray([.1, .2, .4, .2, .1])
    prepared = prepare_target(target)
    def grouped(raw):
        filtered = anti_alias(raw, kernel, 1024, 3)
        return 16*cel(filtered[:16], prepared) + cel(filtered[16:], prepared)
    def whole(raw):
        return 17*jax_loss(raw, prepared, kernel, 3)
    vg1 = jax.jit(jax.value_and_grad(grouped))(wave)
    vg2 = jax.jit(jax.value_and_grad(whole))(wave)
    np.testing.assert_allclose(np.asarray(vg1[0]), np.asarray(vg2[0]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(vg1[1]), np.asarray(vg2[1]), rtol=1e-10, atol=1e-12)
    print('PASS: grouped and partial-group CeL value and waveform gradient')


if __name__ == '__main__':
    main()
