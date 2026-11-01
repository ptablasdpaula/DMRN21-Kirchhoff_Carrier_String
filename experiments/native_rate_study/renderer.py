"""Pinned jaxdiffmodal integration with one fixed physical initial shape.

The upstream integrator is unchanged. The displacement profile named by
``excitation`` is sampled on the fixed reference grid (176 intervals by
default) and projected onto the first ``modes`` continuous string modes; the
modal count never changes how the initial displacement is generated or
normalized. Modes whose continuous frequency lies above half the rate are
still integrated exactly by the upstream recurrence but alias; the caller
chooses ``modes`` accordingly.
"""
import jax
import jax.numpy as jnp
from jaxdiffmodal.ftm import StringParameters, damping_term, stiffness_term, string_eigenvalues
from jaxdiffmodal.time_integrators import make_tm_nl_fn, solve_tf, string_tau_with_density

from experiments.figure_comparison.diaz import initial_shape

jax.config.update('jax_enable_x64', True)


def _single(values, samples, rate, modes, reference_intervals, excitation):
    rho, tension, radius, modulus, length, amplitude, sigma0, sigma1, pluck, pickup = values
    area = jnp.pi * radius**2
    inertia = jnp.pi * radius**4 / 4.
    mass = rho * area
    params = StringParameters(A=area, I=inertia, rho=rho, E=modulus,
                              d1=2*mass*sigma0, d3=2*mass*sigma1,
                              Ts0=tension, length=length)
    indices = jnp.arange(1, modes+1, dtype=values.dtype)
    eig = string_eigenvalues(modes, length)
    positions = jnp.arange(1, reference_intervals, dtype=values.dtype) * length/reference_intervals
    basis = jnp.sin(jnp.pi * indices[:,None] * positions[None,:] / length)
    shape = initial_shape(reference_intervals, excitation, values.dtype, center=pluck)
    u0 = (length/reference_intervals)*(basis @ (amplitude*shape))
    factors = string_tau_with_density(params)*eig/(length/2.)
    nl = make_tm_nl_fn(eig, factors)
    _, trajectory, _ = solve_tf(damping_term(params,eig), stiffness_term(params,eig),
                                1./rate, n_steps=samples, nl_fn=nl,
                                u0=u0, v0=jnp.zeros_like(u0))
    weights = jnp.sin(jnp.pi*indices*pickup)/(length/2.)  # modal readout at the pickup fraction
    return trajectory @ weights


def renderer(values, samples, rate=44100, modes=175, reference_intervals=176, excitation='multimode'):
    if values.ndim != 2 or values.shape[1] != 10: raise ValueError('Expected physical parameters (batch, 10)')
    if not isinstance(reference_intervals, int) or reference_intervals < 4 or reference_intervals % 2:
        raise ValueError('Use an even reference interval count >= 4')
    if not isinstance(modes, int) or not 1 <= modes <= reference_intervals-1 or samples <= 0 or rate <= 0:
        raise ValueError('Use 1 <= modes <= reference_intervals-1, positive samples and rate')
    return jax.vmap(lambda row:_single(row,samples,rate,modes,reference_intervals,excitation))(values)
