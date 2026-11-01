"""Adapter for Díaz and Sandler's upstream jaxdiffmodal string solver.

This calls upstream ``solve_tf`` (including its full modal trajectory) and
``make_tm_nl_fn``.  It intentionally does not replace the integrator with a
streaming scan.  The continuous Laplacian and transfer-function time step are
different discretizations from our spatial FDTD update.

Install jaxdiffmodal revision 6980979dfb8a07b8664bddf7cf3e3d2f097d5ff0;
the cluster benchmark can add the scratch-only diaz-deps directory to
PYTHONPATH. Inputs use the project's ten native coordinates (eight material
parameters, then the pluck and pickup positions as fractions of L). Audio is
returned at the internal rate.
"""
import math
from pathlib import Path
import sys

import jax
import jax.numpy as jnp

from jaxdiffmodal.ftm import StringParameters, damping_term, stiffness_term, string_eigenvalues
from jaxdiffmodal.time_integrators import make_tm_nl_fn, solve_tf, string_tau_with_density

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.kc_string_forward import parse_excitation

jax.config.update("jax_enable_x64", True)


def initial_shape(intervals, excitation='multimode', dtype=jnp.float64, center=0.5):
    """JAX twin of ``src.kc_string_forward.excitation_shape`` on the N-1 interior nodes.

    The same unit-maximum profile that initialises the FDTD solver is sampled
    here, so that both solvers start from the same physical displacement; a
    raised cosine is centred at ``center`` (fraction of L, may be traced).
    """
    kind, fraction = parse_excitation(excitation)
    indices = jnp.arange(1, intervals, dtype=dtype)
    if kind == 'first_mode':
        return jnp.sin(jnp.pi * indices / intervals)
    if kind == 'multimode':
        excited = jnp.arange(1, min(8, intervals-1)+1, dtype=dtype)
        shape = jnp.sum(jnp.sin(jnp.pi * indices[:, None] * excited[None, :] / intervals)
                        / excited[None, :]**2, axis=1)
        return shape / jnp.max(jnp.abs(shape))
    width = max(1, math.floor(.25*(intervals-1))) if fraction is None else fraction*intervals
    distance = (indices - center*intervals) / width
    return jnp.where(jnp.abs(distance) <= 1, .5*(1+jnp.cos(jnp.pi*distance)), 0.)


def material_parameters(values):
    """Identity mapping from the ten direct coordinates.

    Return density, tension, radius, Young's modulus, length, amplitude,
    sigma0, sigma1, pluck and pickup position in the same order as
    ``MaterialParameterBox.to_material``.
    """
    return values


def _single(values, samples, intervals, rate, excitation='multimode'):
    rho, tension, radius, modulus, length, amplitude, sigma0, sigma1, pluck, pickup = material_parameters(values)
    area = jnp.pi * radius**2
    inertia = jnp.pi * radius**4 / 4.0
    mass = rho * area
    params = StringParameters(A=area, I=inertia, rho=rho, E=modulus,
                              d1=2.0*mass*sigma0, d3=2.0*mass*sigma1,
                              Ts0=tension, length=length)

    mode_indices = jnp.arange(1, intervals, dtype=values.dtype)
    eigenvalues = string_eigenvalues(intervals-1, length)
    # Project the same unit-maximum displacement used by our FDTD solver onto
    # the upstream sine basis. Its inverse STL divides by the sine-mode norm
    # L/2; the discrete midpoint quadrature is exact for the retained grid
    # modes at the spatial grid points.
    positions = jnp.arange(1, intervals, dtype=values.dtype) * (length/intervals)
    basis = jnp.sin(jnp.pi * mode_indices[:, None] * positions[None, :] / length)
    shape = initial_shape(intervals, excitation, values.dtype, center=pluck)
    u0 = (length/intervals) * (basis @ (amplitude * shape))

    gamma2 = damping_term(params, eigenvalues)
    omega2 = stiffness_term(params, eigenvalues)
    factors = string_tau_with_density(params) * eigenvalues / (length/2.0)
    nonlinear = make_tm_nl_fn(eigenvalues, factors)
    _, positions_by_time, _ = solve_tf(gamma2, omega2, 1.0/rate,
                                        n_steps=samples, nl_fn=nonlinear,
                                        u0=u0, v0=jnp.zeros_like(u0))
    # Modal readout at the pickup fraction (the midpoint when pickup = 1/2).
    weights = jnp.sin(jnp.pi * mode_indices * pickup) / (length/2.0)
    return positions_by_time @ weights


def renderer(values, steps, intervals=176, rate=132300, excitation='multimode'):
    """Return (batch, steps+2) internal-rate audio from the upstream solver.

    ``steps+2`` matches our two initial states followed by ``steps`` updates.
    ``values`` has shape (batch, 10), ordered density, tension, radius, Young's
    modulus, length, amplitude, sigma0, sigma1, pluck position, pickup position
    (the last two as fractions of L); all are differentiated.
    ``excitation`` names the same fixed profile the FDTD solver accepts.
    """
    if values.ndim != 2 or values.shape[1] != 10:
        raise ValueError('Expected values with shape (batch, 10)')
    if steps < 0 or intervals < 3 or rate <= 0:
        raise ValueError('Invalid simulation grid or step count')
    return jax.vmap(lambda row: _single(row, steps+2, intervals, rate, excitation))(values)
