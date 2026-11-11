"""JAX twin of the spatial PyTorch baseline: the same FDTD scheme under ``jax.jit``.

``experiments.figure_comparison.fdtd`` steps the Kirchhoff--Carrier scheme on
the spatial grid in eager PyTorch, where every one of its small tensor
operations is dispatched separately at every time step. This module is that
arithmetic, operation for operation (the same stencil, start-up step, pluck
profile, pickup interpolant and rank-one solve), written for JAX so that the
whole time loop and its reverse pass are compiled by XLA. It is a baseline
for what compiled automatic differentiation costs; nothing here is derived by
hand.

The time loop is a ``jax.lax.scan`` that reads the pickup inside its body.
With ``checkpoint_steps`` set, it becomes an outer scan over segments of that
many steps whose body, an inner scan, is wrapped in ``jax.checkpoint``:
reverse mode keeps the two states at each segment boundary and recomputes one
segment at a time, the same rematerialisation as ``torch.utils.checkpoint``
in the PyTorch baseline and as the 256-step checkpoints of the CUDA adjoint.
With ``basis='sine'`` the identical scheme is stepped in the discrete sine
basis instead of on the grid: both difference operators are diagonal there,
so the stencils become one multiplication per mode and the pickup is read
with modal weights, which is the formulation the CUDA adjoint uses. The
states are the grid states under an orthonormal change of coordinates, so
waveform and gradients are unchanged; only the arithmetic per step is
cheaper. Inputs are the ten physical coordinates (not multipliers), as for the other
JAX renderers; the output is the internal-rate pickup waveform including the
two initial samples.
"""
import math

import jax
import jax.numpy as jnp

from src.kc_string_forward import SPATIAL_ORDER, STENCILS, excitation_shape, parse_excitation

jax.config.update('jax_enable_x64', True)


def _second(state, inverse_h2, order):
    """Second difference of one string's (N-1,) interior nodes, simply supported ends.

    Order 2 is the three-point stencil with zero end points; order 4 is the
    five-point stencil closed by odd reflection about the ends.
    """
    zero = jnp.zeros_like(state[:1])
    if order == 2:
        padded = jnp.concatenate((zero, state, zero))
        return (padded[:-2] - 2*padded[1:-1] + padded[2:]) * inverse_h2
    padded = jnp.concatenate((-state[:1], zero, state, zero, -state[-1:]))
    count = state.shape[0]
    weights = STENCILS[4]
    total = weights[0] * padded[:count]
    for m in range(1, len(weights)):
        total = total + weights[m] * padded[m:m + count]
    return total * inverse_h2


def _pluck_profile(intervals, excitation, pluck, dtype):
    """(N-1,) initial profile; a raised cosine is centred at the pluck fraction."""
    kind, fraction = parse_excitation(excitation)
    if kind != 'raised_cosine':
        return jnp.asarray(excitation_shape(intervals, excitation).numpy(), dtype=dtype)
    indices = jnp.arange(1, intervals, dtype=dtype)
    width = max(1, math.floor(.25*(intervals-1))) if fraction is None else fraction*intervals
    distance = (indices - pluck*intervals) / width
    return jnp.where(jnp.abs(distance) <= 1, .5*(1+jnp.cos(jnp.pi*distance)), 0.)


def _pickup_weights(intervals, pickup, dtype):
    """(N-1,) nodal weights of the sine interpolant at the pickup fraction."""
    indices = jnp.arange(1, intervals, dtype=dtype)
    modal = (2/intervals)**.5 * jnp.sin(jnp.pi*indices*pickup)
    transform = (2/intervals)**.5 * jnp.sin(jnp.pi*indices[:, None]*indices[None, :]/intervals)
    return modal @ transform


def _single(values, intervals, rate, steps, excitation, order, checkpoint_steps, unroll, basis):
    density, tension, radius, modulus, length, amplitude, sigma0, sigma1, pluck, pickup = values
    dtype = values.dtype
    dt = 1.0 / rate
    h = length / intervals
    area = jnp.pi * radius**2
    mass = density * area
    inertia = jnp.pi * radius**4 / 4
    inv_h2 = 1.0 / h**2

    previous = _pluck_profile(intervals, excitation, pluck, dtype) * amplitude
    if basis == 'sine':
        # Orthonormal discrete sine transform: the difference operators become
        # the eigenvalue table of src.kc_string_forward.discrete_eigenvalues.
        indices = jnp.arange(1, intervals, dtype=dtype)
        s2 = jnp.sin(jnp.pi * indices / (2 * intervals))**2
        eigenvalues = (-4 * s2 if order == 2 else -4 * s2 * (1 + s2 / 3)) * inv_h2
        transform = (2/intervals)**.5 * jnp.sin(jnp.pi*indices[:, None]*indices[None, :]/intervals)
        previous = transform @ previous
        second = lambda state: eigenvalues * state
    else:
        second = lambda state: _second(state, inv_h2, order)
    d2 = second(previous)
    d4 = second(d2)
    nonlinear = -jnp.sum(previous * d2) * d2
    c2 = tension / mass
    k2 = modulus * inertia / mass
    nonlinear_scale = 0.5 * modulus * h / (length * density)
    current = previous + 0.5 * dt**2 * (h * c2 * d2 - h * k2 * d4 + h * nonlinear_scale * nonlinear)
    inverse_mass = 1.0 / (mass + mass * sigma0 * dt)
    tension_loss = tension * dt**2 + 2 * mass * sigma1 * dt
    bending = modulus * inertia * dt**2
    previous_mass = mass * sigma0 * dt - mass
    previous_loss = 2 * mass * sigma1 * dt
    q_scale = 0.5 * dt * jnp.sqrt(h * modulus * area / length)
    if basis == 'sine':
        readout = (2/intervals)**.5 * jnp.sin(jnp.pi * indices * pickup)
    else:
        readout = _pickup_weights(intervals, pickup, dtype)

    def step(state, _):
        before, now = state
        d2 = second(now)
        d4 = second(d2)
        p2 = second(before)
        linear = 2 * mass * now + tension_loss * d2 - bending * d4 + previous_mass * before - previous_loss * p2
        direction = q_scale * d2
        rhs = linear - direction * jnp.sum(direction * before)
        mass_rhs = inverse_mass * rhs
        mass_direction = inverse_mass * direction
        denominator = 1 + jnp.sum(direction * mass_direction)
        projection = jnp.sum(direction * mass_rhs)
        following = mass_rhs - mass_direction * (projection / denominator)
        return (now, following), jnp.sum(following * readout)

    def run(state, length_):
        return jax.lax.scan(step, state, None, length=length_, unroll=unroll)

    state = (previous, current)
    pieces = [jnp.stack((jnp.sum(previous * readout), jnp.sum(current * readout)))]
    if checkpoint_steps is None:
        if steps:
            state, samples = run(state, steps)
            pieces.append(samples)
    else:
        full, remainder = divmod(steps, checkpoint_steps)
        if full:
            body = jax.checkpoint(lambda carry, _: run(carry, checkpoint_steps))
            state, samples = jax.lax.scan(body, state, None, length=full)
            pieces.append(samples.reshape(full * checkpoint_steps))
        if remainder:
            state, samples = jax.checkpoint(lambda carry: run(carry, remainder))(state)
            pieces.append(samples)
    return jnp.concatenate(pieces)


def renderer(values, steps, intervals=176, rate=132300, excitation='multimode', order=SPATIAL_ORDER,
             checkpoint_steps=None, unroll=8, basis='grid'):
    """(batch, steps+2) internal-rate pickup audio of the spatial FDTD scheme.

    Args:
        values: float64 physical coordinates of shape (batch, 10): density,
            tension, radius, Young's modulus, length, amplitude, sigma0, sigma1,
            pluck and pickup position (fractions of L); all are differentiated.
        steps: recurrence steps after the two initial samples.
        intervals: even spatial interval count, so there are intervals-1 nodes.
        rate: internal solver sample rate in Hz.
        excitation: one of the profiles accepted by ``excitation_shape``.
        order: spatial stencil order, 2 (three-point) or 4 (five-point).
        checkpoint_steps: None keeps every step for the reverse pass; a
            positive integer rematerialises the loop in segments of that many
            steps with ``jax.checkpoint``.
        unroll: time steps unrolled into one iteration of the compiled loop.
        basis: 'grid' steps the nodes with the stencils; 'sine' steps the same
            scheme in the discrete sine basis, where the stencils are diagonal.
    """
    if values.ndim != 2 or values.shape[1] != 10:
        raise ValueError('Expected physical parameters with shape (batch, 10)')
    if order not in STENCILS:
        raise ValueError('order must be 2 or 4')
    if not isinstance(intervals, int) or intervals < 4 or intervals % 2:
        raise ValueError('intervals must be an even integer >= 4')
    if not math.isfinite(rate) or rate <= 0 or not isinstance(steps, int) or steps < 0:
        raise ValueError('rate must be positive and steps a nonnegative integer')
    if checkpoint_steps is not None and (not isinstance(checkpoint_steps, int) or isinstance(checkpoint_steps, bool)
                                         or checkpoint_steps < 1):
        raise ValueError('checkpoint_steps must be a positive integer or None')
    if not isinstance(unroll, int) or isinstance(unroll, bool) or unroll < 1:
        raise ValueError('unroll must be a positive integer')
    if basis not in ('grid', 'sine'):
        raise ValueError("basis must be 'grid' or 'sine'")
    return jax.vmap(lambda row: _single(row, intervals, rate, steps, excitation, order, checkpoint_steps, unroll,
                                        basis))(values)
