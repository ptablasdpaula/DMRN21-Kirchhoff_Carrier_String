"""The same scheme in JAX, differentiated automatically: the paper's baselines.

``render`` is ``scheme.reference`` arithmetic, operation for operation, as a
``jax.lax.scan`` that ``jax.jit`` compiles together with its reverse pass:
    Auto (compiled)   basis="grid", checkpoint=None: every state is stored
    + checkpoints     basis="grid", checkpoint=256: ``jax.checkpoint`` per segment
    + modes           basis="sine", checkpoint=256: stepped in the discrete sine
                      basis, where both stencils are diagonal (the kernel's formulation)
The modal states are the grid states under an orthonormal change of
coordinates, so audio and gradients agree; only the cost per step differs.
"""

import jax
import jax.numpy as jnp

from . import physics
from .scheme import ORDER, STENCILS, discrete_eigenvalues, parse_excitation, raised_cosine_half_width, sine_transform

jax.config.update("jax_enable_x64", True)


def _second(state, inverse_h2, order):
    """D_xx on the interior nodes with simply supported ends (as ``scheme.second_derivative``)."""
    zero = jnp.zeros_like(state[:1])
    if order == 2:
        padded = jnp.concatenate((zero, state, zero))
        return (padded[:-2] - 2 * padded[1:-1] + padded[2:]) * inverse_h2
    padded = jnp.concatenate((-state[:1], zero, state, zero, -state[-1:]))
    count = state.shape[0]
    weights = STENCILS[4]
    total = weights[0] * padded[:count]
    for m in range(1, len(weights)):
        total = total + weights[m] * padded[m : m + count]
    return total * inverse_h2


def _pluck_profile(intervals, excitation, pluck):
    """Unit pluck profile, a raised cosine centred at the pluck fraction (as ``scheme.excitation_shape``)."""
    kind, fraction = parse_excitation(excitation)
    n = jnp.arange(1, intervals, dtype=jnp.float64)
    if kind == "first_mode":
        return jnp.sin(jnp.pi * n / intervals)
    distance = (n - pluck * intervals) / raised_cosine_half_width(intervals, fraction)
    return jnp.where(jnp.abs(distance) <= 1, 0.5 * (1 + jnp.cos(jnp.pi * distance)), 0.0)


def _single(values, intervals, rate, steps, excitation, order, checkpoint, unroll, basis):
    """Pickup audio (steps + 2,) of one string from its ten physical values."""
    density, tension, radius, modulus, length, amplitude, sigma_0, sigma_1, pluck, pickup = values
    dt = 1.0 / rate
    h = length / intervals
    area = jnp.pi * radius**2
    mass = density * area
    inertia = jnp.pi * radius**4 / 4
    inv_h2 = 1.0 / h**2
    n = jnp.arange(1, intervals, dtype=jnp.float64)
    modal_readout = (2 / intervals) ** 0.5 * jnp.sin(jnp.pi * n * pickup)
    previous = _pluck_profile(intervals, excitation, pluck) * amplitude
    if basis == "sine":
        transform = jnp.asarray(sine_transform(intervals).numpy())
        eigenvalues = jnp.asarray(discrete_eigenvalues(intervals, order).numpy()) * inv_h2
        previous = transform @ previous
        readout = modal_readout

        def second(state):
            return eigenvalues * state
    else:
        readout = modal_readout @ jnp.asarray(sine_transform(intervals).numpy())

        def second(state):
            return _second(state, inv_h2, order)

    d2 = second(previous)
    d4 = second(d2)
    nonlinear = -jnp.sum(previous * d2) * d2
    c2 = tension / mass
    k2 = modulus * inertia / mass
    nonlinear_scale = 0.5 * modulus * h / (length * density)
    current = previous + 0.5 * dt**2 * (h * c2 * d2 - h * k2 * d4 + h * nonlinear_scale * nonlinear)
    inverse_mass = 1.0 / (mass + mass * sigma_0 * dt)
    tension_loss = tension * dt**2 + 2 * mass * sigma_1 * dt
    bending = modulus * inertia * dt**2
    previous_mass = mass * sigma_0 * dt - mass
    previous_loss = 2 * mass * sigma_1 * dt
    q_scale = 0.5 * dt * jnp.sqrt(h * modulus * area / length)

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

    def run(state, count):
        return jax.lax.scan(step, state, None, length=count, unroll=unroll)

    state = (previous, current)
    pieces = [jnp.stack((jnp.sum(previous * readout), jnp.sum(current * readout)))]
    if checkpoint is None:
        if steps:
            state, samples = run(state, steps)
            pieces.append(samples)
    else:
        full, remainder = divmod(steps, checkpoint)
        if full:
            body = jax.checkpoint(lambda carry, _: run(carry, checkpoint))
            state, samples = jax.lax.scan(body, state, None, length=full)
            pieces.append(samples.reshape(full * checkpoint))
        if remainder:
            state, samples = jax.checkpoint(lambda carry: run(carry, remainder))(state)
            pieces.append(samples)
    return jnp.concatenate(pieces)


def render(values, steps, intervals, rate, excitation, order=ORDER, checkpoint=None, unroll=8, basis="grid"):
    """Pickup audio ``(batch, steps + 2)`` at the internal rate for ``(batch, 10)`` physical SI values.

    ``checkpoint`` is None to store every state for the reverse pass, or the
    segment length to recompute under ``jax.checkpoint``. ``unroll`` steps are
    unrolled into each iteration of the compiled loop.
    """
    if values.ndim != 2 or values.shape[1] != physics.PARAMETER_COUNT:
        raise ValueError(f"Expected physical values of shape (batch, {physics.PARAMETER_COUNT}).")
    if basis not in ("grid", "sine"):
        raise ValueError("basis must be 'grid' or 'sine'.")
    if intervals < 4 or intervals % 2 or steps < 0 or (checkpoint is not None and checkpoint < 1):
        raise ValueError("Invalid grid, step count or checkpoint interval.")
    return jax.vmap(lambda row: _single(row, intervals, rate, steps, excitation, order, checkpoint, unroll, basis))(
        values
    )
