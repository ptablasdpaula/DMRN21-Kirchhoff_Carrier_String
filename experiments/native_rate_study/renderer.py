"""Pinned jaxdiffmodal integration with one fixed physical initial shape.

The upstream integrator is unchanged. The displacement profile named by
``excitation`` is sampled on the fixed reference grid (176 intervals by
default) and projected onto the first ``modes`` continuous string modes; the
modal count never changes how the initial displacement is generated or
normalized. Modes whose continuous frequency lies above half the rate are
still integrated exactly by the upstream recurrence but alias; the caller
chooses ``modes`` accordingly.

``checkpoint_steps`` selects ``solve_tf_checkpointed``: the same recurrence,
coefficients and first RK4 step as upstream ``solve_tf``, but run as a nested
``jax.lax.scan`` whose ``checkpoint_steps``-step segments are wrapped in
``jax.checkpoint``. Reverse mode then keeps the carry at each segment boundary
and recomputes one segment at a time instead of storing every step. The
values are the same; only the memory/time trade-off of the backward pass
changes. Returning the full modal trajectory and its velocities as upstream
does still costs two (steps, modes) arrays per item plus their cotangents;
``readout_in_scan`` instead reads the pickup inside the scan and returns the
pickup waveform only, as the FDTD kernels do.
"""
import jax
import jax.numpy as jnp
from jaxdiffmodal.ftm import StringParameters, damping_term, stiffness_term, string_eigenvalues
from jaxdiffmodal.time_integrators import make_tm_nl_fn, rk4_step, solve_tf, string_tau_with_density

from experiments.figure_comparison.diaz import initial_shape

jax.config.update('jax_enable_x64', True)


def solve_tf_checkpointed(gamma2_mu, omega_mu_squared, dt, n_steps, nl_fn=None, u0=None, v0=None, segment=256,
                          weights=None):
    """Unforced upstream ``solve_tf`` with its recurrence rematerialised in ``segment``-step pieces.

    Mirrors ``jaxdiffmodal.time_integrators.solve_tf`` (coefficients a1, a2, b1,
    the RK4 first step, the two-term update and the output layout) and returns
    the same ``(state, positions, velocities)``; the only difference is that the
    time loop is a scan over segments whose body, an inner scan of ``segment``
    steps, is wrapped in ``jax.checkpoint``. A shorter final segment covers the
    remainder. With ``weights`` (one value per mode) the scan emits the pickup
    sample ``q @ weights`` of every step instead of the modal vectors and the
    function returns ``(state, samples)`` with ``samples`` of length ``n_steps``,
    so no (steps, modes) array is ever materialised.
    """
    if not isinstance(n_steps, int) or n_steps < 2: raise ValueError('n_steps must be an integer >= 2')
    if not isinstance(segment, int) or isinstance(segment, bool) or segment < 1:
        raise ValueError('segment must be a positive integer')
    n_modes = gamma2_mu.shape[0]
    u0 = u0 if u0 is not None else jnp.zeros(n_modes)
    v0 = v0 if v0 is not None else jnp.zeros(n_modes)
    apply_nl = nl_fn if nl_fn is not None else (lambda q: jnp.zeros_like(q))
    gamma_mu = gamma2_mu / 2.0
    omega_mu_damped = jnp.sqrt(omega_mu_squared - gamma_mu**2)
    radius = jnp.exp(-gamma_mu * dt)
    imag = radius * jnp.sin(omega_mu_damped * dt)
    real = radius * jnp.cos(omega_mu_damped * dt)
    b1 = dt * imag / omega_mu_damped
    a1 = 2.0 * real
    a2 = -(radius**2)
    q0 = u0
    q1, v1 = rk4_step(u0, v0, dt, gamma2_mu, omega_mu_squared, nl_fn=nl_fn, excitation=None)

    def advance_state(state, _):
        q_prev, q_curr, v = state
        nl = apply_nl(q_curr)
        q_next = a1 * q_curr + a2 * q_prev - b1 * nl
        v_next = (q_next - q_prev) / (2.0 * dt)
        if weights is not None:
            return (q_curr, q_next, v_next), q_next @ weights
        return (q_curr, q_next, v_next), (q_next, v_next)

    def run(carry, length):
        return jax.lax.scan(advance_state, carry, None, length=length, unroll=8)

    full, remainder = divmod(n_steps - 1, segment)
    state = (q0, q1, v1)
    if weights is not None:
        pieces = []
        if full:
            body = jax.checkpoint(lambda carry, _: run(carry, segment))
            state, y_seg = jax.lax.scan(body, state, None, length=full)
            pieces.append(y_seg.reshape(full * segment))
        if remainder:
            state, y_rem = jax.checkpoint(lambda carry, _: run(carry, remainder))(state, None)
            pieces.append(y_rem)
        samples = jnp.concatenate([jnp.stack((q0 @ weights, q1 @ weights))] + [p[:-1] for p in pieces[:-1]] + [pieces[-1][:-1]], axis=0)
        return state, samples[:n_steps]
    positions, velocities = [], []
    if full:
        body = jax.checkpoint(lambda carry, _: run(carry, segment))
        state, (q_seg, v_seg) = jax.lax.scan(body, state, None, length=full)
        positions.append(q_seg.reshape(full * segment, n_modes))
        velocities.append(v_seg.reshape(full * segment, n_modes))
    if remainder:
        state, (q_rem, v_rem) = jax.checkpoint(lambda carry, _: run(carry, remainder))(state, None)
        positions.append(q_rem); velocities.append(v_rem)
    final_positions = jnp.concatenate(positions, axis=0)
    final_velocities = jnp.concatenate(velocities, axis=0)
    full_positions = jnp.concatenate([q0[None], q1[None], final_positions[:-1]], axis=0)
    full_velocities = jnp.concatenate([v0[None], final_velocities], axis=0)
    return state, full_positions[:n_steps], full_velocities[:n_steps]


def _single(values, samples, rate, modes, reference_intervals, excitation, checkpoint_steps=None, readout_in_scan=False):
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
    weights = jnp.sin(jnp.pi*indices*pickup)/(length/2.)  # modal readout at the pickup fraction
    if checkpoint_steps and readout_in_scan:
        _, output = solve_tf_checkpointed(damping_term(params,eig), stiffness_term(params,eig),
                                          1./rate, n_steps=samples, nl_fn=nl,
                                          u0=u0, v0=jnp.zeros_like(u0), segment=checkpoint_steps, weights=weights)
        return output
    if checkpoint_steps:
        _, trajectory, _ = solve_tf_checkpointed(damping_term(params,eig), stiffness_term(params,eig),
                                                 1./rate, n_steps=samples, nl_fn=nl,
                                                 u0=u0, v0=jnp.zeros_like(u0), segment=checkpoint_steps)
    else:
        _, trajectory, _ = solve_tf(damping_term(params,eig), stiffness_term(params,eig),
                                    1./rate, n_steps=samples, nl_fn=nl,
                                    u0=u0, v0=jnp.zeros_like(u0))
    return trajectory @ weights


def renderer(values, samples, rate=44100, modes=175, reference_intervals=176, excitation='multimode',
             checkpoint_steps=None, readout_in_scan=False):
    if values.ndim != 2 or values.shape[1] != 10: raise ValueError('Expected physical parameters (batch, 10)')
    if not isinstance(reference_intervals, int) or reference_intervals < 4 or reference_intervals % 2:
        raise ValueError('Use an even reference interval count >= 4')
    if not isinstance(modes, int) or not 1 <= modes <= reference_intervals-1 or samples <= 0 or rate <= 0:
        raise ValueError('Use 1 <= modes <= reference_intervals-1, positive samples and rate')
    if checkpoint_steps is not None and (not isinstance(checkpoint_steps, int) or isinstance(checkpoint_steps, bool)
                                         or checkpoint_steps < 1):
        raise ValueError('checkpoint_steps must be a positive integer or None')
    if readout_in_scan and not checkpoint_steps: raise ValueError('readout_in_scan requires checkpoint_steps')
    return jax.vmap(lambda row:_single(row,samples,rate,modes,reference_intervals,excitation,checkpoint_steps,readout_in_scan))(values)
