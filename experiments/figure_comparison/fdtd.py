"""Plain spatial Kirchhoff--Carrier FDTD with PyTorch reverse-mode autograd.

This is deliberately a direct baseline: each time step is expressed in
PyTorch and, by default, its complete computation graph is retained for the
backward pass. With ``checkpoint_steps`` set, the loop is cut into chunks of
that many steps and each chunk goes through ``torch.utils.checkpoint`` (the
non-reentrant form), so only the two states at the chunk boundaries are kept
and the chunk's graph is rebuilt during the backward pass: framework-level
rematerialisation with the same 256-step segments as the cached adjoint, and
nothing else changed. The interface takes relative material multipliers, as do the
existing CUDA kernels. The output is the internal-rate pickup waveform,
including the two initial samples; filtering and loss belong to the caller.
The pluck centre and the pickup position (parameters 9 and 10, fractions of
L) stay inside the graph: the raised cosine is built from the centre and the
readout is the sine interpolant at the pickup, so both receive gradients.

With ``compile_block`` set, the same step arithmetic is run through
``torch.compile`` (TorchDynamo, AOTAutograd and TorchInductor) in blocks of
that many steps: the Python loop over a block is unrolled into one graph, its
forward and backward are compiled once and then called block after block, so
the per-operation dispatch of the eager loop is replaced by fused kernels.
Compilation combines with ``checkpoint_steps`` (each chunk is a whole number
of compiled blocks plus a compiled remainder) and changes no arithmetic.
"""
import functools
import math
import torch
from torch.utils.checkpoint import checkpoint

from src.kc_string_forward import (SPATIAL_ORDER, STENCILS, PARAMETER_COUNT, excitation_shape, nominal_parameters,
                                   parse_excitation, sine_transform)


def _second(state, inverse_h2, order=SPATIAL_ORDER):
    """Second difference along dim 1 with simply supported ends.

    Order 2 is the three-point stencil with zero end points; order 4 is the
    five-point stencil closed by odd reflection about the ends, as in
    ``src.kc_string_forward.second_derivative``.
    """
    if order == 2:
        return torch.cat((
            -2 * state[:, :1] + state[:, 1:2],
            state[:, :-2] - 2 * state[:, 1:-1] + state[:, 2:],
            state[:, -2:-1] - 2 * state[:, -1:],
        ), dim=1) * inverse_h2
    zero = torch.zeros_like(state[:, :1])
    padded = torch.cat((-state[:, :1], zero, state, zero, -state[:, -1:]), dim=1)
    count = state.shape[1]
    weights = STENCILS[4]
    total = weights[0] * padded[:, :count]
    for m in range(1, len(weights)):
        total = total + weights[m] * padded[:, m:m + count]
    return total * inverse_h2


def _pluck_profile(intervals, excitation, pluck):
    """(B, N-1) initial profile; a raised cosine is centred at each item's pluck fraction."""
    kind, fraction = parse_excitation(excitation)
    if kind != 'raised_cosine':
        return excitation_shape(intervals, excitation, pluck.device).unsqueeze(0)
    indices = torch.arange(1, intervals, dtype=torch.float64, device=pluck.device)
    width = max(1, math.floor(.25*(intervals-1))) if fraction is None else fraction*intervals
    distance = (indices.unsqueeze(0) - pluck*intervals) / width
    return torch.where(distance.abs() <= 1, .5*(1+torch.cos(math.pi*distance)), 0.)


def _pickup_weights(intervals, pickup):
    """(B, N-1) nodal weights of the sine interpolant at each item's pickup fraction.

    Kept smooth everywhere (no one-hot shortcut on a node) so that autograd
    differentiates the readout with respect to the position.
    """
    indices = torch.arange(1, intervals, dtype=torch.float64, device=pickup.device)
    modal = (2/intervals)**.5 * torch.sin(torch.pi*indices.unsqueeze(0)*pickup)
    return modal @ sine_transform(intervals, pickup.device)


def _advance(previous, current, readout, inv_h2, mass, tension_loss, bending, previous_mass, previous_loss,
             q_scale, inverse_mass, *, steps, order):
    """Run ``steps`` updates from (previous, current); returns the two final states and the (B, steps) pickup samples.

    Every differentiable tensor is an explicit argument so that the chunk can be
    rematerialised by ``torch.utils.checkpoint`` without closing over the graph.
    """
    samples = []
    for _ in range(steps):
        d2 = _second(current, inv_h2, order)
        d4 = _second(d2, inv_h2, order)
        p2 = _second(previous, inv_h2, order)
        linear = (2 * mass * current + tension_loss * d2 - bending * d4
                  + previous_mass * previous - previous_loss * p2)
        direction = q_scale * d2
        rhs = linear - direction * (direction * previous).sum(dim=1, keepdim=True)
        mass_rhs = inverse_mass * rhs
        mass_direction = inverse_mass * direction
        denominator = 1 + (direction * mass_direction).sum(dim=1, keepdim=True)
        projection = (direction * mass_rhs).sum(dim=1, keepdim=True)
        following = mass_rhs - mass_direction * (projection / denominator)
        samples.append((following * readout).sum(dim=1))
        previous, current = current, following
    return previous, current, torch.stack(samples, dim=1)


@functools.lru_cache(maxsize=None)
def _compiled_block(steps, order, mode):
    """``_advance`` over a fixed number of steps as one ``torch.compile`` graph (static shapes)."""
    def block(previous, current, readout, inv_h2, mass, tension_loss, bending, previous_mass, previous_loss,
              q_scale, inverse_mass):
        return _advance(previous, current, readout, inv_h2, mass, tension_loss, bending, previous_mass,
                        previous_loss, q_scale, inverse_mass, steps=steps, order=order)
    return torch.compile(block, dynamic=False, mode=mode)


def _advance_compiled(previous, current, *coefficients, steps, order, block, mode):
    """``_advance`` as a sequence of compiled blocks of ``block`` steps and one compiled remainder.

    Under ``mode='reduce-overhead'`` the compiled graphs replay as CUDA graphs
    whose outputs live in a reused static pool, so each block's outputs are
    cloned before the next call.
    """
    samples = []
    for size in (block,) * (steps // block) + ((steps % block,) if steps % block else ()):
        previous, current, piece = _compiled_block(size, order, mode)(previous, current, *coefficients)
        if mode == 'reduce-overhead':
            previous, current, piece = previous.clone(), current.clone(), piece.clone()
        samples.append(piece)
    return previous, current, torch.cat(samples, dim=1)


def simulate(theta, intervals=176, rate=132300, steps=13230, excitation='multimode', order=SPATIAL_ORDER,
             checkpoint_steps=None, compile_block=None, compile_mode=None, checkpoint_reentrant=False):
    """Simulate a batch of independent strings using ordinary PyTorch AD.

    Args:
        theta: FP64 relative parameter multipliers of shape (10,) or (B, 10).
        intervals: even spatial interval count, so there are intervals-1 nodes.
        rate: internal solver sample rate in Hz.
        steps: recurrence steps after the two initial samples.
        excitation: one of the profiles accepted by ``excitation_shape``.
        order: spatial stencil order, 2 (three-point) or 4 (five-point).
        checkpoint_steps: None keeps the whole graph (the plain baseline); a
            positive integer rematerialises the loop in chunks of that many
            steps with ``torch.utils.checkpoint`` (use_reentrant=False).
        compile_block: None runs the eager loop; a positive integer compiles
            the loop with ``torch.compile`` in blocks of that many steps.
        compile_mode: ``torch.compile`` mode for the blocks (None is the
            default mode; ``'reduce-overhead'`` adds CUDA graphs).
        checkpoint_reentrant: use the reentrant form of
            ``torch.utils.checkpoint`` (forward under no_grad, a nested
            backward per chunk) instead of the non-reentrant one.

    Returns:
        Pickup displacement of shape (steps+2,) or (B, steps+2).
    """
    if not isinstance(theta, torch.Tensor) or theta.dtype != torch.float64 or theta.ndim not in (1, 2):
        raise ValueError(f'theta must be a float64 tensor of shape ({PARAMETER_COUNT},) or (B,{PARAMETER_COUNT})')
    if order not in STENCILS:
        raise ValueError('order must be 2 or 4')
    single = theta.ndim == 1
    x = theta.unsqueeze(0) if single else theta
    if x.shape[0] < 1 or x.shape[1] != PARAMETER_COUNT:
        raise ValueError(f'theta must be a nonempty batch of {PARAMETER_COUNT} multipliers')
    if not isinstance(intervals, int) or intervals < 4 or intervals % 2:
        raise ValueError('intervals must be an even integer >= 4')
    if not math.isfinite(rate) or rate <= 0 or not isinstance(steps, int) or steps < 0:
        raise ValueError('rate must be positive and steps a nonnegative integer')
    if checkpoint_steps is not None and (not isinstance(checkpoint_steps, int) or isinstance(checkpoint_steps, bool)
                                         or checkpoint_steps < 1):
        raise ValueError('checkpoint_steps must be a positive integer or None')
    if compile_block is not None and (not isinstance(compile_block, int) or isinstance(compile_block, bool)
                                      or compile_block < 1):
        raise ValueError('compile_block must be a positive integer or None')
    if compile_mode is not None and compile_block is None:
        raise ValueError('compile_mode requires compile_block')

    # Values in each column are independent across batch items. Each material
    # parameter keeps a singleton spatial dimension to broadcast over nodes.
    physical = x * nominal_parameters(x.device)
    density, tension, radius, modulus, length, amplitude, sigma0, sigma1, pluck, pickup = (
        value.unsqueeze(1) for value in physical.unbind(dim=1)
    )
    dt = 1.0 / rate
    h = length / intervals
    area = math.pi * radius.square()
    mass = density * area
    inertia = math.pi * radius.pow(4) / 4
    inv_h2 = h.reciprocal().square()

    previous = _pluck_profile(intervals, excitation, pluck) * amplitude
    d2 = _second(previous, inv_h2, order)
    d4 = _second(d2, inv_h2, order)
    nonlinear = -(previous * d2).sum(dim=1, keepdim=True) * d2
    c2 = tension / mass
    k2 = modulus * inertia / mass
    nonlinear_scale = 0.5 * modulus * h / (length * density)
    current = previous + 0.5 * dt**2 * (
        h * c2 * d2 - h * k2 * d4 + h * nonlinear_scale * nonlinear
    )
    inverse_mass = (mass + mass * sigma0 * dt).reciprocal()
    tension_loss = tension * dt**2 + 2 * mass * sigma1 * dt
    bending = modulus * inertia * dt**2
    previous_mass = mass * sigma0 * dt - mass
    previous_loss = 2 * mass * sigma1 * dt
    q_scale = 0.5 * dt * torch.sqrt(h * modulus * area / length)

    readout = _pickup_weights(intervals, pickup)
    blocks = [torch.stack(((previous * readout).sum(dim=1), (current * readout).sum(dim=1)), dim=1)]
    coefficients = (readout, inv_h2, mass, tension_loss, bending, previous_mass, previous_loss, q_scale, inverse_mass)
    chunk = max(1, steps if checkpoint_steps is None else checkpoint_steps)
    for start in range(0, steps, chunk):
        count = min(chunk, steps - start)
        if compile_block is None:
            operation = functools.partial(_advance, steps=count, order=order)
        else:
            operation = functools.partial(_advance_compiled, steps=count, order=order, block=compile_block,
                                          mode=compile_mode)
        if checkpoint_steps is None:
            previous, current, samples = operation(previous, current, *coefficients)
        else:
            previous, current, samples = checkpoint(operation, previous, current, *coefficients,
                                                    use_reentrant=checkpoint_reentrant, preserve_rng_state=False)
        blocks.append(samples)
    audio = torch.cat(blocks, dim=1)
    return audio[0] if single else audio
