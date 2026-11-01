"""Plain spatial Kirchhoff--Carrier FDTD with PyTorch reverse-mode autograd.

This is deliberately a direct, uncheckpointed baseline: each time step is
expressed in PyTorch and its complete computation graph is retained for the
backward pass. The interface takes relative material multipliers, as do the
existing CUDA kernels. The output is the internal-rate pickup waveform,
including the two initial samples; filtering and loss belong to the caller.
The pluck centre and the pickup position (parameters 9 and 10, fractions of
L) stay inside the graph: the raised cosine is built from the centre and the
readout is the sine interpolant at the pickup, so both receive gradients.
"""
import math
import torch

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


def simulate(theta, intervals=176, rate=132300, steps=13230, excitation='multimode', order=SPATIAL_ORDER):
    """Simulate a batch of independent strings using ordinary PyTorch AD.

    Args:
        theta: FP64 relative parameter multipliers of shape (10,) or (B, 10).
        intervals: even spatial interval count, so there are intervals-1 nodes.
        rate: internal solver sample rate in Hz.
        steps: recurrence steps after the two initial samples.
        excitation: one of the profiles accepted by ``excitation_shape``.
        order: spatial stencil order, 2 (three-point) or 4 (five-point).

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
    samples = [(previous * readout).sum(dim=1), (current * readout).sum(dim=1)]
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
    audio = torch.stack(samples, dim=1)
    return audio[0] if single else audio
