"""The damped, non-SAV Kirchhoff–Carrier scheme, carried with forward-mode tangents.

The update is Russo et al.'s ``KCStringSAV_noSplit.m`` with ``SAV = false``, with
an optional five-point stencil (order 4, the paper's) besides the reference's
three-point one (order 2). Every quantity is a *dual* array whose last axis holds
a value followed by its derivatives with respect to the ten relative parameter
multipliers ``theta`` (physical values divided by ``physics.NOMINAL``).

``initialize`` builds the start-up states and the per-string coefficients. Their
tangents are what the hand-written adjoint (``adjoint.py``) needs to turn
adjoints of the coefficients into parameter gradients. ``reference`` steps the
scheme in Python and returns audio with its exact Jacobian, the slow reference
that every test compares against.
"""

import math

import torch

from . import physics

COLUMNS = physics.PARAMETER_COUNT + 1  # a value and ten derivatives
STENCILS = {2: (1.0, -2.0, 1.0), 4: (-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12)}
ORDER = 4


# Dual arithmetic: the last axis is (value, d/dtheta_1, ..., d/dtheta_10).
def dual_constant(value, device=None):
    """A constant: its value and zero derivatives."""
    zeros = torch.zeros(physics.PARAMETER_COUNT, dtype=torch.float64, device=device)
    return torch.cat((torch.as_tensor(value, dtype=torch.float64, device=device).reshape(1), zeros))


def dual_multiply(left, right):
    """Product rule: (ab)' = a'b + ab'."""
    value = left[..., :1] * right[..., :1]
    derivatives = left[..., 1:] * right[..., :1] + left[..., :1] * right[..., 1:]
    return torch.cat((value, derivatives), dim=-1)


def dual_divide(numerator, denominator):
    """Quotient rule, reusing the quotient's value."""
    value = numerator[..., :1] / denominator[..., :1]
    derivatives = (numerator[..., 1:] - value * denominator[..., 1:]) / denominator[..., :1]
    return torch.cat((value, derivatives), dim=-1)


def dual_sqrt(argument):
    """(sqrt a)' = a' / (2 sqrt a)."""
    value = torch.sqrt(argument[..., :1])
    return torch.cat((value, argument[..., 1:] / (2 * value)), dim=-1)


def dual_dot(left, right):
    """Sum over the leading (node) axis of a dual product."""
    return dual_multiply(left, right).sum(dim=0)


def _check_order(order):
    if order not in STENCILS:
        raise ValueError("The spatial order must be 2 (three-point) or 4 (five-point).")


def eigenvalue_factor(order):
    """Largest |eigenvalue| of h^2 D_xx relative to the three-point value 4: 1 (order 2), 4/3 (order 4)."""
    _check_order(order)
    weights = STENCILS[order]
    half = len(weights) // 2
    return -sum(weights[half + m] * math.cos(math.pi * m) for m in range(-half, half + 1)) / 4


def discrete_eigenvalues(intervals, order=ORDER, device=None):
    """Eigenvalues of h^2 D_xx on the N-1 discrete sine modes, with s = sin(pi n / 2N).

    -4 s^2 for the three-point stencil and -4 s^2 (1 + s^2 / 3) for the five-point one.
    """
    _check_order(order)
    n = torch.arange(1, intervals, dtype=torch.float64, device=device)
    s2 = torch.sin(torch.pi * n / (2 * intervals)).square()
    return -4 * s2 if order == 2 else -4 * s2 * (1 + s2 / 3)


def second_derivative(values, inverse_spacing_squared, order=ORDER):
    """D_xx on the interior nodes of a simply supported string (dual values).

    The five-point stencil is closed by odd reflection about the fixed ends
    (u_0 = u_N = 0, u_-1 = -u_1, u_N+1 = -u_N-1), which keeps it diagonal in
    the discrete sine basis. Applying it twice gives the fourth difference.
    """
    _check_order(order)
    if order == 2:
        differences = torch.cat(
            (
                (-2 * values[0] + values[1]).unsqueeze(0),
                values[:-2] - 2 * values[1:-1] + values[2:],
                (values[-2] - 2 * values[-1]).unsqueeze(0),
            )
        )
    else:
        zero = torch.zeros_like(values[:1])
        padded = torch.cat((-values[:1], zero, values, zero, -values[-1:]))
        count = values.shape[0]
        weights = STENCILS[4]
        differences = weights[0] * padded[:count]
        for m in range(1, len(weights)):
            differences = differences + weights[m] * padded[m : m + count]
    return dual_multiply(differences, inverse_spacing_squared)


def parse_excitation(excitation):
    """Split ``first_mode``, ``raised_cosine`` or ``raised_cosine:<half-width as a fraction of L>``.

    Plain ``raised_cosine`` uses the reference's node-based half-width,
    floor((N-1) / 4) nodes; a fraction keeps the same physical pluck on any grid.
    """
    kind, _, argument = str(excitation).partition(":")
    if kind not in ("first_mode", "raised_cosine") or (argument and kind != "raised_cosine"):
        raise ValueError("Choose first_mode, raised_cosine or raised_cosine:<half-width fraction>.")
    if not argument:
        return kind, None
    fraction = float(argument)
    if not 0 < fraction <= 0.5:
        raise ValueError("The raised-cosine half-width fraction must lie in (0, 0.5].")
    return kind, fraction


def raised_cosine_half_width(intervals, fraction=None):
    """Half-width of the raised cosine in nodes."""
    return max(1, math.floor(0.25 * (intervals - 1))) if fraction is None else fraction * intervals


def excitation_shape(intervals, excitation, device=None, center=None, derivative=False):
    """Unit pluck profile on the interior nodes, or its derivative with respect to its centre.

    ``center`` is the raised cosine's centre as a fraction of L (L/2 if omitted).
    The amplitude is applied separately, as a differentiable parameter.
    """
    kind, fraction = parse_excitation(excitation)
    indices = torch.arange(1, intervals, dtype=torch.float64, device=device)
    if kind == "first_mode":
        return torch.zeros_like(indices) if derivative else torch.sin(math.pi * indices / intervals)
    width = raised_cosine_half_width(intervals, fraction)
    node = intervals / 2 if center is None else torch.as_tensor(center, dtype=torch.float64, device=device) * intervals
    distance = (indices - node) / width
    inside = distance.abs() <= 1
    if derivative:
        return torch.where(inside, 0.5 * math.pi * (intervals / width) * torch.sin(math.pi * distance), 0.0)
    return torch.where(inside, 0.5 * (1 + torch.cos(math.pi * distance)), 0.0)


def sine_transform(intervals, device=None):
    """Orthonormal discrete sine transform of the N-1 interior nodes (its own inverse)."""
    n = torch.arange(1, intervals, dtype=torch.float64, device=device)
    return (2 / intervals) ** 0.5 * torch.sin(torch.pi * n[:, None] * n[None, :] / intervals)


def pickup_weights(theta, intervals, spatial, device=None):
    """Dual (N-1, 11) readout weights of the sine interpolant at the pickup position.

    In modal coordinates w_n = sqrt(2/N) sin(pi n x); ``spatial=True`` maps them to
    the nodes. When x N is an integer the weights are exactly that node's one-hot
    vector (or its row of the transform), so a node pickup is reproduced bit for
    bit. Only the pickup column has a nonzero tangent.
    """
    device = theta.device if device is None else device
    nominal_pickup = physics.NOMINAL[physics.PICKUP]
    position = nominal_pickup * theta[physics.PICKUP]
    n = torch.arange(1, intervals, dtype=torch.float64, device=device)
    node = position * intervals
    angle = torch.pi * n * node / intervals
    scale = (2 / intervals) ** 0.5
    value = scale * torch.sin(angle)
    tangent = scale * torch.cos(angle) * torch.pi * n * nominal_pickup
    transform = sine_transform(intervals, device)
    rounded = torch.round(node)
    on_node = (node - rounded).abs() < 1e-12 * intervals
    one_hot = (n == rounded).to(torch.float64)
    if spatial:
        value = torch.where(on_node, one_hot, transform @ value)
        tangent = transform @ tangent
    else:
        value = torch.where(on_node, one_hot @ transform, value)
    # No in-place writes, so that torch.vmap can batch it.
    silent = torch.zeros(intervals - 1, physics.PICKUP, dtype=torch.float64, device=device)
    return torch.cat((value[:, None], silent, tangent[:, None]), dim=1)


def initialize(theta, intervals, sample_rate, excitation, order=ORDER):
    """Start-up states (2, N-1, 11) and the per-string coefficients (8, 11), as duals.

    The coefficients are, in order: 1/h^2, rho A, T k^2 + 2 rho A sigma_1 k,
    E I k^2, rho A sigma_0 k - rho A, 2 rho A sigma_1 k, 1 / (rho A (1 + sigma_0 k))
    and the scale of the nonlinear direction, k/2 sqrt(h E A / L).
    The start-up step keeps the reference's convention, including its factors of h.
    """
    device = theta.device
    nominal = physics.nominal_parameters(device)
    # d(nominal_i theta_i)/d theta_j = nominal_i if i == j else 0.
    parameters = torch.cat(((nominal * theta)[:, None], torch.diag(nominal)), dim=1)
    density, tension, radius, modulus, length, amplitude, sigma_0, sigma_1, pluck, _ = parameters.unbind()
    time_step = 1 / sample_rate
    spacing = length / intervals
    area = math.pi * dual_multiply(radius, radius)
    mass = dual_multiply(density, area)
    inertia = math.pi * dual_multiply(dual_multiply(radius, radius), dual_multiply(radius, radius)) / 4
    inverse_spacing_squared = dual_divide(dual_constant(1, device), dual_multiply(spacing, spacing))
    # The pluck profile depends on its centre through the pluck column only.
    shape = excitation_shape(intervals, excitation, device, center=pluck[0])
    slope = excitation_shape(intervals, excitation, device, center=pluck[0], derivative=True)
    profile = torch.cat((shape[:, None], slope[:, None] * pluck[None, 1:]), dim=1)
    previous = dual_multiply(profile, amplitude)
    second = second_derivative(previous, inverse_spacing_squared, order)
    fourth = second_derivative(second, inverse_spacing_squared, order)
    nonlinear = dual_multiply(-dual_dot(previous, second), second)
    c_squared = dual_divide(tension, mass)
    k_squared = dual_divide(dual_multiply(modulus, inertia), mass)
    nonlinear_scale = dual_divide(0.5 * dual_multiply(modulus, spacing), dual_multiply(length, density))
    current = previous + 0.5 * time_step**2 * (
        dual_multiply(dual_multiply(spacing, c_squared), second)
        - dual_multiply(dual_multiply(spacing, k_squared), fourth)
        + dual_multiply(dual_multiply(spacing, nonlinear_scale), nonlinear)
    )
    inverse_mass = dual_divide(dual_constant(1, device), mass + dual_multiply(mass, sigma_0) * time_step)
    q_scale = 0.5 * time_step * dual_sqrt(dual_divide(dual_multiply(dual_multiply(spacing, modulus), area), length))
    coefficients = torch.stack(
        (
            inverse_spacing_squared,
            mass,
            tension * time_step**2 + 2 * dual_multiply(mass, sigma_1) * time_step,
            dual_multiply(modulus, inertia) * time_step**2,
            dual_multiply(mass, sigma_0) * time_step - mass,
            2 * dual_multiply(mass, sigma_1) * time_step,
            inverse_mass,
            q_scale,
        )
    )
    return torch.stack((previous, current)), coefficients


def advance(states, coefficients, order=ORDER):
    """One time step: the linear update, then the rank-one (Sherman–Morrison) solve."""
    inverse_h2, mass, tension_loss, bending, previous_mass, previous_loss, inverse_mass, q_scale = coefficients.unbind()
    previous, current = states.unbind()
    second = second_derivative(current, inverse_h2, order)
    fourth = second_derivative(second, inverse_h2, order)
    previous_second = second_derivative(previous, inverse_h2, order)
    linear = (
        dual_multiply(2 * mass, current)
        + dual_multiply(tension_loss, second)
        - dual_multiply(bending, fourth)
        + dual_multiply(previous_mass, previous)
        - dual_multiply(previous_loss, previous_second)
    )
    direction = dual_multiply(q_scale, second)
    rhs = linear - dual_multiply(direction, dual_dot(direction, previous))
    mass_scaled_rhs = dual_multiply(inverse_mass, rhs)
    mass_scaled_direction = dual_multiply(inverse_mass, direction)
    denominator = dual_constant(1, states.device) + dual_dot(direction, mass_scaled_direction)
    projection = dual_dot(direction, mass_scaled_rhs)
    following = mass_scaled_rhs - dual_multiply(mass_scaled_direction, dual_divide(projection, denominator))
    return torch.stack((current, following))


@torch.no_grad()
def reference(theta, intervals, sample_rate, steps, excitation, order=ORDER):
    """Pickup audio (steps + 2,) and its exact Jacobian (steps + 2, 10), stepped in Python on the grid.

    Slow; for tests and for checking every faster implementation.
    """
    theta = torch.as_tensor(theta, dtype=torch.float64)
    states, coefficients = initialize(theta, intervals, sample_rate, excitation, order)
    weights = pickup_weights(theta, intervals, spatial=True)
    samples = [dual_dot(weights, states[0]), dual_dot(weights, states[1])]
    for _ in range(steps):
        states = advance(states, coefficients, order)
        samples.append(dual_dot(weights, states[1]))
    samples = torch.stack(samples)
    return samples[:, 0], samples[:, 1:]
