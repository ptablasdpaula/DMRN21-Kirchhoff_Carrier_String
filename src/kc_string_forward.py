"""Compiled forward sensitivities for the damped, non-SAV KC string.

Each row contains a value followed by its ten parameter derivatives (eight
material parameters, then the pluck and pickup positions as fractions of L).
Two (intervals - 1, 11) states replace the reverse-mode computation graph.
The mean-square loss and its derivatives accumulate online. Returned gradients
are with respect to relative multipliers of PARAMETER_NAMES, not physical units.
The reference startup convention is preserved, including its extra h factors.
"""
import math
import os
from functools import lru_cache
from pathlib import Path
import torch
if __package__:
    from . import kc_string as reference
else:
    import kc_string as reference

PARAMETER_NAMES = (
    'material_density', 'tension', 'radius', 'young_modulus',
    'length', 'amplitude', 'sigma_0', 'sigma_1',
    'pluck_position', 'pickup_position',
)
PARAMETER_COUNT = len(PARAMETER_NAMES)
DUAL_COLUMNS = PARAMETER_COUNT + 1
PLUCK, PICKUP = PARAMETER_NAMES.index('pluck_position'), PARAMETER_NAMES.index('pickup_position')


def nominal_parameters(device=None):
    return torch.tensor([getattr(reference, name) for name in PARAMETER_NAMES], dtype=torch.float64, device=device)


def dual_constant(value, device=None):
    return torch.cat((torch.as_tensor(value, dtype=torch.float64, device=device).reshape(1),
                      torch.zeros(PARAMETER_COUNT, dtype=torch.float64, device=device)))


def dual_multiply(left, right):
    # Product rule: (a*b)' = a'*b + a*b'. Last dimension holds the columns.
    value = left[..., :1] * right[..., :1]
    derivatives = left[..., 1:] * right[..., :1] + left[..., :1] * right[..., 1:]
    return torch.cat((value, derivatives), dim=-1)


def dual_divide(numerator, denominator):
    # Quotient rule, using the already computed value to avoid squaring b.
    value = numerator[..., :1] / denominator[..., :1]
    derivatives = (numerator[..., 1:] - value * denominator[..., 1:]) / denominator[..., :1]
    return torch.cat((value, derivatives), dim=-1)


def dual_sqrt(argument):
    value = torch.sqrt(argument[..., :1])
    return torch.cat((value, argument[..., 1:] / (2 * value)), dim=-1)


SPATIAL_ORDER = 4
STENCILS = {2: (1., -2., 1.), 4: (-1/12, 4/3, -5/2, 4/3, -1/12)}


def eigenvalue_factor(order):
    """Largest |eigenvalue| of h²·D_xx relative to the three-point value 4: 1 (order 2), 4/3 (order 4)."""
    if order not in STENCILS:
        raise ValueError('Spatial order must be 2 or 4.')
    weights = STENCILS[order]
    half = len(weights) // 2
    return -sum(weights[half + m] * math.cos(math.pi * m) for m in range(-half, half + 1)) / 4


def discrete_eigenvalues(intervals, order=SPATIAL_ORDER, device=None):
    """Eigenvalues of h²·D_xx on the N-1 discrete sine modes (dimensionless, negative).

    Single source for the sine-basis kernels, the modal diagnostics and the tests:
    -4s² for the three-point stencil, -4s²(1+s²/3) for the five-point one,
    s = sin(πn/2N). Both operators are diagonal in the same sine basis.
    """
    s2 = torch.sin(torch.pi * torch.arange(1, intervals, dtype=torch.float64, device=device) / (2 * intervals)).square()
    if order == 2:
        return -4 * s2
    if order == 4:
        return -4 * s2 * (1 + s2 / 3)
    raise ValueError('Spatial order must be 2 or 4.')


def second_derivative(values, inverse_spacing_squared, order=SPATIAL_ORDER):
    """Second difference on the interior nodes with simply supported ends.

    Order 2 is the reference code's three-point stencil. Order 4 is the five-point
    stencil (-1, 16, -30, 16, -1)/12 closed by odd reflection about the fixed ends
    (u_{-1} = u_N = 0, u_{-2} = -u_1, u_{N+1} = -u_{N-1}); the fourth difference is
    this operator applied twice. The reflection keeps the operator diagonal in the
    discrete sine basis.
    """
    if order == 2:
        differences = torch.cat((
            (-2 * values[0] + values[1]).unsqueeze(0),
            values[:-2] - 2 * values[1:-1] + values[2:],
            (values[-2] - 2 * values[-1]).unsqueeze(0),
        ))
    elif order == 4:
        zero = torch.zeros_like(values[:1])
        padded = torch.cat((-values[:1], zero, values, zero, -values[-1:]))
        count = values.shape[0]
        weights = STENCILS[4]
        differences = weights[0] * padded[:count]
        for m in range(1, len(weights)):
            differences = differences + weights[m] * padded[m:m + count]
    else:
        raise ValueError('Spatial order must be 2 or 4.')
    return dual_multiply(differences, inverse_spacing_squared)


def dual_dot(left, right):
    return dual_multiply(left, right).sum(dim=0)


EXCITATIONS = ('first_mode', 'multimode', 'raised_cosine')


def parse_excitation(excitation):
    """Split an excitation string into (kind, raised-cosine half-width fraction or None).

    The plain kinds are ``first_mode``, ``multimode`` and ``raised_cosine``.
    ``raised_cosine:<fraction>`` fixes the half-width as a fraction of the
    string length instead of the reference's node count, so that the same
    physical pluck can be placed on grids with different interval counts.
    """
    kind, _, argument = str(excitation).partition(':')
    if kind not in EXCITATIONS:
        raise ValueError('Choose first_mode, multimode or raised_cosine[:half_width_fraction] excitation.')
    if not argument:
        return kind, None
    if kind != 'raised_cosine':
        raise ValueError('Only raised_cosine accepts a half-width fraction.')
    fraction = float(argument)
    if not math.isfinite(fraction) or not 0 < fraction <= .5:
        raise ValueError('The raised-cosine half-width fraction must lie in (0, 0.5].')
    return kind, fraction


def raised_cosine_fraction(intervals):
    """The reference's node-based half-width, floor(.25 (N-1)) nodes, as a fraction of L."""
    return max(1, math.floor(.25*(intervals-1)))/intervals


def excitation_shape(intervals, excitation='first_mode', device=None, center=None, derivative=False):
    """Spatial profile of the pluck; amplitude remains a separate differentiable parameter.

    ``center`` is the raised cosine's centre as a fraction of L (a float or a
    0-d tensor; the reference's L/2 when omitted). With ``derivative=True`` the
    profile's derivative with respect to that fraction is returned instead
    (zero for the position-free ``first_mode`` and ``multimode`` kinds).
    """
    kind, fraction = parse_excitation(excitation)
    indices = torch.arange(1, intervals, dtype=torch.float64, device=device)
    if kind == 'first_mode':
        return torch.zeros_like(indices) if derivative else torch.sin(math.pi * indices / intervals)
    if kind == 'multimode':
        if derivative:
            return torch.zeros_like(indices)
        modes = torch.arange(1, min(8, intervals-1)+1, dtype=torch.float64, device=device)
        shape = (torch.sin(math.pi*indices[:, None]*modes/intervals)/modes**2).sum(dim=1)
        return shape / shape.abs().max()
    # Same node-centred width convention as the MATLAB reference unless a
    # physical half-width fraction was given. In node units the centre is
    # center*N, which is the reference's N/2 exactly at the nominal position.
    width = max(1, math.floor(.25*(intervals-1))) if fraction is None else fraction*intervals
    node = intervals/2 if center is None else torch.as_tensor(center, dtype=torch.float64, device=device)*intervals
    distance = (indices-node)/width
    inside = distance.abs() <= 1
    if derivative:
        return torch.where(inside, .5*math.pi*(intervals/width)*torch.sin(math.pi*distance), 0.)
    return torch.where(inside, .5*(1+torch.cos(math.pi*distance)), 0.)


def sine_transform(intervals, device=None):
    """Orthonormal discrete sine transform of the N-1 interior nodes (its own inverse)."""
    indices = torch.arange(1, intervals, dtype=torch.float64, device=device)
    return (2/intervals)**.5*torch.sin(torch.pi*indices[:, None]*indices[None, :]/intervals)


def pickup_weights(relative_parameters, intervals, spatial, device=None):
    """Dual (N-1, 11) readout weights of the sine interpolant at the pickup position.

    Value column: w_i = sqrt(2/N) sin(pi i x_o) in modal coordinates. Whenever
    x_o*N is an integer the value column is copied from that row of the sine
    transform (modal) or is the exact one-hot vector on that node (spatial), so
    the nominal midpoint reproduces the former node readout bit for bit. The
    only nonzero tangent is the pickup column, dw/dtheta = nominal * pi i cos(.)
    scaled alike; ``spatial=True`` applies the transform to it.
    """
    device = relative_parameters.device if device is None else device
    nominal = nominal_parameters(device)
    position = nominal[PICKUP]*relative_parameters[PICKUP]
    indices = torch.arange(1, intervals, dtype=torch.float64, device=device)
    node = position*intervals
    angle = torch.pi*indices*node/intervals
    scale = (2/intervals)**.5
    value = scale*torch.sin(angle)
    tangent = scale*torch.cos(angle)*torch.pi*indices*nominal[PICKUP]
    transform = sine_transform(intervals, device)
    rounded = torch.round(node)
    on_node = (node-rounded).abs() < 1e-12*intervals
    one_hot = (indices == rounded).to(torch.float64)
    if spatial:
        value = torch.where(on_node, one_hot, transform@value)
        tangent = transform@tangent
    else:
        value = torch.where(on_node, one_hot@transform, value)
    # Built without in-place writes so that torch.vmap can batch it.
    silent = torch.zeros(intervals-1, PICKUP, dtype=torch.float64, device=device)
    return torch.cat((value[:, None], silent, tangent[:, None]), dim=1)


def initialize(relative_parameters, intervals, sample_rate, excitation='first_mode', order=SPATIAL_ORDER):
    device = relative_parameters.device
    nominal = nominal_parameters(device)
    # d(nominal_i * theta_i)/dtheta_j = nominal_i when i == j, otherwise zero.
    parameters = torch.cat(((nominal * relative_parameters)[:, None], torch.diag(nominal)), dim=1)
    density, tension, radius, modulus, length, amplitude, sigma_0, sigma_1, pluck, _ = parameters.unbind()
    time_step = 1 / sample_rate
    spacing = length / intervals
    area = math.pi * dual_multiply(radius, radius)
    mass = dual_multiply(density, area)
    inertia = math.pi * dual_multiply(dual_multiply(radius, radius), dual_multiply(radius, radius)) / 4
    inverse_spacing_squared = dual_divide(dual_constant(1, device), dual_multiply(spacing, spacing))
    # The pluck profile depends on its centre: value plus the chain rule through
    # the pluck column (its only nonzero tangent), then the product with amplitude.
    shape = excitation_shape(intervals, excitation, device, center=pluck[0])
    shape_derivative = excitation_shape(intervals, excitation, device, center=pluck[0], derivative=True)
    previous = dual_multiply(torch.cat((shape[:, None], shape_derivative[:, None]*pluck[None, 1:]), dim=1), amplitude)
    second = second_derivative(previous, inverse_spacing_squared, order)
    fourth = second_derivative(second, inverse_spacing_squared, order)
    nonlinear = dual_multiply(-dual_dot(previous, second), second)
    c_squared = dual_divide(tension, mass)
    K_squared = dual_divide(dual_multiply(modulus, inertia), mass)
    nonlinear_scale = dual_divide(0.5 * dual_multiply(modulus, spacing), dual_multiply(length, density))
    current = previous + 0.5 * time_step**2 * (
        dual_multiply(dual_multiply(spacing, c_squared), second)
        - dual_multiply(dual_multiply(spacing, K_squared), fourth)
        + dual_multiply(dual_multiply(spacing, nonlinear_scale), nonlinear)
    )
    inverse_mass = dual_divide(dual_constant(1, device), mass + dual_multiply(mass, sigma_0) * time_step)
    q_scale = 0.5 * time_step * dual_sqrt(dual_divide(dual_multiply(dual_multiply(spacing, modulus), area), length))
    coefficients = torch.stack((inverse_spacing_squared, mass,
                               tension * time_step**2 + 2 * dual_multiply(mass, sigma_1) * time_step,
                               dual_multiply(modulus, inertia) * time_step**2,
                               dual_multiply(mass, sigma_0) * time_step - mass,
                               2 * dual_multiply(mass, sigma_1) * time_step, inverse_mass, q_scale))
    weights = pickup_weights(relative_parameters, intervals, True, device)
    first, second_sample = dual_dot(weights, previous), dual_dot(weights, current)
    initial_loss = (dual_multiply(first, first) + dual_multiply(second_sample, second_sample)) / reference.amplitude**2
    return torch.stack((previous, current)), coefficients, initial_loss


def advance(states, coefficients, weights, loss_sum, order=SPATIAL_ORDER):
    inverse_spacing_squared, mass, tension_loss, bending, previous_mass, previous_loss, inverse_mass, q_scale = coefficients.unbind()
    previous, current = states.unbind()
    second = second_derivative(current, inverse_spacing_squared, order)
    fourth = second_derivative(second, inverse_spacing_squared, order)
    previous_second = second_derivative(previous, inverse_spacing_squared, order)
    linear = (dual_multiply(2 * mass, current) + dual_multiply(tension_loss, second)
              - dual_multiply(bending, fourth) + dual_multiply(previous_mass, previous)
              - dual_multiply(previous_loss, previous_second))
    direction = dual_multiply(q_scale, second)
    rhs = linear - dual_multiply(direction, dual_dot(direction, previous))
    mass_scaled_rhs = dual_multiply(inverse_mass, rhs)
    mass_scaled_direction = dual_multiply(inverse_mass, direction)
    denominator = dual_constant(1) + dual_dot(direction, mass_scaled_direction)
    projection = dual_dot(direction, mass_scaled_rhs)
    following = mass_scaled_rhs - dual_multiply(mass_scaled_direction, dual_divide(projection, denominator))
    sample = dual_dot(weights, following)
    loss_sum = loss_sum + dual_multiply(sample, sample) / reference.amplitude**2
    return torch.stack((current, following)), loss_sum, sample[0]


def advance_block(states, coefficients, weights, loss_sum, steps, order=SPATIAL_ORDER):
    samples = []
    for _ in range(steps):
        states, loss_sum, sample = advance(states, coefficients, weights, loss_sum, order)
        samples.append(sample)
    return states, loss_sum, torch.stack(samples)


class DualReference:
    """Validation-only evaluator of pickup mean-square loss and ten gradients.

    Keep this instance across optimisation iterations to reuse compilation.
    Spatial interval count, sample rate and chunk size are fixed for the run.
    No reverse-mode or checkpointed execution is offered by this production API.
    """
    def __init__(self, intervals=128, sample_rate=44100, block_steps=128, order=SPATIAL_ORDER):
        if intervals < 4 or intervals % 2 or sample_rate <= 0 or block_steps < 1:
            raise ValueError('Use an even interval count >= 4, a positive sample rate and chunk size.')
        if order not in STENCILS:
            raise ValueError('Spatial order must be 2 or 4.')
        self.intervals = intervals
        self.sample_rate = sample_rate
        self.block_steps = block_steps
        self.order = order
        self.operation = torch.compile(advance_block, backend="inductor", fullgraph=True, dynamic=False)

    @torch.no_grad()
    def loss_and_gradient(self, relative_parameters, duration_seconds):
        """Return (audio, mean_square_loss, gradient_wrt_relative_parameters).

        The loss uses a fixed reference amplitude normalizer. It is the benchmark
        objective, not a generic audio loss. Parameter updates must respect the
        fixed-grid stability region; this function does not clip parameters.
        """
        relative_parameters = torch.as_tensor(relative_parameters, dtype=torch.float64, device='cpu').detach()
        if relative_parameters.shape != (PARAMETER_COUNT,) or not torch.isfinite(relative_parameters).all():
            raise ValueError(f'Expected {PARAMETER_COUNT} finite relative parameter multipliers.')
        if not math.isfinite(duration_seconds) or duration_seconds < 0:
            raise ValueError('Duration must be finite and non-negative.')
        steps = round(duration_seconds * self.sample_rate)
        states, coefficients, loss_sum = initialize(relative_parameters, self.intervals, self.sample_rate, order=self.order)
        weights = pickup_weights(relative_parameters, self.intervals, True)
        output = torch.empty(steps + 2, dtype=torch.float64)
        output[0] = dual_dot(weights, states[0])[0]
        output[1] = dual_dot(weights, states[1])[0]
        for index in range(0, steps, self.block_steps):
            count = min(self.block_steps, steps - index)
            states, loss_sum, samples = self.operation(states, coefficients, weights, loss_sum, count, self.order)
            output[index + 2:index + 2 + count] = samples
        return output, loss_sum[0] / (steps + 2), loss_sum[1:] / (steps + 2)


@lru_cache(maxsize=1)
def _native_kernel():
    """Compile once and reuse PyTorch's extension cache across processes."""
    from torch.utils.cpp_extension import load
    import platform
    native_flag = '-mcpu=native' if platform.machine().lower() in ('arm64', 'aarch64') else '-march=native'
    flags = ['/O2', '/fp:strict'] if os.name == 'nt' else ['-O3', native_flag, '-ffp-contract=off']
    return load(name='kc_string_dual_cpu',
                sources=[str(Path(__file__).with_name('kc_string_dual.cpp'))],
                extra_cflags=flags, with_cuda=False, verbose=False)


def native_block(states, coefficients, weights, loss_sum, steps, order=SPATIAL_ORDER):
    return _native_kernel().advance_block(states, coefficients, weights, loss_sum, steps, reference.amplitude**2, order)


def _accelerated():
    if __package__:
        from .accelerated import evaluate
    else:
        from accelerated import evaluate
    return evaluate


class _AudioWithTangents(torch.autograd.Function):
    @staticmethod
    def forward(ctx, relative_parameters, intervals, sample_rate, steps, output_stride, excitation, backend, chunk_steps, threads, order):
        if backend == 'native' and relative_parameters.device.type == 'cpu':
            states, coefficients, _ = initialize(relative_parameters, intervals, sample_rate, excitation, order)
            weights = pickup_weights(relative_parameters, intervals, True)
            audio, tangents = _native_kernel().simulate(states, coefficients, weights, steps, output_stride, order)
        else:
            audio, tangents = _accelerated()(relative_parameters, intervals, sample_rate, steps, output_stride,
                                       excitation, backend, True, chunk_steps, threads, order=order)
        ctx.save_for_backward(tangents)
        return audio

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        (tangents,) = ctx.saved_tensors
        # Chain rule for any scalar loss computed from the returned audio:
        # d loss/d theta_j = sum_i (d loss/d audio_i) (d audio_i/d theta_j).
        return grad_output @ tangents, None, None, None, None, None, None, None, None, None


class ForwardString:
    """First-order differentiable CPU/CUDA string with an ordinary PyTorch audio output.

    theta has ten float64 relative multipliers, ordered by PARAMETER_NAMES.
    The spatial/time grid stays fixed throughout optimisation. The first call
    builds the selected C++/CUDA extension; later calls reuse its cached binary.
    Use any differentiable PyTorch loss and optimizer. Second derivatives are
    deliberately unsupported. No reverse simulation or checkpointing is used.
    """
    def __init__(self, intervals=128, sample_rate=44100, output_stride=1, excitation="first_mode",
                 backend="native", chunk_steps=8192, threads=256, order=SPATIAL_ORDER):
        if backend not in ("native", "spatial", "modal"):
            raise ValueError("Choose native, spatial or modal backend.")
        if not isinstance(chunk_steps, int) or chunk_steps < 0 or threads not in (128, 256):
            raise ValueError("Use nonnegative chunk_steps (zero means full render), threads 128 or 256.")
        if order not in STENCILS:
            raise ValueError("Choose spatial order 2 (three-point) or 4 (five-point).")
        self.backend, self.chunk_steps, self.threads, self.order = backend, chunk_steps, threads, order
        if not isinstance(intervals, int) or intervals < 4 or intervals % 2:
            raise ValueError('Use an even integer interval count >= 4.')
        if not math.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError('Sample rate must be finite and positive.')
        if not isinstance(output_stride, int) or output_stride < 1:
            raise ValueError('Output stride must be a positive integer.')
        self.intervals = intervals
        self.sample_rate = sample_rate
        excitation_shape(intervals, excitation)
        self.excitation = excitation
        self.output_stride = output_stride

    def _steps(self, theta, duration_seconds):
        if not isinstance(theta, torch.Tensor) or theta.device.type not in ('cpu','cuda') or theta.dtype != torch.float64:
            raise ValueError('theta must be a CPU or CUDA float64 tensor.')
        if theta.shape != (PARAMETER_COUNT,) or not torch.isfinite(theta).all():
            raise ValueError(f'theta must contain {PARAMETER_COUNT} finite relative parameter multipliers.')
        if not math.isfinite(duration_seconds) or duration_seconds < 0:
            raise ValueError('Duration must be finite and non-negative.')
        return round(duration_seconds * self.sample_rate)

    def __call__(self, theta, duration_seconds=0.06):
        """Return differentiable pickup audio, including the two startup samples.

        output_stride selects indices 0, stride, 2*stride, ... of that signal.
        This is observation subsampling, not an anti-aliasing filter. For audio
        downsampling, keep stride=1 and apply a differentiable filter in PyTorch.
        The caller must keep theta inside the fixed-grid stability region.
        """
        steps = self._steps(theta, duration_seconds)
        if not torch.is_grad_enabled() or not theta.requires_grad:
            if self.backend != 'native' or theta.is_cuda:
                return _accelerated()(theta, self.intervals, self.sample_rate, steps, self.output_stride,
                                self.excitation, self.backend, False, self.chunk_steps, self.threads, order=self.order)[0]
        return _AudioWithTangents.apply(theta, self.intervals, self.sample_rate, steps, self.output_stride,
                                       self.excitation, self.backend, self.chunk_steps, self.threads, self.order)

    @torch.no_grad()
    def output_and_tangents(self, theta, duration_seconds=0.06):
        """Return detached audio and its (n_out, 10) first-order Jacobian."""
        steps = self._steps(theta, duration_seconds)
        if self.backend != 'native' or theta.is_cuda:
            return _accelerated()(theta, self.intervals, self.sample_rate, steps, self.output_stride,
                            self.excitation, self.backend, True, self.chunk_steps, self.threads, order=self.order)
        states, coefficients, _ = initialize(theta.detach(), self.intervals, self.sample_rate, self.excitation, self.order)
        weights = pickup_weights(theta.detach(), self.intervals, True)
        return _native_kernel().simulate(states, coefficients, weights, steps, self.output_stride, self.order)
