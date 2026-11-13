"""Compiled forward sensitivities for the damped, non-SAV KC string.

Each row contains a value followed by its eight parameter derivatives.
Two (intervals - 1, 9) states replace the reverse-mode computation graph.
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
)


def nominal_parameters():
    return torch.tensor([getattr(reference, name) for name in PARAMETER_NAMES], dtype=torch.float64)


def dual_constant(value):
    return torch.cat((torch.as_tensor(value, dtype=torch.float64).reshape(1), torch.zeros(8, dtype=torch.float64)))


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


def second_derivative(values, inverse_spacing_squared):
    differences = torch.cat((
        (-2 * values[0] + values[1]).unsqueeze(0),
        values[:-2] - 2 * values[1:-1] + values[2:],
        (values[-2] - 2 * values[-1]).unsqueeze(0),
    ))
    return dual_multiply(differences, inverse_spacing_squared)


def dual_dot(left, right):
    return dual_multiply(left, right).sum(dim=0)


def excitation_shape(intervals, excitation='first_mode'):
    """Fixed spatial profile; amplitude remains a differentiable physical parameter."""
    indices = torch.arange(1, intervals, dtype=torch.float64)
    if excitation == 'first_mode':
        return torch.sin(math.pi * indices / intervals)
    if excitation == 'multimode':
        modes = torch.arange(1, min(8, intervals-1)+1, dtype=torch.float64)
        shape = (torch.sin(math.pi*indices[:, None]*modes/intervals)/modes**2).sum(dim=1)
        return shape / shape.abs().max()
    if excitation == 'raised_cosine':
        # Same node-centred width convention as the MATLAB reference.
        width = max(1, math.floor(.25*(intervals-1)))
        distance = (indices-intervals/2)/width
        return torch.where(distance.abs() <= 1, .5*(1+torch.cos(math.pi*distance)), 0.)
    raise ValueError('Choose first_mode, multimode or raised_cosine excitation.')


def initialize(relative_parameters, intervals, sample_rate, excitation='first_mode'):
    nominal = nominal_parameters()
    # d(nominal_i * theta_i)/dtheta_j = nominal_i when i == j, otherwise zero.
    parameters = torch.cat(((nominal * relative_parameters)[:, None], torch.diag(nominal)), dim=1)
    density, tension, radius, modulus, length, amplitude, sigma_0, sigma_1 = parameters.unbind()
    time_step = 1 / sample_rate
    spacing = length / intervals
    area = math.pi * dual_multiply(radius, radius)
    mass = dual_multiply(density, area)
    inertia = math.pi * dual_multiply(dual_multiply(radius, radius), dual_multiply(radius, radius)) / 4
    inverse_spacing_squared = dual_divide(dual_constant(1), dual_multiply(spacing, spacing))
    previous = excitation_shape(intervals, excitation)[:, None] * amplitude
    second = second_derivative(previous, inverse_spacing_squared)
    fourth = second_derivative(second, inverse_spacing_squared)
    nonlinear = dual_multiply(-dual_dot(previous, second), second)
    c_squared = dual_divide(tension, mass)
    K_squared = dual_divide(dual_multiply(modulus, inertia), mass)
    nonlinear_scale = dual_divide(0.5 * dual_multiply(modulus, spacing), dual_multiply(length, density))
    current = previous + 0.5 * time_step**2 * (
        dual_multiply(dual_multiply(spacing, c_squared), second)
        - dual_multiply(dual_multiply(spacing, K_squared), fourth)
        + dual_multiply(dual_multiply(spacing, nonlinear_scale), nonlinear)
    )
    inverse_mass = dual_divide(dual_constant(1), mass + dual_multiply(mass, sigma_0) * time_step)
    q_scale = 0.5 * time_step * dual_sqrt(dual_divide(dual_multiply(dual_multiply(spacing, modulus), area), length))
    coefficients = torch.stack((inverse_spacing_squared, mass,
                               tension * time_step**2 + 2 * dual_multiply(mass, sigma_1) * time_step,
                               dual_multiply(modulus, inertia) * time_step**2,
                               dual_multiply(mass, sigma_0) * time_step - mass,
                               2 * dual_multiply(mass, sigma_1) * time_step, inverse_mass, q_scale))
    midpoint = (intervals - 1) // 2
    initial_loss = (dual_multiply(previous[midpoint], previous[midpoint])
                    + dual_multiply(current[midpoint], current[midpoint])) / reference.amplitude**2
    return torch.stack((previous, current)), coefficients, initial_loss


def advance(states, coefficients, loss_sum):
    inverse_spacing_squared, mass, tension_loss, bending, previous_mass, previous_loss, inverse_mass, q_scale = coefficients.unbind()
    previous, current = states.unbind()
    second = second_derivative(current, inverse_spacing_squared)
    fourth = second_derivative(second, inverse_spacing_squared)
    previous_second = second_derivative(previous, inverse_spacing_squared)
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
    sample = following[following.shape[0] // 2]
    loss_sum = loss_sum + dual_multiply(sample, sample) / reference.amplitude**2
    return torch.stack((current, following)), loss_sum, sample[0]


def advance_block(states, coefficients, loss_sum, steps):
    samples = []
    for _ in range(steps):
        states, loss_sum, sample = advance(states, coefficients, loss_sum)
        samples.append(sample)
    return states, loss_sum, torch.stack(samples)


class DualReference:
    """Validation-only evaluator of midpoint mean-square loss and eight gradients.

    Keep this instance across optimisation iterations to reuse compilation.
    Spatial interval count, sample rate and chunk size are fixed for the run.
    No reverse-mode or checkpointed execution is offered by this production API.
    """
    def __init__(self, intervals=128, sample_rate=44100, block_steps=128):
        if intervals < 4 or intervals % 2 or sample_rate <= 0 or block_steps < 1:
            raise ValueError('Use an even interval count >= 4, a positive sample rate and chunk size.')
        self.intervals = intervals
        self.sample_rate = sample_rate
        self.block_steps = block_steps
        self.operation = torch.compile(advance_block, backend="inductor", fullgraph=True, dynamic=False)

    @torch.no_grad()
    def loss_and_gradient(self, relative_parameters, duration_seconds):
        """Return (audio, mean_square_loss, gradient_wrt_relative_parameters).

        The loss uses a fixed reference amplitude normalizer. It is the benchmark
        objective, not a generic audio loss. Parameter updates must respect the
        fixed-grid stability region; this function does not clip parameters.
        """
        relative_parameters = torch.as_tensor(relative_parameters, dtype=torch.float64, device='cpu').detach()
        if relative_parameters.shape != (8,) or not torch.isfinite(relative_parameters).all():
            raise ValueError('Expected eight finite relative parameter multipliers.')
        if not math.isfinite(duration_seconds) or duration_seconds < 0:
            raise ValueError('Duration must be finite and non-negative.')
        steps = round(duration_seconds * self.sample_rate)
        states, coefficients, loss_sum = initialize(relative_parameters, self.intervals, self.sample_rate)
        output = torch.empty(steps + 2, dtype=torch.float64)
        midpoint = (self.intervals - 1) // 2
        output[:2] = states[:, midpoint, 0]
        for index in range(0, steps, self.block_steps):
            count = min(self.block_steps, steps - index)
            states, loss_sum, samples = self.operation(states, coefficients, loss_sum, count)
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


def native_block(states, coefficients, loss_sum, steps):
    return _native_kernel().advance_block(states, coefficients, loss_sum, steps, reference.amplitude**2)


class _AudioWithTangents(torch.autograd.Function):
    @staticmethod
    def forward(ctx, relative_parameters, intervals, sample_rate, steps, output_stride, excitation):
        states, coefficients, _ = initialize(relative_parameters, intervals, sample_rate, excitation)
        audio, tangents = _native_kernel().simulate(states, coefficients, steps, output_stride)
        ctx.save_for_backward(tangents)
        return audio

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        (tangents,) = ctx.saved_tensors
        # Chain rule for any scalar loss computed from the returned audio:
        # d loss/d theta_j = sum_i (d loss/d audio_i) (d audio_i/d theta_j).
        return grad_output @ tangents, None, None, None, None, None


class ForwardString:
    """First-order differentiable CPU string with an ordinary PyTorch audio output.

    theta has eight float64 relative multipliers, ordered by PARAMETER_NAMES.
    The spatial/time grid stays fixed throughout optimisation. The first call
    builds a host-optimised C++ extension; later calls reuse its cached binary.
    Use any differentiable PyTorch loss and optimizer. Second derivatives are
    deliberately unsupported. No reverse simulation or checkpointing is used.
    """
    def __init__(self, intervals=128, sample_rate=44100, output_stride=1, excitation="first_mode"):
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
        if not isinstance(theta, torch.Tensor) or theta.device.type != 'cpu' or theta.dtype != torch.float64:
            raise ValueError('theta must be a CPU float64 tensor.')
        if theta.shape != (8,) or not torch.isfinite(theta).all():
            raise ValueError('theta must contain eight finite relative parameter multipliers.')
        if not math.isfinite(duration_seconds) or duration_seconds < 0:
            raise ValueError('Duration must be finite and non-negative.')
        return round(duration_seconds * self.sample_rate)

    def __call__(self, theta, duration_seconds=0.06):
        """Return differentiable midpoint audio, including the two startup samples.

        output_stride selects indices 0, stride, 2*stride, ... of that signal.
        This is observation subsampling, not an anti-aliasing filter. For audio
        downsampling, keep stride=1 and apply a differentiable filter in PyTorch.
        The caller must keep theta inside the fixed-grid stability region.
        """
        steps = self._steps(theta, duration_seconds)
        return _AudioWithTangents.apply(theta, self.intervals, self.sample_rate, steps, self.output_stride, self.excitation)

    @torch.no_grad()
    def output_and_tangents(self, theta, duration_seconds=0.06):
        """Return detached audio and its (n_out, 8) first-order Jacobian."""
        steps = self._steps(theta, duration_seconds)
        states, coefficients, _ = initialize(theta.detach(), self.intervals, self.sample_rate, self.excitation)
        return _native_kernel().simulate(states, coefficients, steps, self.output_stride)
