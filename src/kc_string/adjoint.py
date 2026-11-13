"""The paper's method: the scheme stepped in its sine modes, with a hand-written adjoint on the GPU.

In the discrete sine basis both difference operators are diagonal, so each step
is one multiplication per mode plus a rank-one coupling through the nonlinear
tension. ``adjoint.cu`` runs one CUDA block per string and one thread per mode.
The forward pass keeps a checkpoint every ``checkpoint`` steps; the backward pass
recomputes one segment at a time and runs the scheme's adjoint recurrence
through it. That yields adjoints of the start-up states and of the per-string
coefficients, which the coefficients' tangents (``scheme.initialize``) turn into
gradients of the ten parameter multipliers.
"""

from functools import cache
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable

from . import physics
from .scheme import ORDER, discrete_eigenvalues, dual_multiply, initialize, pickup_weights, sine_transform

MAX_INTERVALS = 256  # one thread per mode in a block


@cache
def _kernel():
    """Build the CUDA extension on first use (cached by PyTorch across processes)."""
    from torch.utils.cpp_extension import load

    return load(
        name="kc_checkpoint_adjoint",
        sources=[str(Path(__file__).with_name("adjoint.cu"))],
        extra_cuda_cflags=["-O3", "--fmad=false", "-lineinfo"],
        with_cuda=True,
        verbose=False,
    )


def _modal_inputs(theta, intervals, rate, excitation, order):
    """Start-up states in modal coordinates, kernel coefficients and modal pickup weights, all dual."""
    states, coefficients = torch.vmap(lambda row: initialize(row, intervals, rate, excitation, order))(theta)
    pickup = torch.vmap(lambda row: pickup_weights(row, intervals, spatial=False))(theta).contiguous()
    # The kernel works with h^2 D_xx: fold 1/h^2 into the coefficients that multiply D_xx.
    h2 = coefficients[:, 0].clone()
    coefficients[:, 2] = dual_multiply(coefficients[:, 2], h2)
    coefficients[:, 3] = dual_multiply(dual_multiply(coefficients[:, 3], h2), h2)
    coefficients[:, 5] = dual_multiply(coefficients[:, 5], h2)
    coefficients[:, 7] = dual_multiply(coefficients[:, 7], h2)
    transform = sine_transform(intervals, theta.device)
    states = torch.einsum("ij,btjd->btid", transform.T, states).contiguous()
    return states, coefficients, pickup


class _Adjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, intervals, rate, steps, excitation, checkpoint, order):
        states, coefficients, pickup = _modal_inputs(theta, intervals, rate, excitation, order)
        eigenvalues = discrete_eigenvalues(intervals, order, theta.device)
        audio, checkpoints = _kernel().forward(
            states[..., 0].contiguous(),
            coefficients[..., 0].contiguous(),
            eigenvalues,
            pickup[..., 0].contiguous(),
            steps,
            checkpoint,
        )
        ctx.save_for_backward(states, coefficients, eigenvalues, pickup, checkpoints)
        ctx.steps, ctx.checkpoint = steps, checkpoint
        return audio

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_audio):
        states, coefficients, eigenvalues, pickup, checkpoints = ctx.saved_tensors
        state_bar, coefficient_bar, pickup_bar = _kernel().backward(
            coefficients[..., 0].contiguous(),
            eigenvalues,
            pickup[..., 0].contiguous(),
            grad_audio.contiguous(),
            checkpoints,
            ctx.steps,
            ctx.checkpoint,
        )
        # Chain rule through the per-mode coefficients A = 2 m + e c2 - e^2 c3, P = c4 - e c5, Q = e c7
        # and the inverse mass m (the kernel returns adjoints of A, P, Q per mode, then of m).
        n = eigenvalues.numel()
        e = eigenvalues[None, :, None]
        tangents = coefficients[..., 1:]
        d_a = 2 * tangents[:, 1, None, :] + e * tangents[:, 2, None, :] - e.square() * tangents[:, 3, None, :]
        d_p = tangents[:, 4, None, :] - e * tangents[:, 5, None, :]
        d_q = e * tangents[:, 7, None, :]
        gradient = torch.einsum("bti,btid->bd", state_bar, states[..., 1:])
        gradient = gradient + torch.einsum("bi,bid->bd", coefficient_bar[:, :n], d_a)
        gradient = gradient + torch.einsum("bi,bid->bd", coefficient_bar[:, n : 2 * n], d_p)
        gradient = gradient + torch.einsum("bi,bid->bd", coefficient_bar[:, 2 * n : 3 * n], d_q)
        gradient = gradient + coefficient_bar[:, 3 * n, None] * tangents[:, 6, :]
        gradient = gradient + torch.einsum("bi,bid->bd", pickup_bar, pickup[..., 1:])
        return gradient, None, None, None, None, None, None


def simulate(theta, intervals, rate, steps, excitation, checkpoint=256, order=ORDER):
    """Differentiable pickup audio ``(batch, steps + 2)`` for a ``(batch, 10)`` batch of multipliers.

    CUDA and float64 only, at most ``MAX_INTERVALS`` intervals. The backward pass
    is the hand-written adjoint; only first derivatives are available.
    """
    if theta.dtype != torch.float64 or not theta.is_cuda or theta.ndim != 2:
        raise ValueError(f"Expected a CUDA float64 tensor of shape (batch, {physics.PARAMETER_COUNT}).")
    if theta.shape[0] == 0 or theta.shape[1] != physics.PARAMETER_COUNT:
        raise ValueError(f"Expected a nonempty batch of {physics.PARAMETER_COUNT} multipliers.")
    if intervals < 4 or intervals % 2 or intervals > MAX_INTERVALS or steps < 0 or checkpoint < 1:
        raise ValueError("Invalid grid, step count or checkpoint interval.")
    return _Adjoint.apply(theta, intervals, rate, steps, excitation, checkpoint, order)
