"""Batched, differentiable rendering at an output rate through the hand-written adjoint."""

import math

import torch
import torch.nn.functional as F

from . import adjoint, physics


def antialias_kernel(oversampling, device=None):
    """Hann-windowed sinc low-pass, cutoff 0.45 of the output rate, 32 * oversampling + 1 taps, unit DC gain.

    Shaped ``(1, 1, taps)`` for ``conv1d``.
    """
    half_width = 16 * oversampling
    index = torch.arange(-half_width, half_width + 1, dtype=torch.float64)
    cutoff = 0.45 / oversampling
    window = torch.hann_window(index.numel(), periodic=False, dtype=torch.float64)
    kernel = 2 * cutoff * torch.sinc(2 * cutoff * index) * window
    return (kernel / kernel.sum()).reshape(1, 1, -1).to(device)


def decimate(audio, oversampling, samples, kernel=None):
    """Replicate-pad, low-pass and keep every ``oversampling``-th sample of ``(batch, n)`` audio."""
    if kernel is None:
        kernel = antialias_kernel(oversampling, audio.device)
    half_width = (kernel.shape[-1] - 1) // 2
    padded = F.pad(audio[:, None, :], (half_width, half_width), mode="replicate")
    return F.conv1d(padded, kernel, stride=oversampling)[:, 0, :samples]


def tail_fade(samples, fade_samples, device=None):
    """Unit window ending in a raised-cosine fade over ``fade_samples``."""
    window = torch.ones(samples, dtype=torch.float64, device=device)
    if fade_samples > 0:
        ramp = torch.linspace(0, math.pi, fade_samples, dtype=torch.float64, device=device)
        window[-fade_samples:] = 0.5 * (1 + torch.cos(ramp))
    return window


class Renderer:
    """``(batch, 10)`` physical SI parameters to ``(batch, samples)`` audio at ``grid.sample_rate``.

    The scheme runs at the grid's internal rate in the CUDA kernel and is
    decimated to the output rate. ``derivative=1`` returns the pickup velocity
    (a first difference at the output rate) instead of its displacement.
    """

    def __init__(self, grid, duration, excitation, device="cuda", checkpoint=256, fade_seconds=0.0, derivative=0):
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("The duration must be finite and positive.")
        if derivative not in (0, 1):
            raise ValueError("derivative must be 0 (displacement) or 1 (velocity).")
        self.grid, self.duration, self.excitation = grid, float(duration), excitation
        self.device, self.checkpoint, self.derivative = torch.device(device), checkpoint, derivative
        self.steps = round(duration * grid.integration_rate)
        self.samples = round(duration * grid.sample_rate)
        self.kernel = antialias_kernel(grid.oversampling, self.device) if grid.oversampling > 1 else None
        fade = round(fade_seconds * grid.sample_rate)
        self.fade = tail_fade(self.samples, fade, self.device) if fade > 0 else None

    def raw(self, physical):
        """Pickup displacement (m) at the internal rate, including the two start-up samples."""
        theta = physics.multipliers(physical.to(self.device, torch.float64))
        grid = self.grid
        return adjoint.simulate(
            theta, grid.intervals, grid.integration_rate, self.steps, self.excitation, self.checkpoint, grid.order
        )

    def __call__(self, physical):
        raw = self.raw(physical)
        if self.kernel is None:
            audio = raw[:, : self.samples]
        else:
            audio = decimate(raw, self.grid.oversampling, self.samples, self.kernel)
        if self.derivative:
            audio = torch.diff(audio, dim=-1, prepend=audio[:, :1]) * self.grid.sample_rate
        return audio if self.fade is None else audio * self.fade

    def describe(self):
        """The rendering settings, for result records."""
        grid = self.grid
        return {
            "sample_rate": grid.sample_rate,
            "oversampling": grid.oversampling,
            "integration_rate": grid.integration_rate,
            "intervals": grid.intervals,
            "order": grid.order,
            "duration": self.duration,
            "steps": self.steps,
            "samples": self.samples,
            "excitation": self.excitation,
            "checkpoint": self.checkpoint,
            "signal": "velocity" if self.derivative else "displacement",
        }
