"""Batched rendering through the cached CUDA adjoint at an arbitrary output rate."""
import math
import torch
import torch.nn.functional as F
from src.kc_string_forward import nominal_parameters, PARAMETER_COUNT
from src.synthesis import antialias_kernel


def theta_from_physical(physical):
    """Solver multipliers from ``(..., 10)`` physical SI values."""
    if physical.shape[-1] != PARAMETER_COUNT:
        raise ValueError(f'Expected {PARAMETER_COUNT} physical parameters.')
    return physical/nominal_parameters(physical.device)


def decimate_batch(raw, oversampling, samples, kernel=None):
    """Batched twin of ``src.synthesis.decimate``: replicate pad, Hann-sinc filter, keep every OF-th sample."""
    if raw.ndim != 2:
        raise ValueError('raw audio must be (batch, samples).')
    if kernel is None:
        kernel = antialias_kernel(oversampling, raw.device)
    half_width = (kernel.shape[-1]-1)//2
    padded = F.pad(raw[:, None, :], (half_width, half_width), mode='replicate')
    return F.conv1d(padded, kernel, stride=oversampling)[:, 0, :samples]


def tail_fade(samples, fade_samples, device=None):
    """Unit window with a raised-cosine fade over the last ``fade_samples`` samples."""
    window = torch.ones(samples, dtype=torch.float64, device=device)
    if fade_samples > 0:
        ramp = 0.5*(1+torch.cos(torch.linspace(0, math.pi, fade_samples, dtype=torch.float64, device=device)))
        window[-fade_samples:] = ramp
    return window


class BatchRenderer:
    """Render a ``(B, 10)`` batch of physical parameters to ``(B, samples)`` audio at ``grid.sample_rate``.

    Forward and backward run in the cached CUDA adjoint (``cache_count`` saved
    scalars per step, ``checkpoint`` steps between state checkpoints). CUDA and
    float64 only, as the kernel requires. ``raw`` returns the internal-rate
    pickup displacement in metres.
    """
    def __init__(self, grid, duration, excitation='raised_cosine:0.08', device='cuda',
                 checkpoint=256, cache_count=3, fade_seconds=0.0, max_steps=50_000_000):
        self.device = torch.device(device)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('Duration must be finite and positive.')
        self.grid, self.duration, self.excitation = grid, float(duration), excitation
        self.checkpoint, self.cache_count = checkpoint, cache_count
        self.steps = round(duration*grid.integration_rate)
        self.samples = round(duration*grid.sample_rate)
        if self.steps > max_steps:
            raise ValueError(f'Render needs {self.steps:,} internal steps, above the limit {max_steps:,}.')
        if grid.intervals > 256:
            raise ValueError('The cached adjoint supports at most 256 intervals.')
        self.kernel = antialias_kernel(grid.oversampling, self.device) if grid.oversampling > 1 else None
        fade = round(fade_seconds*grid.sample_rate)
        self.fade = tail_fade(self.samples, fade, self.device) if fade > 0 else None

    def raw(self, physical):
        from experiments.scalar_cache.reconstruct import simulate   # JIT-builds the CUDA kernel on first use
        if physical.ndim != 2 or physical.dtype != torch.float64:
            raise ValueError('physical parameters must be a (batch, 10) float64 tensor.')
        theta = theta_from_physical(physical.to(self.device))
        return simulate(theta, self.grid.intervals, self.grid.integration_rate, self.steps, self.excitation,
                        self.checkpoint, self.cache_count, self.grid.order)

    def __call__(self, physical):
        raw = self.raw(physical)
        audio = decimate_batch(raw, self.grid.oversampling, self.samples, self.kernel) if self.kernel is not None \
            else raw[:, :self.samples]
        return audio*self.fade if self.fade is not None else audio

    def describe(self):
        return {'sample_rate': self.grid.sample_rate, 'oversampling': self.grid.oversampling,
                'integration_rate': self.grid.integration_rate, 'intervals': self.grid.intervals,
                'order': self.grid.order, 'duration': self.duration, 'steps': self.steps, 'samples': self.samples,
                'excitation': self.excitation, 'checkpoint': self.checkpoint, 'cache_count': self.cache_count}
