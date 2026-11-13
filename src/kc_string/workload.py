"""The paper's benchmark workload (Table 1), defined once for the benchmark and the tests.

One sound-matching step: a batch of strings, each of the ten parameters of the
nominal steel string moved by up to +-3 %, is simulated for 1 s on 176 intervals
at 88.2 kHz, decimated to 44.1 kHz, and the mean absolute error of each waveform
against the nominal string's is backpropagated to all ten parameters, with a
checkpoint every 256 steps.
"""

import math

import torch

from . import physics
from .boxes import MaterialParameterBox
from .grid import Grid

SPREAD = 0.03
BOX = MaterialParameterBox(
    ranges=tuple((v * math.exp(-SPREAD), v * math.exp(SPREAD)) for v in physics.NOMINAL),
)
GRID = Grid(BOX, sample_rate=44100, oversampling=2, intervals=176)
DURATION = 1.0
EXCITATION = "raised_cosine"  # the reference's half-width, floor((N-1)/4) nodes: 0.24 L
CHECKPOINT = 256
STEPS = round(DURATION * GRID.integration_rate)
SAMPLES = round(DURATION * GRID.sample_rate)
BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128)


def batch(size):
    """``size`` strings (physical SI values); item i is the same string in every batch size."""
    i = torch.arange(size, dtype=torch.float64)[:, None]
    j = torch.arange(physics.PARAMETER_COUNT, dtype=torch.float64)[None, :]
    values = physics.nominal_parameters()[None, :] * torch.exp(SPREAD * torch.sin((i + 1) * (j + 1) * 0.73))
    low, high = BOX.bounds()
    return torch.maximum(low, torch.minimum(high, values))


def mae(audio, target):
    """Mean absolute error of each waveform against the target, summed over the batch."""
    return (audio - target).abs().mean(dim=-1).sum()
