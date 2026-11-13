"""Bounded parameter boxes: physical ranges and the unconstrained coordinates optimisers use.

A box maps optimiser logits to physical values (log-uniform for the positive
coordinates, linear for the others) and gives the worst cases of its stability
condition. Subclasses only supply their ranges, names and ``to_material``.
"""

import math
from dataclasses import dataclass

import torch

from . import physics


class BoundedParameters:
    """Bounds and the logit transforms shared by every box; all methods accept ``(..., n)`` tensors."""

    names: tuple
    units: tuple
    positive_count: int  # the first coordinates are positive and log-interpolated

    def _validate_ranges(self):
        object.__setattr__(self, "ranges", tuple(tuple(float(v) for v in pair) for pair in self.ranges))
        if len(self.ranges) != len(self.names) or any(len(pair) != 2 for pair in self.ranges):
            raise ValueError(f"Supply one lower/upper pair for each of {len(self.names)} parameters.")
        for i, (low, high) in enumerate(self.ranges):
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                raise ValueError(f"{self.names[i]} needs finite, increasing bounds.")
            if low < 0 or (i < self.positive_count and low == 0):
                raise ValueError("Physical parameters must be positive; losses may be zero.")

    def bounds(self, device=None):
        """Lower and upper bounds as two float64 tensors."""
        return torch.tensor(self.ranges, dtype=torch.float64, device=device).unbind(dim=1)

    def decode(self, logits):
        """Physical values from unconstrained logits."""
        low, high = self.bounds(logits.device)
        n = self.positive_count
        fraction = logits.sigmoid()
        log_low, log_high = low[:n].log(), high[:n].log()
        positive = (log_low + fraction[..., :n] * (log_high - log_low)).exp().clamp(low[:n], high[:n])
        linear = low[n:] + fraction[..., n:] * (high[n:] - low[n:])
        return torch.cat((positive, linear), dim=-1)

    def encode(self, values, margin=1e-6):
        """Logits of values clamped into the box; endpoints move inward by ``margin`` to stay finite."""
        low, high = self.bounds(values.device)
        n = self.positive_count
        values = values.clamp(low, high)
        fraction = torch.cat(
            (
                (values[..., :n].log() - low[:n].log()) / (high[:n].log() - low[:n].log()),
                (values[..., n:] - low[n:]) / (high[n:] - low[n:]),
            ),
            dim=-1,
        )
        return torch.logit(fraction.clamp(margin, 1 - margin))

    def corners(self):
        """All 2^n corners of the box, as physical values."""
        n = len(self.names)
        signs = torch.tensor([[(i >> k) & 1 for k in range(n)] for i in range(2**n)], dtype=torch.float64)
        low, high = self.bounds()
        return low + signs * (high - low)

    def to_material(self, values):
        """The solver's ten physical SI parameters for ``(..., n)`` box values."""
        raise NotImplementedError


@dataclass(frozen=True)
class MaterialParameterBox(BoundedParameters):
    """The solver's ten physical parameters directly, in ``physics.PARAMETER_NAMES`` order."""

    ranges: tuple
    names = physics.PARAMETER_NAMES
    units = ("kg/m³", "N", "m", "Pa", "m", "m", "1/s", "m²/s", "L", "L")
    positive_count = 6

    def __post_init__(self):
        self._validate_ranges()

    def to_material(self, values):
        return values

    def stability_cases(self):
        """(c, K, sigma_1, L) at the worst corner: least density and length, most tension, modulus and sigma_1.

        The radius enters as A / r^2 + C r^2, which is convex in r^2, so both of its ends are checked.
        """
        density, tension = self.ranges[0][0], self.ranges[1][1]
        modulus, length, sigma_1 = self.ranges[3][1], self.ranges[4][0], self.ranges[7][1]
        return [
            (*physics.wave_speed_and_stiffness(density, tension, radius, modulus), sigma_1, length)
            for radius in self.ranges[2]
        ]
