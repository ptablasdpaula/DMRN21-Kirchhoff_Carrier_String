"""Perceptual coordinates and bounded optimisation variables.

f0 is the tension-only fundamental; f1 = f0*sqrt(1+B) includes linear stiffness.
Density and length are fixed. All tensors use physical units, CPU float64.
"""
from dataclasses import dataclass
import math
import torch
from . import kc_string as reference
from .kc_string_forward import nominal_parameters

NAMES = ('f0', 'B', 'amplitude', 'radius', 'sigma_0', 'sigma_1')
UNITS = ('Hz', 'dimensionless', 'm', 'm', '1/s', 'm²/s')


def reference_parameters():
    B = math.pi**2 * reference.young_modulus * reference.moment_of_inertia / (reference.tension * reference.length**2)
    return torch.tensor([reference.wave_speed/(2*reference.length), B, reference.amplitude,
                         reference.radius, reference.sigma_0, reference.sigma_1], dtype=torch.float64)


def material_parameters(values, density=reference.material_density, length=reference.length):
    """Differentiable map to rho, T, r, E, L, a, sigma0, sigma1."""
    f0, B, amplitude, radius, sigma0, sigma1 = values.unbind()
    area = math.pi * radius**2
    inertia = math.pi * radius**4 / 4
    tension = 4 * density * area * length**2 * f0**2
    modulus = B * tension * length**2 / (math.pi**2 * inertia)
    return torch.stack((values.new_tensor(density), tension, radius, modulus,
                        values.new_tensor(length), amplitude, sigma0, sigma1))


def material_multipliers(values, density=reference.material_density, length=reference.length):
    return material_parameters(values, density, length) / nominal_parameters()


@dataclass(frozen=True)
class ParameterBox:
    ranges: tuple = ((60., 100.), (1e-5, 1e-3), (.0005, .003),
                     (.9*reference.radius, 1.1*reference.radius), (0., 2.), (0., .001))
    density: float = reference.material_density
    length: float = reference.length

    def __post_init__(self):
        object.__setattr__(self, 'ranges', tuple(tuple(float(v) for v in pair) for pair in self.ranges))
        if len(self.ranges) != 6 or any(len(pair) != 2 for pair in self.ranges):
            raise ValueError('Supply one lower/upper pair for each of the six parameters.')
        for i, (low, high) in enumerate(self.ranges):
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                raise ValueError(f'{NAMES[i]} needs finite, increasing bounds.')
            if low < 0 or (i < 4 and low == 0):
                raise ValueError('f0, B, amplitude and radius must be positive; losses may be zero.')
        if not all(math.isfinite(x) and x > 0 for x in (self.density, self.length)):
            raise ValueError('Fixed density and length must be positive and finite.')

    def bounds(self):
        return torch.tensor(self.ranges, dtype=torch.float64).unbind(dim=1)

    def validate(self, values):
        low, high = self.bounds()
        if values.shape != (6,) or values.dtype != torch.float64 or values.device.type != 'cpu':
            raise ValueError('Parameters must be a six-element CPU float64 tensor.')
        if not bool(torch.isfinite(values).all()) or not bool(((values >= low) & (values <= high)).all()):
            raise ValueError('Parameters must be finite and inside the prepared box.')

    def decode(self, unconstrained):
        low, high = self.bounds()
        # Log interpolation for positive parameters, linear interpolation for losses.
        fraction = unconstrained.sigmoid()
        positive = (low[:4].log() + fraction[:4]*(high[:4].log()-low[:4].log())).exp()
        # Avoid a one-ulp exp(log(bound)) overshoot at saturated logits.
        positive = positive.clamp(low[:4], high[:4])
        losses = low[4:] + fraction[4:]*(high[4:]-low[4:])
        return torch.cat((positive, losses))

    def encode(self, values):
        self.validate(values)
        low, high = self.bounds()
        fraction = torch.cat(((values[:4].log()-low[:4].log())/(high[:4].log()-low[:4].log()),
                              (values[4:]-low[4:])/(high[4:]-low[4:])))
        # Finite logits cannot represent endpoints. Nudge selected endpoint starts
        # inside by one millionth of their coordinate range; frozen entries stay exact.
        return torch.logit(fraction.clamp(1e-6, 1-1e-6))

    def perturb(self, target):
        self.validate(target)
        low, high = self.bounds()
        signs = target.new_tensor([1., -1., 1., -1., 1., -1.])
        return (target * torch.exp(signs * math.log(1.1))).clamp(low, high)
