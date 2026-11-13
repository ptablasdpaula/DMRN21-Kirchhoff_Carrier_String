"""Direct material or perceptual coordinates with bounded optimisation variables.

f0 is the tension-only fundamental; f1 = f0*sqrt(1+B) includes linear stiffness.
Only perceptual mode fixes density and length. All tensors use physical units and float64 on CPU or CUDA.
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
    """Differentiable map to rho, T, r, E, L, a, sigma0, sigma1, plus the nominal pluck and pickup positions."""
    f0, B, amplitude, radius, sigma0, sigma1 = values.unbind()
    area = math.pi * radius**2
    inertia = math.pi * radius**4 / 4
    tension = 4 * density * area * length**2 * f0**2
    modulus = B * tension * length**2 / (math.pi**2 * inertia)
    return torch.stack((values.new_tensor(density), tension, radius, modulus,
                        values.new_tensor(length), amplitude, sigma0, sigma1,
                        values.new_tensor(reference.pluck_position), values.new_tensor(reference.pickup_position)))


def material_multipliers(values, density=reference.material_density, length=reference.length):
    return material_parameters(values, density, length) / nominal_parameters(values.device)


class BoundedParameters:
    """Common bounds and transforms; each coordinate system supplies its map."""
    def _validate_ranges(self):
        object.__setattr__(self, 'ranges', tuple(tuple(float(v) for v in pair) for pair in self.ranges))
        if len(self.ranges) != len(self.names) or any(len(pair) != 2 for pair in self.ranges):
            raise ValueError(f'Supply one lower/upper pair for each of {len(self.names)} parameters.')
        for i, (low, high) in enumerate(self.ranges):
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                raise ValueError(f'{self.names[i]} needs finite, increasing bounds.')
            if low < 0 or (i < self.positive_count and low == 0):
                raise ValueError('Physical parameters must be positive; losses may be zero.')

    def bounds(self, device=None):
        return torch.tensor(self.ranges, dtype=torch.float64, device=device).unbind(dim=1)

    def validate(self, values):
        low, high = self.bounds(values.device)
        if values.shape != (len(self.names),) or values.dtype != torch.float64 or values.device.type not in ('cpu','cuda'):
            raise ValueError(f'Parameters must be a {len(self.names)}-element CPU/CUDA float64 tensor.')
        if not bool(torch.isfinite(values).all()) or not bool(((values >= low) & (values <= high)).all()):
            raise ValueError('Parameters must be finite and inside the prepared box.')

    def decode(self, unconstrained):
        low, high = self.bounds(unconstrained.device); n = self.positive_count
        fraction = unconstrained.sigmoid()
        positive = (low[:n].log() + fraction[:n]*(high[:n].log()-low[:n].log())).exp()
        positive = positive.clamp(low[:n], high[:n])
        losses = low[n:] + fraction[n:]*(high[n:]-low[n:])
        return torch.cat((positive, losses))

    def encode(self, values):
        self.validate(values)
        low, high = self.bounds(values.device); n = self.positive_count
        fraction = torch.cat(((values[:n].log()-low[:n].log())/(high[:n].log()-low[:n].log()),
                              (values[n:]-low[n:])/(high[n:]-low[n:])))
        # Selected endpoint starts move inward for finite logits; frozen values stay exact.
        return torch.logit(fraction.clamp(1e-6, 1-1e-6))

    def perturb(self, target):
        self.validate(target)
        low, high = self.bounds(target.device)
        signs = target.new_tensor([1. if i%2==0 else -1. for i in range(len(self.names))])
        return (target * torch.exp(signs * math.log(1.1))).clamp(low, high)


@dataclass(frozen=True)
class ParameterBox(BoundedParameters):
    ranges: tuple = ((60., 100.), (1e-5, 1e-3), (.0005, .003),
                     (.9*reference.radius, 1.1*reference.radius), (0., 2.), (0., .001))
    density: float = reference.material_density
    length: float = reference.length
    names = NAMES
    units = UNITS
    positive_count = 4
    coordinate_system = 'perceptual'

    def __post_init__(self):
        self._validate_ranges()
        if not all(math.isfinite(x) and x > 0 for x in (self.density, self.length)):
            raise ValueError('Fixed density and length must be positive and finite.')

    def defaults(self):
        return reference_parameters()

    def to_material(self, values):
        return material_parameters(values, self.density, self.length)

    def stability_cases(self):
        f0, B, sigma1 = self.ranges[0][1], self.ranges[1][1], self.ranges[5][1]
        c = 2*self.length*f0
        return [(c, c*self.length*math.sqrt(B)/math.pi, sigma1, self.length)]

    @property
    def length_range(self):
        return (self.length, self.length)


BENCHMARK_SPREAD = .05


@dataclass(frozen=True)
class BenchmarkBox(ParameterBox):
    """Narrow perceptual box for the timing benchmark.

    Every control lies within exp(±0.05) of the reference string; the benchmark
    perturbs by at most exp(±0.045). Its stability corner admits grids (176
    intervals at 2× with the five-point stencil) that the full fitting box's
    corner does not.
    """
    ranges: tuple = tuple((float(v)*math.exp(-BENCHMARK_SPREAD), float(v)*math.exp(BENCHMARK_SPREAD))
                          for v in reference_parameters())


@dataclass(frozen=True)
class MaterialParameterBox(BoundedParameters):
    """Direct physical controls in the native kernel's parameter order.

    The last two entries are the pluck centre and the pickup position as
    fractions of the length; they are bounded linearly, away from the ends.
    """
    ranges: tuple = tuple((.8*v, 1.2*v) for v in (
        reference.material_density, reference.tension, reference.radius,
        reference.young_modulus, reference.length, reference.amplitude)) + ((0.,2.), (0.,.001), (.05,.95), (.05,.95))
    names = ('material_density','tension','radius','young_modulus','length','amplitude','sigma_0','sigma_1',
             'pluck_position','pickup_position')
    units = ('kg/m³','N','m','Pa','m','m','1/s','m²/s','L','L')
    positive_count = 6
    coordinate_system = 'material'

    def __post_init__(self):
        self._validate_ranges()

    def defaults(self):
        return nominal_parameters()

    def to_material(self, values):
        # No f0/B map in either synthesis or fitting.
        return values

    @property
    def length_range(self):
        return self.ranges[4]

    def stability_cases(self):
        # Score is maximised by rho_min, T_max, E_max, L_min, sigma1_max.
        # Radius enters as A/r² + C*r², convex in r²: check BOTH endpoints.
        rho=self.ranges[0][0]; tension=self.ranges[1][1]
        modulus=self.ranges[3][1]; length=self.ranges[4][0]; sigma1=self.ranges[7][1]
        return [(math.sqrt(tension/(rho*math.pi*r*r)), math.sqrt(modulus*r*r/(4*rho)), sigma1, length)
                for r in self.ranges[2]]


BENCHMARK_MATERIAL_SPREAD = .03


@dataclass(frozen=True)
class BenchmarkMaterialBox(MaterialParameterBox):
    """Narrow box around the nominal string for the timing benchmark.

    Every one of the ten parameters lies within exp(±0.03) of the nominal
    string; the benchmark perturbs by at most that. Its stability corner
    (worst case 0.93 of the bound) admits 176 intervals at 2× oversampling with
    the five-point stencil; ±5 % on all material parameters would not (1.07).
    """
    ranges: tuple = tuple((float(v)*math.exp(-BENCHMARK_MATERIAL_SPREAD), float(v)*math.exp(BENCHMARK_MATERIAL_SPREAD))
                          for v in nominal_parameters())


def perceptual_from_material(values):
    """Derived diagnostics only; material-mode synthesis never calls this map."""
    rho,tension,radius,modulus,length,amplitude,sigma0,sigma1=values[:8].unbind()
    c=torch.sqrt(tension/(rho*math.pi*radius**2))
    B=math.pi**2*modulus*(math.pi*radius**4/4)/(tension*length**2)
    return torch.stack((c/(2*length),B,amplitude,radius,sigma0,sigma1))
