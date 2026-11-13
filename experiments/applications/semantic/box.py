"""A wide string box for semantic (CLAP-guided) optimisation.

Coordinates: stress s = T/(pi r^2) (so the wave speed does not depend on the
radius and box corners stay physical), radius, length, amplitude, sigma_0,
sigma_1 (all log-interpolated), then a material axis m in [0, 1] that moves
(density, Young's modulus) log-linearly from nylon to steel, and the pluck and
pickup positions (linear). ``to_material`` returns the solver's ten SI
parameters.
"""
from dataclasses import dataclass
import math
import torch
from src.grid import GridConfig
from src.kc_string_forward import nominal_parameters
from src.parameters import BoundedParameters

NYLON = (1150.0, 3e9)       # kg/m^3, Pa
STEEL = (8050.0, 174e9)     # the paper's nominal string
SAMPLE_RATE = 48000         # CLAP's input rate
DURATION = 2.5              # tiles exactly four times into CLAP's 10 s window
EXCITATION = 'raised_cosine:0.08'
FADE_SECONDS = 0.02


@dataclass(frozen=True)
class WideStringBox(BoundedParameters):
    ranges: tuple = ((50e6, 1e9), (0.15e-3, 0.6e-3), (0.5, 1.0), (0.2e-3, 3e-3), (0.05, 50.0), (1e-6, 2e-3),
                     (0.0, 1.0), (0.10, 0.90), (0.05, 0.95))
    names = ('stress', 'radius', 'length', 'amplitude', 'sigma_0', 'sigma_1', 'material', 'pluck_position', 'pickup_position')
    units = ('Pa', 'm', 'm', 'm', '1/s', 'm²/s', 'nylon→steel', 'L', 'L')
    positive_count = 6
    coordinate_system = 'material'

    def __post_init__(self):
        self._validate_ranges()

    @staticmethod
    def material(m):
        """Density and Young's modulus along the nylon→steel axis (log-linear)."""
        rho = NYLON[0]**(1-m)*STEEL[0]**m
        modulus = NYLON[1]**(1-m)*STEEL[1]**m
        return rho, modulus

    def to_material(self, values):
        stress, radius, length, amplitude, sigma0, sigma1, m, pluck, pickup = values.unbind(-1)
        rho, modulus = self.material(m)
        tension = stress*math.pi*radius**2
        return torch.stack((rho, tension, radius, modulus, length, amplitude, sigma0, sigma1, pluck, pickup), dim=-1)

    def theta(self, values):
        return self.to_material(values)/nominal_parameters(values.device)

    def defaults(self):
        """The paper's nominal steel string in these coordinates."""
        nominal = nominal_parameters()
        rho, tension, radius, _, length, amplitude, sigma0, sigma1, pluck, pickup = nominal.tolist()
        stress = tension/(math.pi*radius**2)
        return torch.tensor([stress, radius, length, amplitude, sigma0, sigma1, 1.0, pluck, pickup], dtype=torch.float64)

    @property
    def length_range(self):
        return self.ranges[2]

    def stability_cases(self):
        """(c, K, sigma1_max, L_min) at the material ends × radius ends: the score is convex in both."""
        stress_max = self.ranges[0][1]
        sigma1_max, length_min = self.ranges[5][1], self.ranges[2][0]
        cases = []
        for m in (0.0, 1.0):
            rho, modulus = self.material(m)
            for radius in self.ranges[1]:
                cases.append((math.sqrt(stress_max/rho), math.sqrt(modulus*radius**2/(4*rho)), sigma1_max, length_min))
        return cases


def production_grid(box=None):
    return GridConfig(box or WideStringBox(), sample_rate=SAMPLE_RATE, oversampling=12, intervals=184)


def pilot_grid(box=None):
    return GridConfig(box or WideStringBox(), sample_rate=SAMPLE_RATE, oversampling=8, intervals=148)


def perceptual(material):
    """f0, f1, B and the T60 of mode 1 and of the mode nearest 1 kHz for ``(..., 10)`` SI values."""
    rho, tension, radius, modulus, length, amplitude, sigma0, sigma1 = material[..., :8].unbind(-1)
    mu = rho*math.pi*radius**2
    f0 = torch.sqrt(tension/mu)/(2*length)
    B = math.pi**2*modulus*(math.pi*radius**4/4)/(tension*length**2)
    f1 = f0*torch.sqrt(1+B)
    def t60(n):
        return 3*math.log(10)/(sigma0 + sigma1*(n*math.pi/length)**2)
    n_1k = torch.clamp(torch.round(1000.0/f0), min=1)
    return {'f0_hz': f0, 'f1_hz': f1, 'B': B, 'mu_kg_per_m': mu, 't60_mode1_s': t60(1), 't60_1khz_s': t60(n_1k)}
