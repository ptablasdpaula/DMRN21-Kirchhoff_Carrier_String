"""The nominal steel string and closed-form quantities of any string.

The nominal string and its damping follow Russo et al.'s ``KCStringSAV_noSplit.m``
(``nLinStringsConv-SAV/KCstring``), including its factor 6 in the decay rates.
Every function of string parameters accepts floats or tensors of any shape.
"""

import math

import torch

# The ten solver parameters, in the order used everywhere (positions as fractions of L).
PARAMETER_NAMES = (
    "material_density",
    "tension",
    "radius",
    "young_modulus",
    "length",
    "amplitude",
    "sigma_0",
    "sigma_1",
    "pluck_position",
    "pickup_position",
)
PARAMETER_COUNT = len(PARAMETER_NAMES)
PLUCK = PARAMETER_NAMES.index("pluck_position")
PICKUP = PARAMETER_NAMES.index("pickup_position")


def linear_mass_density(density, radius):
    """Mass per unit length rho * pi * r^2 (kg/m)."""
    return density * (math.pi * radius**2)


def moment_of_inertia(radius):
    """Second moment of area pi * r^4 / 4 of a round string (m^4)."""
    return math.pi * radius**4 / 4


def fundamental(density, tension, radius, length):
    """Tension-only fundamental f0 = sqrt(T / mu) / (2 L) (Hz)."""
    return (tension / linear_mass_density(density, radius)) ** 0.5 / (2 * length)


def inharmonicity(tension, radius, modulus, length):
    """Inharmonicity coefficient B = pi^2 E I / (T L^2); f1 = f0 sqrt(1 + B)."""
    return math.pi**2 * modulus * moment_of_inertia(radius) / (tension * length**2)


def t60(sigma_0, sigma_1, mode, length):
    """Amplitude T60 (s) of a mode under the damping sigma_0 + sigma_1 (n pi / L)^2."""
    return 3 * math.log(10) / (sigma_0 + sigma_1 * (mode * math.pi / length) ** 2)


def string_quantities(material):
    """f0, f1 = f0 sqrt(1 + B), B and the mass per length of ``(..., 10)`` physical values (floats or tensors)."""
    density, tension, radius, modulus, length = (material[..., i] for i in range(5))
    f0 = fundamental(density, tension, radius, length)
    b = inharmonicity(tension, radius, modulus, length)
    return {"f0_hz": f0, "f1_hz": f0 * (1 + b) ** 0.5, "B": b, "mu_kg_per_m": linear_mass_density(density, radius)}


def wave_speed_and_stiffness(density, tension, radius, modulus):
    """Wave speed c = sqrt(T / mu) and stiffness K = sqrt(E I / mu)."""
    mass = linear_mass_density(density, radius)
    return (tension / mass) ** 0.5, (modulus * moment_of_inertia(radius) / mass) ** 0.5


def minimum_grid_spacing(time_step, wave_speed, stiffness, sigma_1):
    """Smallest stable spacing of the three-point scheme: h^4 - (c^2 k^2 + 4 sigma_1 k) h^2 - 4 K^2 k^2 = 0."""
    wave_and_damping = wave_speed**2 * time_step**2 + 4 * sigma_1 * time_step
    discriminant = wave_and_damping**2 + 16 * stiffness**2 * time_step**2
    return math.sqrt((wave_and_damping + math.sqrt(discriminant)) / 2)


def _squared_wavenumber(angular_frequency, wave_speed, stiffness):
    """Positive root z of K^2 z^2 + c^2 z - omega^2 = 0 (z is the squared wavenumber)."""
    discriminant = wave_speed**4 + 4 * stiffness**2 * angular_frequency**2
    return (-(wave_speed**2) + math.sqrt(discriminant)) / (2 * stiffness**2)


def _reference_damping(wave_speed, stiffness):
    """sigma_0, sigma_1 giving T60 = 15 s at 0 Hz and 10 s at 1 kHz, as in the reference."""
    z1 = _squared_wavenumber(0.0, wave_speed, stiffness)
    z2 = _squared_wavenumber(2 * math.pi * 1000.0, wave_speed, stiffness)
    rate1, rate2 = 6 * math.log(10) / 15.0, 6 * math.log(10) / 10.0
    sigma_1 = (rate2 - rate1) / (z2 - z1)
    return rate1 - sigma_1 * z1, sigma_1


# The nominal steel string of the reference code (0.016 gauge, 1 m, 75 N).
# The density is written as in the reference (8.05 * 10^3, which is 8050.000000000001 in float64).
DENSITY, TENSION, RADIUS, MODULUS, LENGTH = 8.05 * 10**3, 75.0, 3.5560e-4, 174e9, 1.0
AMPLITUDE = 2e-3
# Same arithmetic order as the reference, so the nominal damping is reproduced bit for bit.
_MASS = linear_mass_density(DENSITY, RADIUS)
WAVE_SPEED = math.sqrt(TENSION / _MASS)
STIFFNESS = math.sqrt(MODULUS * moment_of_inertia(RADIUS) / _MASS)
SIGMA_0, SIGMA_1 = _reference_damping(WAVE_SPEED, STIFFNESS)
NOMINAL = (DENSITY, TENSION, RADIUS, MODULUS, LENGTH, AMPLITUDE, SIGMA_0, SIGMA_1, 0.5, 0.5)


def nominal_parameters(device=None):
    """The nominal string's ten parameters as a float64 tensor."""
    return torch.tensor(NOMINAL, dtype=torch.float64, device=device)


def multipliers(physical):
    """Solver inputs: ``(..., 10)`` physical SI values divided by the nominal string."""
    if physical.shape[-1] != PARAMETER_COUNT:
        raise ValueError(f"Expected {PARAMETER_COUNT} physical parameters in the last axis.")
    return physical / nominal_parameters(physical.device)
