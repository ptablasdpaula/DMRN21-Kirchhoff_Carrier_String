"""Shared ten-coordinate physical inputs for the batch comparison.

Order: density, tension, radius, Young's modulus, length, amplitude, sigma0,
sigma1, pluck position, pickup position (the last two as fractions of L).
The batch is ``benchmarks.exact_methods.varied_material``: every coordinate
of the nominal string moved by up to +-3 %. All ten are independent
differentiable inputs; no f0/B chain rule is applied anywhere.
"""
import torch

from src.kc_string_forward import PARAMETER_COUNT, nominal_parameters

NAMES = ('density', 'tension', 'radius', 'young_modulus', 'length', 'amplitude', 'sigma_0', 'sigma_1',
         'pluck_position', 'pickup_position')


def physical_values(batch):
    from benchmarks.exact_methods import varied_material
    return varied_material(batch).detach()


def theta_from_physical(values):
    """Map independent physical SI coordinates to solver multipliers."""
    if values.ndim != 2 or values.shape[1] != PARAMETER_COUNT:
        raise ValueError(f'Expected physical values with shape (batch, {PARAMETER_COUNT})')
    return values / nominal_parameters(values.device)


def gradient_scales(device):
    return torch.clamp(nominal_parameters(device).abs(), min=1e-12)
