"""Batched box transforms, quasi-random starts and grid helpers for application fits."""
import math
import torch
from src.grid import GridConfig
from src.kc_string_forward import SPATIAL_ORDER


def decode_batch(box, logits):
    """``BoundedParameters.decode`` for ``(..., n)`` logits (the original slices along dim 0)."""
    low, high = box.bounds(logits.device)
    n = box.positive_count
    fraction = logits.sigmoid()
    log_low, log_high = low[:n].log(), high[:n].log()
    positive = (log_low + fraction[..., :n]*(log_high-log_low)).exp().clamp(low[:n], high[:n])
    linear = low[n:] + fraction[..., n:]*(high[n:]-low[n:])
    return torch.cat((positive, linear), dim=-1)


def encode_rows(box, values, margin=1e-6):
    """Inverse of ``decode_batch`` for ``(..., n)`` values; values are clamped into the box first."""
    low, high = box.bounds(values.device)
    n = box.positive_count
    values = values.clamp(low, high)
    fraction = torch.cat(((values[..., :n].log()-low[:n].log())/(high[:n].log()-low[:n].log()),
                          (values[..., n:]-low[n:])/(high[n:]-low[n:])), dim=-1)
    return torch.logit(fraction.clamp(margin, 1-margin))


def sample_logits(box, batch, seed=0, low=0.02, high=0.98, device=None):
    """Scrambled Sobol starts covering the box (log-uniform over the positive coordinates)."""
    n = len(box.names)
    engine = torch.quasirandom.SobolEngine(n, scramble=True, seed=seed)
    unit = engine.draw(batch, dtype=torch.float64).clamp(low, high)
    return torch.logit(unit).to(device) if device is not None else torch.logit(unit)


def largest_stable_intervals(box, oversampling, sample_rate=44100, cap=256, order=SPATIAL_ORDER, minimum=4):
    """Largest even interval count <= cap that passes the box-corner stability check, or None."""
    for intervals in range(cap - cap % 2, minimum-1, -2):
        try:
            GridConfig(box, sample_rate=sample_rate, oversampling=oversampling, intervals=intervals,
                       order=order, max_intervals=max(cap, 4096))
        except ValueError:
            continue
        return intervals
    return None


def corner_values(box):
    """All 2^n corners of the box as decoded values (n <= 12)."""
    n = len(box.names)
    if n > 12:
        raise ValueError('Too many coordinates for an exhaustive corner sweep.')
    signs = torch.tensor([[(i >> k) & 1 for k in range(n)] for i in range(2**n)], dtype=torch.float64)
    return decode_batch(box, torch.logit(signs.clamp(1e-9, 1-1e-9)))


def f0_range(box, to_material=None):
    """Tension-only fundamental range over the box corners (f0 is monotone in every coordinate)."""
    material = corner_values(box)
    if to_material is not None:
        material = to_material(material)
    rho, tension, radius, _, length = material[:, :5].unbind(dim=1)
    f0 = torch.sqrt(tension/(rho*math.pi*radius**2))/(2*length)
    return float(f0.min()), float(f0.max())
