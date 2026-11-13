"""Quasi-random starting points and frequency ranges of parameter boxes."""

import torch

from kc_string import physics


def sample_logits(box, batch, seed=0, low=0.02, high=0.98, device=None):
    """Scrambled Sobol starts covering the box (log-uniform over its positive coordinates)."""
    engine = torch.quasirandom.SobolEngine(len(box.names), scramble=True, seed=seed)
    unit = engine.draw(batch, dtype=torch.float64).clamp(low, high)
    return torch.logit(unit).to(device)


def f0_range(box):
    """Range of the tension-only fundamental over the box (f0 is monotone in every coordinate, so corners suffice)."""
    f0 = physics.string_quantities(box.to_material(box.corners()))["f0_hz"]
    return float(f0.min()), float(f0.max())
