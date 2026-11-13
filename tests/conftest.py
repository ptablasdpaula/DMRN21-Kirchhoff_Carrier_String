"""Shared test settings: skip GPU and Octave tests where they cannot run."""

import shutil

import pytest
import torch

OCTAVE = shutil.which("octave-cli") or shutil.which("octave")


def pytest_collection_modifyitems(items):
    for item in items:
        if "cuda" in item.keywords and not torch.cuda.is_available():
            item.add_marker(pytest.mark.skip(reason="needs a CUDA GPU"))
        if "octave" in item.keywords and OCTAVE is None:
            item.add_marker(pytest.mark.skip(reason="needs Octave"))


@pytest.fixture
def perturbed_theta():
    """Every multiplier moved off 1, with the pluck at 0.45 L and the pickup at 0.55 L.

    Off the centre, the position derivatives do not vanish by symmetry.
    """
    return torch.tensor([1.1, 0.9, 1.05, 0.95, 0.9, 1.1, 0.8, 1.1, 0.9, 1.1], dtype=torch.float64)
