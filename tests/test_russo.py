"""Our scheme reproduces Russo et al.'s Octave code, and its gradients match finite differences of it."""

import pytest
import russo
import torch
from conftest import OCTAVE

from kc_string import physics, scheme

INTERVALS = 176
DURATION = 0.002
# Order 2 is the reference's own three-point scheme; order 4 is the paper's five-point stencil substituted into it.
CASES = [(2, "first_mode", 132300), (2, "raised_cosine", 132300), (4, "raised_cosine", 88200)]


@pytest.mark.octave
@pytest.mark.parametrize(("order", "excitation", "rate"), CASES)
@pytest.mark.parametrize("perturbed", [False, True], ids=["nominal", "perturbed"])
def test_waveform_matches_russo(order, excitation, rate, perturbed, perturbed_theta):
    theta = perturbed_theta if perturbed else torch.ones(physics.PARAMETER_COUNT, dtype=torch.float64)
    steps = round(DURATION * rate)
    physical = (theta * physics.nominal_parameters()).tolist()
    expected = russo.run(OCTAVE, physical, INTERVALS, rate, steps, excitation, order)
    audio, _ = scheme.reference(theta, INTERVALS, rate, steps, excitation, order)
    torch.testing.assert_close(audio, expected, rtol=1e-9, atol=1e-13)


@pytest.mark.octave
def test_gradient_matches_finite_differences_of_russo(perturbed_theta):
    """d/dtheta of mean(audio^2) / a^2 against central differences of the Octave output, all ten parameters."""
    order, excitation, rate = CASES[-1]
    steps = round(DURATION * rate)
    scale = physics.AMPLITUDE**2
    audio, jacobian = scheme.reference(perturbed_theta, INTERVALS, rate, steps, excitation, order)
    analytic = 2 * (audio[:, None] * jacobian).mean(dim=0) / scale
    finite_differences = []
    for i, name in enumerate(physics.PARAMETER_NAMES):
        # The damping gradients are small, so their steps are larger to stay above round-off.
        step = {"sigma_0": 1e-3, "sigma_1": 1e-2}.get(name, 1e-4)
        losses = []
        for sign in (-1, 1):
            theta = perturbed_theta.clone()
            theta[i] += sign * step
            physical = (theta * physics.nominal_parameters()).tolist()
            output = russo.run(OCTAVE, physical, INTERVALS, rate, steps, excitation, order)
            losses.append(float(output.square().mean()) / scale)
        finite_differences.append((losses[1] - losses[0]) / (2 * step))
    torch.testing.assert_close(analytic, torch.tensor(finite_differences, dtype=torch.float64), rtol=1e-3, atol=2e-9)
