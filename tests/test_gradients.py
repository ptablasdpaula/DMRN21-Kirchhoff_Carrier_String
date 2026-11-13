"""Every implementation in Table 1 returns the reference scheme's audio and gradients.

The reference is ``scheme.reference``: the scheme stepped in Python with exact
forward-mode tangents (itself checked against Russo et al.'s code in
``test_russo.py``). Each implementation's gradient of a random linear functional
of the audio is compared with the reference Jacobian's, in multiplier units.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from kc_string import adjoint, jax_scheme, physics, scheme, workload

STEPS = 300
BATCH = 3


@pytest.fixture(scope="module")
def case():
    """A workload batch, a random weighting of its audio, and the reference audio and gradient."""
    values = workload.batch(BATCH)
    theta = physics.multipliers(values)
    rate = workload.GRID.integration_rate
    weights = torch.randn(BATCH, STEPS + 2, dtype=torch.float64, generator=torch.Generator().manual_seed(749))
    audio, gradient = [], []
    for row, weight in zip(theta, weights, strict=True):
        samples, jacobian = scheme.reference(row, workload.GRID.intervals, rate, STEPS, workload.EXCITATION)
        audio.append(samples)
        gradient.append(weight @ jacobian)
    return values, weights, torch.stack(audio), torch.stack(gradient)


@pytest.mark.parametrize(
    ("basis", "checkpoint", "unroll"),
    [("grid", None, 8), ("grid", 64, 4), ("sine", 64, 4), ("sine", 128, 1)],
    ids=["auto", "checkpoints", "modes", "modes-remainder"],
)
def test_jax_autodiff_matches_reference(case, basis, checkpoint, unroll):
    values, weights, expected_audio, expected_gradient = case
    rate = workload.GRID.integration_rate

    def objective(physical):
        audio = jax_scheme.render(
            physical,
            STEPS,
            workload.GRID.intervals,
            rate,
            workload.EXCITATION,
            checkpoint=checkpoint,
            unroll=unroll,
            basis=basis,
        )
        return jnp.sum(audio * jnp.asarray(weights.numpy())), audio

    (_, audio), gradient = jax.jit(jax.value_and_grad(objective, has_aux=True))(jnp.asarray(values.numpy()))
    np.testing.assert_allclose(np.asarray(audio), expected_audio.numpy(), rtol=1e-10, atol=1e-14)
    # d/d physical times the nominal value is d/d multiplier.
    nominal = physics.nominal_parameters().numpy()
    np.testing.assert_allclose(np.asarray(gradient) * nominal, expected_gradient.numpy(), rtol=1e-8, atol=1e-12)


@pytest.mark.cuda
@pytest.mark.parametrize("checkpoint", [64, 256], ids=["segments-and-remainder", "one-segment"])
def test_hand_written_adjoint_matches_reference(case, checkpoint):
    values, weights, expected_audio, expected_gradient = case
    theta = physics.multipliers(values).cuda().requires_grad_()
    audio = adjoint.simulate(
        theta, workload.GRID.intervals, workload.GRID.integration_rate, STEPS, workload.EXCITATION, checkpoint
    )
    (audio * weights.cuda()).sum().backward()
    torch.testing.assert_close(audio.detach().cpu(), expected_audio, rtol=1e-10, atol=1e-14)
    torch.testing.assert_close(theta.grad.cpu(), expected_gradient, rtol=1e-8, atol=1e-12)
