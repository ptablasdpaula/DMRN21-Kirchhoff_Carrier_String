"""The JAX twin of the spatial PyTorch baseline reproduces it: waveform and all ten gradients.

Run with the JAX environment (it also carries PyTorch); CPU is enough:
    JAX_PLATFORMS=cpu python test/figure_fdtd_jax.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('JAX_ENABLE_X64', 'true')
import jax
import jax.numpy as jnp
import numpy as np
import torch

from experiments.figure_comparison.fdtd import simulate
from experiments.figure_comparison.fdtd_jax import renderer
from experiments.figure_comparison.physical import physical_values, theta_from_physical


def compare(intervals, excitation, order, steps, checkpoint, unroll, basis='grid', rate=132300, batch=3):
    values = physical_values(batch).double()
    weight = torch.randn(batch, steps + 2, dtype=torch.float64, generator=torch.Generator().manual_seed(749))
    x = values.clone().requires_grad_()
    expected = simulate(theta_from_physical(x), intervals, rate, steps, excitation, order, checkpoint_steps=checkpoint)
    expected_grad, = torch.autograd.grad((expected * weight).sum(), x)

    w = jnp.asarray(weight.numpy())

    def objective(v):
        audio = renderer(v, steps, intervals, rate, excitation, order, checkpoint, unroll, basis)
        return jnp.sum(audio * w), audio

    (_, actual), actual_grad = jax.jit(jax.value_and_grad(objective, has_aux=True))(jnp.asarray(values.numpy()))
    np.testing.assert_allclose(np.asarray(actual), expected.detach().numpy(), rtol=1e-10, atol=1e-13)
    scale = values.numpy()
    np.testing.assert_allclose(np.asarray(actual_grad) * scale, expected_grad.numpy() * scale, rtol=5e-9, atol=1e-12)


def test_matches_pytorch():
    for intervals, excitation, order in ((8, 'first_mode', 2), (16, 'raised_cosine', 4), (176, 'multimode', 4),
                                         (176, 'raised_cosine', 4), (24, 'raised_cosine:0.2', 4)):
        # No steps; one plain scan; whole segments plus a remainder; a
        # remainder only; segments that divide the run exactly.
        for steps, checkpoint, unroll in ((0, None, 1), (36, None, 8), (300, 128, 8), (77, 256, 1), (128, 64, 4)):
            for basis in BASES:
                compare(intervals, excitation, order, steps, checkpoint, unroll, basis)
                print('ok', intervals, excitation, order, steps, checkpoint, unroll, basis, flush=True)


BASES = tuple(b for b in ('grid', 'sine') if f'--{b}' in sys.argv) or ('grid', 'sine')


if __name__ == '__main__':
    test_matches_pytorch()
    print('JAX FDTD twin matches the PyTorch baseline (' + ', '.join(BASES) + ')')
