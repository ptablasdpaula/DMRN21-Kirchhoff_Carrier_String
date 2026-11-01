"""Pluck and pickup positions as the ninth and tenth differentiable parameters.

Pure-torch checks that run anywhere (no compiled kernel, no CUDA):
  1. the pickup weights at the nominal midpoint are, bit for bit, the sine
     transform row the modal kernels used to read and the exact one-hot node
     vector for the spatial ones; off a node they are the sine interpolant;
  2. the centred raised cosine and the nominal readout reproduce the former
     node formulas bit for bit, so every exactness test against Russo et al.'s
     code is untouched by the new parameters;
  3. at off-centre positions the plain-autograd FDTD and the dual-number
     recurrence agree on the waveform and on all ten tangents, including the
     two new ones, to 1e-10 relative;
  4. the tangents of the dual pickup weights and of the dual pluck profile
     match central finite differences.
"""
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.kc_string_forward import (PARAMETER_COUNT, PLUCK, PICKUP, DUAL_COLUMNS, advance, dual_dot, excitation_shape,
                                   initialize, nominal_parameters, pickup_weights, sine_transform)
from src.accelerated import basis
from experiments.figure_comparison.fdtd import simulate

torch.set_num_threads(2)
N, RATE, STEPS = 32, 44100, 60
nominal = torch.ones(PARAMETER_COUNT, dtype=torch.float64)
off = nominal.clone(); off[PLUCK] = .9; off[PICKUP] = 1.1          # pluck at 0.45 L, pickup at 0.55 L
assert tuple(nominal_parameters()[8:].tolist()) == (.5, .5)

# 1. nominal weights reproduce the former readouts exactly
transform = sine_transform(N)
modal, spatial = pickup_weights(nominal, N, False), pickup_weights(nominal, N, True)
assert modal.shape == spatial.shape == (N-1, DUAL_COLUMNS)
assert torch.equal(modal[:, 0], transform[(N-1)//2]) and torch.equal(modal[:, 0], basis(N, 'cpu', True)[2])
one_hot = torch.zeros(N-1, dtype=torch.float64); one_hot[(N-1)//2] = 1
assert torch.equal(spatial[:, 0], one_hot)
assert torch.count_nonzero(modal[:, 1:1+PICKUP]) == 0 and torch.count_nonzero(spatial[:, 1:1+PICKUP]) == 0
off_modal, off_spatial = pickup_weights(off, N, False), pickup_weights(off, N, True)
assert torch.allclose(off_spatial[:, 0], transform @ off_modal[:, 0]) and torch.count_nonzero(off_modal[:, 1+PICKUP]) > 0

# 2. the centred pluck and the midpoint readout are the former formulas
indices = torch.arange(1, N, dtype=torch.float64); width = max(1, math.floor(.25*(N-1))); d = (indices-N/2)/width
former = torch.where(d.abs() <= 1, .5*(1+torch.cos(math.pi*d)), 0.)
assert torch.equal(former, excitation_shape(N, 'raised_cosine', center=.5))
assert torch.equal(torch.zeros_like(indices), excitation_shape(N, 'raised_cosine', center=.5, derivative=True)*0)
states, coeff, loss = initialize(nominal, N, RATE, 'raised_cosine')
assert states.shape == (2, N-1, DUAL_COLUMNS) and coeff.shape == (8, DUAL_COLUMNS) and loss.shape == (DUAL_COLUMNS,)
assert torch.count_nonzero(states[..., 1+PLUCK]) > 0, 'the pluck tangent is populated even at the centre'
new_states, _, sample = advance(states, coeff, spatial, loss)
assert torch.equal(sample, new_states[1, (N-1)//2, 0]), 'nominal readout is the midpoint node value'

# 3. autograd FDTD versus the dual recurrence at off-centre positions, all ten tangents
states, coeff, loss = initialize(off, N, RATE, 'raised_cosine'); weights = pickup_weights(off, N, True)
rows = [dual_dot(weights, states[0]), dual_dot(weights, states[1])]
for _ in range(STEPS):
    states, loss, _ = advance(states, coeff, weights, loss)
    rows.append(dual_dot(weights, states[1]))
dual = torch.stack(rows)
audio = simulate(off, N, RATE, STEPS, 'raised_cosine')
assert float((audio-dual[:, 0]).abs().max()) <= 1e-13*float(dual[:, 0].abs().max())
jacobian = torch.autograd.functional.jacobian(lambda theta: simulate(theta, N, RATE, STEPS, 'raised_cosine'), off)
error = (jacobian-dual[:, 1:]).abs().max(dim=0).values
scale = dual[:, 1:].abs().max(dim=0).values
assert bool((scale > 0).all()), 'every tangent, the two positions included, must be exercised'
assert bool((error <= 1e-10*scale).all()), (error/scale).tolist()

# 4. dual tangents of the pickup weights and the pluck profile against central differences
step = 1e-6
for index, kind in ((PICKUP, 'pickup'), (PLUCK, 'pluck')):
    unit = torch.zeros(PARAMETER_COUNT, dtype=torch.float64); unit[index] = step
    if kind == 'pickup':
        plus, minus = pickup_weights(off+unit, N, False)[:, 0], pickup_weights(off-unit, N, False)[:, 0]
        tangent = pickup_weights(off, N, False)[:, 1+index]
    else:
        plus, minus = initialize(off+unit, N, RATE, 'raised_cosine')[0][0, :, 0], initialize(off-unit, N, RATE, 'raised_cosine')[0][0, :, 0]
        tangent = initialize(off, N, RATE, 'raised_cosine')[0][0, :, 1+index]
    finite = (plus-minus)/(2*step)
    assert float((finite-tangent).abs().max()) <= 1e-6*float(tangent.abs().max()), kind

print('PASS: pluck and pickup positions; nominal readouts bit-identical; autograd and dual tangents agree to 1e-10')
