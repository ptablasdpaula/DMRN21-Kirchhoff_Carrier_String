"""How much of the benchmark grid's error do higher-order spatial differences remove?

Same string, pluck, duration and metric as ``accuracy_study.py``. Russo et al.'s
time step is kept unchanged; only the spatial second-difference operator (and
hence the fourth difference, its square) is swapped:

* order 2: the reference three-point stencil (the benchmark scheme);
* order 4 and 6: the standard five- and seven-point central stencils, closed at
  the fixed ends by odd reflection (u_{-i} = -u_i), which is the simply
  supported condition the sine basis already assumes;
* ``spectral``: the exact sine-mode eigenvalues (not a finite difference; for
  reference only).

Every operator is diagonal in the same discrete sine basis, so the cost per
step, the rank-one solve and the adjoint structure are unchanged; only the
eigenvalues differ. The pure-PyTorch reference recurrence from
``src.kc_string_forward`` is used with its spatial operator patched, value
column only; nothing in ``src`` is modified.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('JAX_ENABLE_X64', 'true')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import torch

from benchmarks.accuracy_study import (BENCHMARK_INTERVALS, BENCHMARK_OVERSAMPLING, OUTPUT_RATE, modal_audio,
                                       modes_below_nyquist, relative, relative_spectrum, string_constants)
from src import kc_string_forward as forward
from src.kc_string_forward import nominal_parameters, raised_cosine_fraction
from src.synthesis import decimate

# Orders 2 and 4 are the production stencils (src.kc_string_forward.STENCILS);
# order 6 and the exact eigenvalues are studied here only.
STENCILS = {2: [1., -2., 1.],
            4: [-1/12, 4/3, -5/2, 4/3, -1/12],
            6: [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]}


def eigenvalue_factor(order):
    """Largest |eigenvalue| of h^2 D_xx relative to the three-point stencil's 4."""
    if order == 'spectral':
        return math.pi**2 / 4
    if order in forward.STENCILS:
        return forward.eigenvalue_factor(order)
    w = STENCILS[order]
    p = len(w) // 2
    return -sum(w[p + m] * math.cos(math.pi * m) for m in range(-p, p + 1)) / 4


def make_second_derivative(order, intervals):
    """Return a drop-in for ``kc_string_forward.second_derivative`` on the N-1 interior nodes.

    The returned function ignores the ``order`` argument the recurrence passes.
    """
    if order == 'spectral':
        index = torch.arange(1, intervals, dtype=torch.float64)
        transform = math.sqrt(2 / intervals) * torch.sin(math.pi * index[:, None] * index[None, :] / intervals)
        eigen = -(index * math.pi / intervals)**2   # eigenvalue of h^2 D_xx: -(n pi h / L)^2

        def second(values, inverse_spacing_squared, order=None):
            return forward.dual_multiply(transform @ (eigen[:, None] * (transform @ values)), inverse_spacing_squared)
        return second
    weights = STENCILS[order]
    p = len(weights) // 2

    def second(values, inverse_spacing_squared, order=None):
        zero = torch.zeros_like(values[:1])
        head = torch.cat((-values[:p-1].flip(0), zero)) if p > 1 else zero
        tail = torch.cat((zero, -values[-(p-1):].flip(0))) if p > 1 else zero
        padded = torch.cat((head, values, tail))
        n = values.shape[0]
        total = weights[0] * padded[:n]
        for m in range(1, len(weights)):
            total = total + weights[m] * padded[m:m+n]
        return forward.dual_multiply(total, inverse_spacing_squared)
    return second


def fdtd_audio(theta, intervals, oversampling, order, excitation, duration):
    """Decimated 44.1 kHz midpoint audio of Russo's step with the chosen spatial operator."""
    rate = OUTPUT_RATE * oversampling
    steps = round(duration * rate)
    original = forward.second_derivative, forward.dual_constant
    # Production orders go through the API; the others replace the operator.
    native_order = order if order in forward.STENCILS else forward.SPATIAL_ORDER
    if order not in forward.STENCILS:
        forward.second_derivative = make_second_derivative(order, intervals)
    try:
        states, coefficients, loss = forward.initialize(theta, intervals, rate, excitation, native_order)
        # Value column only: the recurrence's dual arithmetic is linear in the
        # column count, so a one-column constant keeps every operand one column.
        states, coefficients, loss = states[..., :1], coefficients[..., :1], loss[:1]
        forward.dual_constant = lambda value, device=None: torch.as_tensor(value, dtype=torch.float64, device=device).reshape(1)
        middle = (intervals - 1) // 2
        output = torch.empty(steps + 2, dtype=torch.float64)
        output[0], output[1] = states[0, middle, 0], states[1, middle, 0]
        with torch.no_grad():
            for i in range(steps):
                states, loss, sample = forward.advance(states, coefficients, loss, native_order)
                output[i + 2] = sample
                if i % 4096 == 0 and not bool(torch.isfinite(sample)):
                    return None
    finally:
        forward.second_derivative, forward.dual_constant = original
    if not bool(torch.isfinite(output).all()):
        return None
    return decimate(output, oversampling, round(duration * OUTPUT_RATE))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, default=ROOT/'benchmarks/accuracy_study_results/stencil-1s-nominal.json')
    p.add_argument('--duration', type=float, default=1.)
    p.add_argument('--cases', default='176:2:2,176:2:4,176:3:2,176:3:4,176:3:6,176:3:spectral,176:6:2,176:6:4,176:6:spectral,256:4:4',
                   help='intervals:oversampling:order, comma separated')
    a = p.parse_args(argv)
    torch.set_num_threads(1)
    material = nominal_parameters()
    theta = torch.ones(8, dtype=torch.float64)
    c, K, sigma1, length = string_constants(material)
    count = round(a.duration * OUTPUT_RATE)
    nyquist = modes_below_nyquist(material)

    start = time.perf_counter()
    raw = modal_audio(material, nyquist, OUTPUT_RATE, 'raised_cosine', a.duration)
    reference = decimate(raw, 1, count)
    print(f'modal reference ({nyquist} modes, band-matched): {time.perf_counter()-start:.1f} s', flush=True)

    rows = []
    for case in a.cases.split(','):
        n, of, order = case.split(':')
        n, of = int(n), int(of)
        order = order if order == 'spectral' else int(order)
        k = 1 / (OUTPUT_RATE * of)
        h = length / n
        factor = eigenvalue_factor(order)
        # Margin-free stability utilisation of the linearised update with this operator's largest eigenvalue.
        score = (c*c*k*k + 4*sigma1*k) * factor / h**2 + 4*K*K*k*k * factor**2 / h**4
        # The reference's node-based half-width is grid dependent; other grids pin the benchmark grid's physical width.
        excitation = 'raised_cosine' if n == BENCHMARK_INTERVALS else f'raised_cosine:{raised_cosine_fraction(BENCHMARK_INTERVALS)!r}'
        start = time.perf_counter()
        audio = fdtd_audio(theta, n, of, order, excitation, a.duration)
        seconds = time.perf_counter() - start
        row = {'intervals': n, 'oversampling': of, 'rate': OUTPUT_RATE*of, 'order': order, 'spacing': h, 'excitation': excitation,
               'eigenvalue_factor': factor, 'stability_score': score, 'seconds': seconds,
               'work_vs_benchmark': n*of/(BENCHMARK_INTERVALS*BENCHMARK_OVERSAMPLING)}
        if audio is None:
            row.update(status='unstable')
        else:
            row.update(status='ok', audio_relative_l2=relative(audio, reference),
                       fft_magnitude_relative_l2=relative_spectrum(audio, reference))
        rows.append(row)
        print(json.dumps(row), flush=True)

    result = {'metadata': {'duration_seconds': a.duration, 'parameters': 'nominal', 'excitation': 'raised_cosine',
                           'reference': f'modal {nyquist} modes, 44.1 kHz, band-matched (1x project filter)',
                           'time_step': "Russo et al.'s non-SAV reference step, unchanged",
                           'pure_torch_reference_recurrence': 'src.kc_string_forward.initialize/advance, value column'},
              'rows': rows}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = [f'# Spatial stencil study: nominal string, raised cosine, {a.duration:g} s', '',
             f'Reference: {result["metadata"]["reference"]}. Time step unchanged; only the spatial operator changes.', '',
             '| Intervals | Oversampling | Order | Work vs benchmark | Stability score | Waveform rel. L2 | FFT-mag. rel. L2 |',
             '|---:|---:|---|---:|---:|---:|---:|']
    for r in rows:
        err = f"{r['audio_relative_l2']:.3e} | {r['fft_magnitude_relative_l2']:.3e}" if r['status'] == 'ok' else 'unstable | —'
        lines.append(f"| {r['intervals']} | {r['oversampling']}x | {r['order']} | {r['work_vs_benchmark']:.2f} | "
                     f"{r['stability_score']:.3f} | {err} |")
    a.output.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print('Wrote', a.output)


if __name__ == '__main__':
    main()
