"""Accuracy of the benchmarked FDTD grid and of modal truncation against a grid-converged reference.

One string (the nominal one, or the first benchmark item), the raised-cosine
pluck of the benchmark, one second, and every signal at the 44.1 kHz output
rate:

* FDTD on the benchmark grid (176 intervals, 3x oversampling) and along
  Russo et al.'s refinement path, h = 1.01 h_min(k) with k = 1/(44100 OF) and
  N = 2 floor(L / 2h). The paper's code recomputes k so that h sits exactly on
  the margin; here the rate stays an integer multiple of 44.1 kHz, as in
  ``src/grid.py``, so the project's strided decimation filter applies. The
  finest run is the reference. The physical half-width of the pluck is held
  fixed at the benchmark grid's value on every grid.
* Modal (jaxdiffmodal ``solve_tf``, continuous eigenvalues) with the benchmark
  mode count and with every mode below the output Nyquist, at 44.1 kHz (raw,
  as benchmarked, and band-matched through the project's 1x filter) and at
  132.3 kHz decimated exactly like the FDTD.

Outputs a JSON record and a markdown table. Nothing here is timed for the
benchmark; it only measures discretisation differences.
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

import numpy as np
import torch

from benchmarks.exact_methods import GRID
from src import kc_string as reference
from src.grid import GridConfig
from src.kc_string_forward import SPATIAL_ORDER, ForwardString, eigenvalue_factor, nominal_parameters, raised_cosine_fraction
from src.synthesis import decimate

OUTPUT_RATE = 44100
BENCHMARK_INTERVALS = GRID.intervals
BENCHMARK_OVERSAMPLING = GRID.oversampling
BENCHMARK_ORDER = GRID.order
BENCHMARK_NAME = f'fdtd-{BENCHMARK_INTERVALS}-{BENCHMARK_OVERSAMPLING}x'


def string_constants(material):
    """(c, K, sigma_1, L) from the eight material coordinates."""
    rho, tension, radius, modulus, length, _, _, sigma1 = [float(v) for v in material][:8]  # positions do not enter the constants
    c = math.sqrt(tension / (rho * math.pi * radius**2))
    K = math.sqrt(modulus * radius**2 / (4 * rho))
    return c, K, sigma1, length


def refinement(oversampling, c, K, sigma1, length, order=SPATIAL_ORDER):
    """(intervals, spacing, time step) on Russo et al.'s path at a fixed integer rate.

    The five-point stencil's eigenvalue factor tightens the spacing bound by its
    square root, so the fourth-order path uses fewer intervals at each rate.
    """
    k = 1 / (OUTPUT_RATE * oversampling)
    h = 1.01 * math.sqrt(eigenvalue_factor(order)) * reference.minimum_grid_spacing(k, c, K, sigma1)
    intervals = 2 * math.floor(length / h / 2)
    return intervals, length / intervals, k


def modes_below_nyquist(material, rate=OUTPUT_RATE):
    """Largest n with n f0 sqrt(1 + B n^2) <= rate/2 for the continuous stiff string."""
    rho, tension, radius, modulus, length, _, _, _ = [float(v) for v in material][:8]
    c, K, _, _ = string_constants(material)
    f0 = c / (2 * length)
    B = (math.pi * K / (c * length))**2
    n = 0
    while (n + 1) * f0 * math.sqrt(1 + B * (n + 1)**2) <= rate / 2:
        n += 1
    return n


def fdtd_audio(theta, intervals, oversampling, excitation, duration, device='cpu', backend='native', order=SPATIAL_ORDER):
    """Decimated 44.1 kHz midpoint audio of the scheme on one grid.

    ``native`` is the benchmark target's own CPU path (it also carries the
    eight tangent columns, which are discarded here); ``spatial`` is the
    fused primal-only kernel when its extension can be built.
    """
    rate = OUTPUT_RATE * oversampling
    model = ForwardString(intervals, rate, excitation=excitation, backend=backend, order=order)
    with torch.no_grad():
        raw = model(theta.to(device), duration)
    return decimate(raw, oversampling, round(duration * OUTPUT_RATE)).cpu()


def modal_audio(material, modes, rate, excitation, duration):
    """Internal-rate midpoint audio of the upstream modal solver, as a CPU tensor."""
    import jax.numpy as jnp
    from experiments.native_rate_study.renderer import renderer
    samples = round(duration * rate) + 2
    x = jnp.asarray(np.asarray(material, dtype=np.float64)[None])
    raw = renderer(x, samples, rate, modes, BENCHMARK_INTERVALS, excitation)[0]
    return torch.from_numpy(np.array(raw, dtype=np.float64))


def relative(a, b):
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(b))


def relative_spectrum(a, b):
    return relative(torch.fft.rfft(a).abs(), torch.fft.rfft(b).abs())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, help='Default: accuracy_study_results/raised-cosine-1s-<parameters>-order<order>.json')
    p.add_argument('--duration', type=float, default=1.)
    p.add_argument('--oversamplings', default='3,6,12,24,48', help='Refinement path; the last is the reference')
    p.add_argument('--modal-modes', default='100', help='Benchmark modal counts; the sub-Nyquist count is added')
    p.add_argument('--modal-rates', default=f'44100,{GRID.integration_rate}')
    p.add_argument('--parameters', choices=('nominal', 'batch0'), default='nominal')
    p.add_argument('--device', default='cpu', help='Device for the FDTD runs; the adjoint kernels are not used')
    p.add_argument('--backend', choices=('native', 'spatial', 'modal'), default='native')
    p.add_argument('--order', type=int, choices=(2, 4), default=BENCHMARK_ORDER, help='Spatial stencil order of every FDTD run')
    p.add_argument('--skip-modal', action='store_true')
    a = p.parse_args(argv)
    if a.output is None:
        a.output = ROOT/f'benchmarks/accuracy_study_results/raised-cosine-{a.duration:g}s-{a.parameters}-order{a.order}.json'

    nominal = nominal_parameters()
    if a.parameters == 'nominal':
        material = nominal.clone()
    else:
        from experiments.figure_comparison.physical import physical_values
        material = physical_values(1)[0]
    theta = material / nominal
    c, K, sigma1, length = string_constants(material)
    count = round(a.duration * OUTPUT_RATE)
    fraction = raised_cosine_fraction(BENCHMARK_INTERVALS)
    fixed_width = f'raised_cosine:{fraction!r}'
    oversamplings = [int(x) for x in a.oversamplings.split(',')]
    grid = GridConfig(order=a.order)

    signals, configurations = {}, []

    def record(name, audio, seconds, **info):
        if audio.shape != (count,) or not bool(torch.isfinite(audio).all()):
            raise RuntimeError(f'{name}: unexpected shape {tuple(audio.shape)} or non-finite output')
        signals[name] = audio
        configurations.append({'name': name, 'seconds': seconds, **info})
        print(f'{name}: {seconds:.1f} s', flush=True)

    start = time.perf_counter()
    audio = fdtd_audio(theta, BENCHMARK_INTERVALS, BENCHMARK_OVERSAMPLING, 'raised_cosine', a.duration, a.device, a.backend, a.order)
    record(BENCHMARK_NAME, audio, time.perf_counter()-start, solver='fdtd', intervals=BENCHMARK_INTERVALS,
           oversampling=BENCHMARK_OVERSAMPLING, rate=OUTPUT_RATE*BENCHMARK_OVERSAMPLING, order=a.order,
           spacing=length/BENCHMARK_INTERVALS, excitation='raised_cosine',
           stability_score=grid.physical_score(c*c, K*K, sigma1, length, intervals=BENCHMARK_INTERVALS,
                                               oversampling=BENCHMARK_OVERSAMPLING),
           note='benchmark grid (benchmarks.exact_methods.GRID)')
    path = []
    for oversampling in oversamplings:
        intervals, spacing, k = refinement(oversampling, c, K, sigma1, length, a.order)
        score = grid.physical_score(c*c, K*K, sigma1, length, intervals=intervals, oversampling=oversampling)
        if score > 1:
            raise RuntimeError(f'{intervals} intervals at {oversampling}x violates the margin ({score})')
        name = f'fdtd-{intervals}-{oversampling}x'
        start = time.perf_counter()
        audio = fdtd_audio(theta, intervals, oversampling, fixed_width, a.duration, a.device, a.backend, a.order)
        record(name, audio, time.perf_counter()-start, solver='fdtd', intervals=intervals, oversampling=oversampling,
               rate=OUTPUT_RATE*oversampling, order=a.order, spacing=spacing, excitation=fixed_width, stability_score=score,
               russo_time_step=reference.stable_time_step(spacing, c, K, sigma1),
               note='h = 1.01 sqrt(gamma) h_min(k) at the fixed integer rate; Russo et al. would recompute k from h')
        path.append(name)
    ref = path[-1]

    nyquist_modes = modes_below_nyquist(material)
    modal_counts = sorted({int(x) for x in a.modal_modes.split(',')} | {nyquist_modes})
    modal_rates = [int(x) for x in a.modal_rates.split(',')]
    if not a.skip_modal:
        for modes in modal_counts:
            for rate in modal_rates:
                if rate % OUTPUT_RATE:
                    raise ValueError('Modal rates must be integer multiples of 44.1 kHz')
                factor = rate // OUTPUT_RATE
                start = time.perf_counter()
                raw = modal_audio(material, modes, rate, 'raised_cosine', a.duration)
                seconds = time.perf_counter()-start
                info = dict(solver='modal', modes=modes, rate=rate, excitation='raised_cosine',
                            integrator='jaxdiffmodal.time_integrators.solve_tf')
                if factor == 1:
                    record(f'modal-{modes}-44.1k-raw', raw[:count], seconds, filtered=False,
                           note='as benchmarked: no filter at the output rate', **info)
                    record(f'modal-{modes}-44.1k-band', decimate(raw, 1, count), 0., filtered=True,
                           note='same signal through the 1x project filter (same cutoff and width as the FDTD decimation)', **info)
                else:
                    record(f'modal-{modes}-{rate/1000:g}k', decimate(raw, factor, count), seconds, filtered=True,
                           note='decimated with the FDTD anti-alias filter', **info)

    def compare(candidate, target, label):
        return {'candidate': candidate, 'reference': target, 'label': label,
                'audio_relative_l2': relative(signals[candidate], signals[target]),
                'fft_magnitude_relative_l2': relative_spectrum(signals[candidate], signals[target])}

    comparisons = [compare(BENCHMARK_NAME, ref, 'benchmark FDTD grid vs grid-converged FDTD')]
    comparisons += [compare(name, ref, 'refinement path vs finest') for name in path[:-1]]
    comparisons += [compare(path[i], path[i+1], 'successive refinements') for i in range(len(path)-1)]
    modal_names = [n for n in signals if n.startswith('modal')]
    comparisons += [compare(name, ref, 'modal vs grid-converged FDTD') for name in modal_names]
    comparisons += [compare(name, BENCHMARK_NAME, 'modal vs benchmark FDTD grid (the sweep diagnostic uses the raw 44.1k signal)')
                    for name in modal_names]
    for modes in modal_counts:
        other = [m for m in modal_counts if m != modes]
        for m in other:
            if f'modal-{modes}-44.1k-raw' in signals and f'modal-{m}-44.1k-raw' in signals and m > modes:
                comparisons.append(compare(f'modal-{modes}-44.1k-raw', f'modal-{m}-44.1k-raw', 'modal truncation'))
        for rate in modal_rates:
            if rate != OUTPUT_RATE and f'modal-{modes}-{rate/1000:g}k' in signals and f'modal-{modes}-44.1k-band' in signals:
                comparisons.append(compare(f'modal-{modes}-44.1k-band', f'modal-{modes}-{rate/1000:g}k', 'modal time stepping'))
    # Modal against every grid: if the modal solution is the continuum limit, these
    # errors fall like h^2 all the way down, at the same rate as the successive
    # FDTD differences.
    modal_reference = f'modal-{nyquist_modes}-44.1k-band' if f'modal-{nyquist_modes}-44.1k-band' in signals else None
    levels = [BENCHMARK_NAME] + path
    if modal_reference:
        comparisons += [compare(name, modal_reference, 'FDTD grid vs sub-Nyquist modal (band-matched)') for name in levels]
    spacing = {conf['name']: conf['spacing'] for conf in configurations if conf['solver'] == 'fdtd'}
    ratios = []
    successive = [c for c in comparisons if c['label'] == 'successive refinements']
    # Spatial error falls as h^order; on this path (k halves per level, h shrinks
    # by about sqrt 2) that is about 2 per level at order 2 and 4 at order 4, the
    # latter matching the second-order time step's own (k_c/k_f)^2 = 4.
    for first, second in zip(successive, successive[1:]):
        ratios.append({'coarse': first['candidate'], 'fine': second['candidate'],
                       'audio_error_ratio': first['audio_relative_l2']/second['audio_relative_l2'],
                       'spacing_power_ratio': (spacing[first['candidate']]/spacing[second['candidate']])**a.order})
    modal_errors = [c for c in comparisons if c['label'].startswith('FDTD grid vs sub-Nyquist modal')]
    for first, second in zip(modal_errors, modal_errors[1:]):
        ratios.append({'coarse': first['candidate'], 'fine': second['candidate'], 'against': modal_reference,
                       'audio_error_ratio': first['audio_relative_l2']/second['audio_relative_l2'],
                       'spacing_power_ratio': (spacing[first['candidate']]/spacing[second['candidate']])**a.order})
    # Richardson-style estimate of the finest grid's own distance to the limit from
    # its last increment: e_N = d_N/(r-1) with r = (h_{N-1}/h_N)^order.
    last = successive[-1] if successive else None
    estimate = None
    if last:
        r = (spacing[last['candidate']]/spacing[last['reference']])**a.order
        estimate = {'grid': ref, 'last_increment': last['audio_relative_l2'], 'spacing_power_ratio': r,
                    'estimated_error_vs_limit': last['audio_relative_l2']/(r-1)}

    result = {'metadata': {'duration_seconds': a.duration, 'output_rate': OUTPUT_RATE, 'parameters': a.parameters,
                           'spatial_order': a.order, 'benchmark_grid': BENCHMARK_NAME,
                           'material': [float(v) for v in material], 'wave_speed': c, 'stiffness': K,
                           'sigma_1': sigma1, 'length': length, 'f0': c/(2*length),
                           'inharmonicity_B': (math.pi*K/(c*length))**2,
                           'raised_cosine_half_width_fraction': fraction, 'modes_below_output_nyquist': nyquist_modes,
                           'reference': ref, 'pickup': 'midpoint (even modes vanish); the pluck is symmetric',
                           'device': a.device, 'fdtd_backend': a.backend, 'torch_version': torch.__version__},
              'configurations': configurations, 'comparisons': comparisons, 'convergence_ratios': ratios,
              'finest_grid_error_estimate': estimate}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = [f'# Accuracy study: {a.parameters} string, raised cosine, {a.duration:g} s, outputs at 44.1 kHz, spatial order {a.order}', '',
             f'Reference: `{ref}` (h = 1.01 sqrt(gamma) h_min on Russo et al.\'s path, gamma = {eigenvalue_factor(a.order):g}). '
             f'f0 = {c/(2*length):.2f} Hz, B = {(math.pi*K/(c*length))**2:.3e}; {nyquist_modes} continuous modes lie below 22.05 kHz.', '',
             '| Candidate | Reference | Waveform rel. L2 | FFT-magnitude rel. L2 | Note |', '|---|---|---:|---:|---|']
    for comp in comparisons:
        lines.append(f"| {comp['candidate']} | {comp['reference']} | {comp['audio_relative_l2']:.3e} | "
                     f"{comp['fft_magnitude_relative_l2']:.3e} | {comp['label']} |")
    if ratios:
        lines += ['', f'Error ratios between refinement levels; order {a.order} in space predicts the (h_coarse/h_fine)^{a.order} '
                  'column for the spatial part (the second-order time step contributes (k_c/k_f)^2 = 4 per level):', '',
                  f'| Coarse | Fine | Against | Error ratio | (h_c/h_f)^{a.order} |', '|---|---|---|---:|---:|']
        lines += [f"| {r['coarse']} | {r['fine']} | {r.get('against', 'next level')} | {r['audio_error_ratio']:.2f} | "
                  f"{r['spacing_power_ratio']:.2f} |" for r in ratios]
    if estimate:
        lines += ['', f"Finest grid `{estimate['grid']}`: last increment {estimate['last_increment']:.3e}, "
                  f"so its estimated distance to the limit is {estimate['estimated_error_vs_limit']:.3e} "
                  f"(Richardson with r = {estimate['spacing_power_ratio']:.2f})."]
    lines += ['', '| Configuration | Seconds | Details |', '|---|---:|---|']
    for conf in configurations:
        detail = {k: v for k, v in conf.items() if k not in ('name', 'seconds', 'note')}
        lines.append(f"| {conf['name']} | {conf['seconds']:.1f} | {json.dumps(detail)} |")
    a.output.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print('Wrote', a.output)


if __name__ == '__main__':
    main()
