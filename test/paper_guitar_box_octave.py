#!/usr/bin/env python3
"""Compare the high-rate guitar-domain box corner to the original KC script.

This is a short waveform validation, not a full-duration or timing benchmark.
The original submodule's actual non-SAV update and raised-cosine initialization
are executed in Octave with physical inputs and grid/time step fixed to match.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'test'))
import torch
import reference_paper_exact as paper
from experiments.scalar_cache.reconstruct import simulate
from src.grid import GridConfig
from src.kc_string_forward import ForwardString, nominal_parameters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('Requires a CUDA worker')
    octave = shutil.which('octave-cli') or shutil.which('octave')
    if not octave:
        raise RuntimeError('Requires Octave')
    paper.RATE, paper.INTERVALS = 44100 * 55, 176
    seconds, steps = .02, round(.02 * paper.RATE)
    length, tension, rho, f0, b = .1625, 70.3, 8050., 1318.4, .001984
    mu = tension / (2 * length * f0)**2
    EI = b * tension * length**2 / math.pi**2
    radius = math.sqrt(mu / (math.pi * rho))
    young = 4 * EI / (math.pi * radius**4)
    # Nominal centred pluck and midpoint readout: the Octave script's own node conventions.
    physical = torch.tensor([rho, tension, radius, young, length, .002, 1., .001, .5, .5], dtype=torch.float64)
    theta = physical / nominal_parameters()
    # Russo et al.'s Octave code is the three-point scheme, so this check pins order 2.
    score = float(GridConfig(order=2).physical_score(tension/mu, EI/mu, .001, length,
                                                     intervals=paper.INTERVALS, oversampling=55))
    if score >= 1:
        raise RuntimeError(f'Outside strict grid bound: {score}')
    with tempfile.TemporaryDirectory(prefix='kc-guitar-box-reference-') as tmp:
        reference, energy, reference_seconds = paper.run_octave(physical.tolist(), steps, octave,
                                                                  Path(tmp), 'raised_cosine', 2)
        model = ForwardString(paper.INTERVALS, paper.RATE, excitation='raised_cosine', order=2)
        with torch.no_grad():
            cpu = model(theta, seconds)
        outputs = {'cpu': cpu}
        for count in (1, 3):
            with torch.no_grad():
                outputs[f'cuda_{count}_scalar'] = simulate(theta.to('cuda'), paper.INTERVALS,
                    paper.RATE, steps, 'raised_cosine', 256, count, order=2).cpu()
        results = {}
        for name, audio in outputs.items():
            if not torch.isfinite(audio).all() or not torch.isfinite(reference).all():
                raise AssertionError(f'{name} or reference contains non-finite waveform')
            diff = audio - reference
            relative_l2 = float(diff.norm() / reference.norm())
            torch.testing.assert_close(audio, reference, rtol=1e-7, atol=1e-10)
            if relative_l2 > 1e-7:
                raise AssertionError(f'{name}: relative L2 error {relative_l2} exceeds 1e-7')
            results[name] = {'max_abs_waveform_error_m': float(diff.abs().max()),
                             'relative_l2_waveform_error': relative_l2}
        result = {'status': 'passed', 'seconds': seconds, 'steps': steps, 'intervals': paper.INTERVALS,
                  'sample_rate_hz': paper.RATE, 'oversampling': 55, 'branch': 'SAV=false',
                  'excitation': 'raised_cosine', 'output': 'raw midpoint, before filtering',
                  'physical_parameters': dict(zip(paper.PARAMETERS, physical.tolist())),
                  'case': {'f0_hz': f0, 'B': b, 'stability_score': score},
                  'gpu': torch.cuda.get_device_name(), 'reference_octave_seconds': reference_seconds,
                  'reference_energies': energy, 'comparisons': results,
                  'tolerance': {'waveform_rtol': 1e-7, 'waveform_atol_m': 1e-10, 'relative_l2_max': 1e-7}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
