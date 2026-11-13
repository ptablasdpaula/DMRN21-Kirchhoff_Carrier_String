#!/usr/bin/env python3
"""Compare native fixed-grid KC against the *actual* submodule's non-SAV code.

Only configure physical inputs, damped branch, grid/time step and sample count;
all reference initialization, update and energy equations are left intact. Raw
internal-rate output is compared before the Renderer antialias filter. The
optional gradient test differentiates an independent Octave objective by
central finite differences, holding the grid and damping coefficients fixed.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.kc_string_forward import (ForwardString, nominal_parameters, initialize, advance, second_derivative,
                                   pickup_weights, PARAMETER_COUNT)
from src.grid import GridConfig

# Order 2 reproduces Russo et al.'s three-point scheme at the paper's original
# grid; order 4 substitutes the five-point operator into the same Octave code
# and runs at the benchmark's 2x rate.
RATES = {2: 132300, 4: 88200}
RATE = RATES[2]
INTERVALS = 176
FIVE_POINT_OCTAVE = ("e = ones(nPoints,1); Dxx = spdiags([-e 16*e -30*e 16*e -e], -2:2, nPoints, nPoints); "
                     "Dxx(1,1) = -29; Dxx(end,end) = -29; Dxx = Dxx/(12*h^2);")
PARAMETERS = ('rho', 'T0', 'radius', 'E', 'L', 'amplitude', 'sigma0', 'sigma1', 'pluck_position', 'pickup_position')
# Interpolated readout substituted into the Octave script when the pickup is
# not on a node: the sine interpolant of the interior nodes at x_o (fraction of L).
OCTAVE_PICKUP = ("nn = (1:nPoints)'; wOut = (2/N) * (sin(pi*nn*(1:nPoints)/N)' * sin(pi*nn*{xo:.17g}));")
SCRIPT = ROOT / 'nLinStringsConv-SAV/KCstring/KCStringSAV_noSplit.m'


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'Expected one occurrence of {old!r}; found {text.count(old)}')
    return text.replace(old, new, 1)


def run_octave(physical, steps, octave, temp, excitation='first_mode', order=2):
    text = SCRIPT.read_text()
    text = replace_once(text, 'SAV = true;', 'SAV = false;')
    text = replace_once(text, 'dampOn = false;', 'dampOn = true;')
    if excitation == 'raised_cosine':
        text = replace_once(text, 'initType = 1;', 'initType = 2;')
    if order == 4:
        # Five-point second difference closed by odd reflection; Dxxxx = Dxx.'*Dxx
        # below then equals its square, so startup, update, q and energies switch.
        text = replace_once(text, 'Dxx = Dxp*Dxm;', FIVE_POINT_OCTAVE)
    elif order != 2:
        raise ValueError('order must be 2 or 4')
    assignments = '\n'.join(f'{name} = {value:.17g};' for name, value in zip(PARAMETERS[:6], physical[:6]))
    text = replace_once(text, 'Area = pi*radius^2;', assignments + '\nArea = pi*radius^2;')
    text = replace_once(text, 'sigmaL = sigma0;',
                        f'sigma0 = {physical[6]:.17g}; sigma1 = {physical[7]:.17g};\nsigmaL = sigma0;')
    # Pluck centre and pickup as fractions of L. The script's node numbering is
    # 1..nPoints for x = j/N, so a fraction maps to the node coordinate x*N; the
    # raised-cosine loop accepts a non-integer centre as it stands. The readout
    # stays the script's own integer node whenever the pickup sits on one (the
    # nominal midpoint is node N/2), otherwise it becomes the sine interpolant.
    pluck_node, pickup_node = physical[8]*INTERVALS, physical[9]*INTERVALS
    text = replace_once(text, 'excitPos = outPos;', f'excitPos = {pluck_node:.17g};')
    if abs(pickup_node-round(pickup_node)) < 1e-9:
        text = replace_once(text, 'outPos = floor((nPoints + 1)/2);', f'outPos = {int(round(pickup_node))};')
    else:
        text = replace_once(text, 'outPos = floor((nPoints + 1)/2);',
                            'outPos = floor((nPoints + 1)/2);\n' + OCTAVE_PICKUP.format(xo=physical[9]))
        text = replace_once(text, 'Out(1) = uPrev(outPos);', "Out(1) = wOut.'*uPrev;")
        text = replace_once(text, 'Out(2) = u(outPos);', "Out(2) = wOut.'*u;")
        text = replace_once(text, 'Out(n+2)= uNext(outPos);', "Out(n+2)= wOut.'*uNext;")
    start = '% grid spacing\n'
    end = 'nPoints = N - 1;'
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    text = before + start + f'N = {INTERVALS}; h = L/N; k = 1/{RATE}; SR = {RATE}; timeSamples = {steps};\n\n' + end + after
    text = text.split('%%%%% Play & Plot', 1)[0]
    out = temp / 'reference.bin'
    # Binary output avoids text precision loss. The four diagnostics are the
    # first and last computed energy and the first two raw output samples.
    text += "\nfid=fopen('reference.bin','wb','ieee-le'); fwrite(fid,[Out(:); En(1); En(end); Kin(1); PotLin(1); PotNlin(1)],'double'); fclose(fid);\n"
    script = temp / 'run_reference.m'
    script.write_text(text)
    start_time = time.monotonic()
    result = subprocess.run([octave, '--no-gui', '--quiet', str(script)], cwd=temp,
                            capture_output=True, text=True, timeout=3400)
    if result.returncode:
        raise RuntimeError(f'Octave failed: {result.stdout}\n{result.stderr}')
    data = np.fromfile(out, dtype='<f8')
    if data.size != steps + 7:
        raise RuntimeError(f'Unexpected reference output size: {data.size}')
    return torch.from_numpy(data[:steps+2].copy()), data[steps+2:].tolist(), time.monotonic()-start_time


def first_energy(theta, excitation='first_mode', order=2):
    states, coeff, loss = initialize(theta, INTERVALS, RATE, excitation, order)
    advanced, _, _ = advance(states, coeff, pickup_weights(theta, INTERVALS, True), loss, order)
    prev, current = states[:, :, 0]
    following = advanced[1, :, 0]
    density, tension, radius, modulus, length, _, sigma0, sigma1 = (theta * nominal_parameters()).tolist()[:8]
    k = 1/RATE; h = length/INTERVALS; area = math.pi*radius**2
    mass = density*area; inertia = math.pi*radius**4/4
    inverse_h2 = torch.tensor([1/h**2], dtype=torch.float64)
    def dxx(x):
        return second_derivative(x[:, None], inverse_h2, order)[:, 0]
    dc = dxx(current); d4c = dxx(dc)
    kin = .5*mass*h*torch.dot(following-current, following-current)/k**2
    potlin = -.5*tension*h*torch.dot(following,dc)+.5*modulus*inertia*h*torch.dot(following,d4c)
    potnlin = modulus*area*h**2/(8*length)*(-torch.dot(following,dc))**2
    dd = .5*(following-prev)/k; dm = (current-prev)/k
    loss = 2*mass*h*(sigma1*torch.dot(dd,dxx(dm))-sigma0*torch.dot(dd,dd))
    return [float(kin+potlin+potnlin-loss*k), float(kin), float(potlin), float(potnlin)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('long','gradients'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--duration', type=float)
    p.add_argument('--excitation', choices=('first_mode', 'raised_cosine'), default='first_mode')
    p.add_argument('--order', type=int, choices=(2, 4), default=2,
                   help='2: Russo et al.\'s three-point scheme at 132.3 kHz; 4: five-point variant at 88.2 kHz')
    p.add_argument('--oversampling', type=int, help='Override the internal rate as 44100 x this factor')
    args = p.parse_args()
    global RATE
    RATE = 44100*args.oversampling if args.oversampling else RATES[args.order]
    octave = shutil.which('octave-cli') or shutil.which('octave')
    if not octave: raise RuntimeError('Octave is required')
    duration = args.duration if args.duration is not None else (4. if args.mode == 'long' else .002)
    steps = round(duration * RATE)
    if steps < 1: raise ValueError('At least one step is required')
    theta = torch.ones(PARAMETER_COUNT, dtype=torch.float64)
    if args.mode == 'gradients':
        # Off-centre pluck (0.45 L) and pickup (0.55 L): at the centre both
        # position derivatives vanish by symmetry and the check would be empty.
        theta *= torch.tensor([1.1,.9,1.05,.95,.9,1.1,.8,1.1,.9,1.1], dtype=torch.float64)
    physical = theta * nominal_parameters()
    grid = GridConfig(order=args.order)
    oversampling = RATE // 44100
    score = float(grid.physical_score(physical[1]/(physical[0]*math.pi*physical[2]**2),
                  physical[3]*physical[2]**2/(4*physical[0]), physical[7], physical[4],
                  intervals=INTERVALS, oversampling=oversampling))
    if score > 1: raise RuntimeError(f'Outside stability margin: {score}')
    with tempfile.TemporaryDirectory(prefix='kc-paper-') as tmp:
        tmp = Path(tmp)
        ref, energies, elapsed = run_octave(physical.tolist(), steps, octave, tmp, args.excitation, args.order)
        forward = ForwardString(INTERVALS, RATE, excitation=args.excitation, order=args.order)
        t = time.monotonic()
        if args.mode == 'gradients':
            audio, jac = forward.output_and_tangents(theta, duration)
        else:
            with torch.no_grad(): audio = forward(theta, duration)
        cpu_time = time.monotonic()-t
        diff = audio-ref
        result = dict(mode=args.mode, duration_s=duration, steps=steps, intervals=INTERVALS, spatial_order=args.order,
                      sample_rate=RATE, branch='SAV=false', excitation=args.excitation, precision='float64',
                      output='raw internal-rate pickup waveform, includes two startup samples',
                      physical_parameters=dict(zip(PARAMETERS, physical.tolist())), stability_score=score,
                      octave_seconds=elapsed, native_seconds_including_extension_load=cpu_time,
                      max_abs_error=float(diff.abs().max()), rms_error=float(diff.square().mean().sqrt()),
                      normalized_l2_error=float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(ref)),
                      reference_first_samples=ref[:4].tolist(), native_first_samples=audio[:4].tolist(),
                      reference_energies=dict(zip(('first_total','last_total','first_kinetic','first_linear_potential','first_nonlinear_potential'),energies)),
                      native_first_energies=dict(zip(('first_total','first_kinetic','first_linear_potential','first_nonlinear_potential'), first_energy(theta, args.excitation, args.order))))
        if not torch.isfinite(audio).all() or not torch.isfinite(ref).all(): raise RuntimeError('Nonfinite waveform')
        if not torch.allclose(audio, ref, rtol=1e-7, atol=1e-10):
            raise AssertionError(f'Waveforms disagree: {result}')
        if args.mode == 'long':
            if not torch.cuda.is_available():
                raise RuntimeError('Long check requires CUDA for direct cached-versus-Octave comparison')
            from experiments.scalar_cache.reconstruct import simulate as cached
            result['cached_cuda'] = {}
            for cache_count in (1, 3):
                with torch.no_grad():
                    raw = cached(theta.to('cuda'), INTERVALS, RATE, steps, args.excitation, 256, cache_count, order=args.order).cpu()
                torch.testing.assert_close(raw, ref, rtol=1e-7, atol=1e-10)
                result['cached_cuda'][str(cache_count)] = {
                    'max_abs_audio_error_vs_octave': float((raw-ref).abs().max()),
                    'relative_l2_audio_error_vs_octave': float(torch.linalg.vector_norm(raw-ref)/torch.linalg.vector_norm(ref)),
                }
        if args.mode == 'gradients':
            # Loss normalization is fixed; derivatives are w.r.t. multipliers,
            # which span every physical input, including density and length.
            amplitude_scale = float(nominal_parameters()[5])
            analytic = 2*(audio[:, None]*jac).mean(dim=0)/amplitude_scale**2
            gradrows = []
            for i, name in enumerate(PARAMETERS):
                vals = []
                # Small damping gradients are sensitive to waveform round-off:
                # perturb their dimensionless multipliers more strongly.
                eps = 1e-2 if name == 'sigma1' else (1e-3 if name == 'sigma0' else 1e-4)
                # The positions are smooth in the multiplier; the same 1e-4 step applies.
                for sign in (-1,1):
                    shifted = theta.clone(); shifted[i] += sign*eps
                    y, _, _ = run_octave((shifted*nominal_parameters()).tolist(), steps, octave, tmp, args.excitation, args.order)
                    vals.append(float(y.square().mean()/amplitude_scale**2))
                fd = (vals[1]-vals[0])/(2*eps)
                gradrows.append(dict(parameter=name, step=eps, analytic=float(analytic[i]), octave_finite_difference=fd,
                                     abs_error=abs(float(analytic[i])-fd), relative_error=abs(float(analytic[i])-fd)/max(abs(fd), 1e-12)))
            result['gradients'] = gradrows
            # The Octave objective is finite-differenced, so use a mixed
            # tolerance rather than promising bitwise gradient equality.
            finite_diff = torch.tensor([r['octave_finite_difference'] for r in gradrows], dtype=torch.float64)
            torch.testing.assert_close(analytic, finite_diff, rtol=1e-3, atol=2e-9)
            result['gradient_tolerance'] = {'rtol': 1e-3, 'atol': 2e-9,
                                            'note': 'relative-multiplier gradient of normalized mean-square loss'}
            if torch.cuda.is_available():
                from experiments.scalar_cache.reconstruct import simulate as cached
                result['cached_cuda'] = {}
                for cache_count in (1, 3):
                    x = theta.to('cuda').detach().requires_grad_()
                    raw = cached(x, INTERVALS, RATE, steps, args.excitation, 256, cache_count, order=args.order)
                    loss = raw.square().mean()/amplitude_scale**2
                    loss.backward()
                    gpu_audio, gpu_grad = raw.detach().cpu(), x.grad.detach().cpu()
                    torch.testing.assert_close(gpu_audio, ref, rtol=1e-7, atol=1e-10)
                    torch.testing.assert_close(gpu_grad, finite_diff, rtol=1e-3, atol=2e-9)
                    result['cached_cuda'][str(cache_count)] = {
                        'max_abs_audio_error_vs_octave': float((gpu_audio-ref).abs().max()),
                        'relative_l2_audio_error_vs_octave': float(torch.linalg.vector_norm(gpu_audio-ref)/torch.linalg.vector_norm(ref)),
                        'gradients': gpu_grad.tolist(),
                        'max_abs_gradient_error_vs_octave_finite_difference': float((gpu_grad-finite_diff).abs().max()),
                    }
            else:
                raise RuntimeError('Gradient check also requires CUDA for cached-versus-Octave comparison')
            # Additional deliberately different guitar-like material inputs,
            # same damped non-SAV branch and the exact same fixed grid.
            cases = {
                'steel': [8050., 75., .000356, 174e9, .65, .001, 1., .0001, .5, .5],
                'nylon': [1150., 75., .0005, 3e9, .65, .001, 1., .0001, .5, .5],
            }
            result['additional_material_cases'] = {}
            for name, material in cases.items():
                v = torch.tensor(material, dtype=torch.float64)
                c2, K2, s1, L = v[1]/(v[0]*math.pi*v[2]**2), v[3]*v[2]**2/(4*v[0]), v[7], v[4]
                # These strings are shorter and stiffer than the nominal one; keep the
                # grid and raise the rate until the margin-adjusted bound holds (the
                # five-point stencil needs more than the base rate for them).
                case_oversampling = oversampling
                while grid.physical_score(c2, K2, s1, L, intervals=INTERVALS, oversampling=case_oversampling) > 1:
                    case_oversampling += 1
                    if case_oversampling > 64:
                        raise RuntimeError(f'{name} case outside stability margin at every rate up to 64x')
                case_score = float(grid.physical_score(c2, K2, s1, L, intervals=INTERVALS, oversampling=case_oversampling))
                case_rate = 44100*case_oversampling
                case_steps = round(duration*case_rate)
                RATE = case_rate
                case_ref, _, _ = run_octave(material, case_steps, octave, tmp, args.excitation, args.order)
                RATE = 44100*oversampling
                case_model = ForwardString(INTERVALS, case_rate, excitation=args.excitation, order=args.order)
                with torch.no_grad(): case_actual = case_model(v/nominal_parameters(), duration)
                torch.testing.assert_close(case_actual, case_ref, rtol=1e-7, atol=1e-10)
                result['additional_material_cases'][name] = {
                    'physical_parameters': dict(zip(PARAMETERS, material)),
                    'oversampling': case_oversampling, 'sample_rate': case_rate,
                    'stability_score': case_score,
                    'max_abs_audio_error_vs_octave': float((case_actual-case_ref).abs().max()),
                    'relative_l2_audio_error_vs_octave': float(torch.linalg.vector_norm(case_actual-case_ref)/torch.linalg.vector_norm(case_ref)),
                }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__': main()
