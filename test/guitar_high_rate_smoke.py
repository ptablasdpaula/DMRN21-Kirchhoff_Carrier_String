"""Short nonlinear CPU/CUDA checks on an illustrative fretted guitar family.

Run on a GPU worker; this is not an exhaustive proof or a timing benchmark.
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from experiments.scalar_cache.reconstruct import simulate
from src.grid import GridConfig
from src.kc_string_forward import ForwardString, nominal_parameters


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--oversampling', type=int, default=34)
    args = p.parse_args()
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    rate, n, seconds = 44_100 * args.oversampling, 176, 0.02
    steps = round(rate * seconds)
    # Approximate Woodhouse Table I nylon high-E open values; homogeneous
    # equivalent materials recover the measured mu and EI used by the model.
    L0, T, first, B0, rho = .65, 70.3, 329.6, 43e-6, 8050.
    f0 = first / math.sqrt(1 + B0)
    mu = T / (2 * L0 * f0)**2
    EI = B0 * T * L0**2 / math.pi**2
    radius = math.sqrt(mu / (math.pi * rho))
    young = 4 * EI / (math.pi * radius**4)
    nominal = nominal_parameters()
    # This study reasons about the three-point scheme's stability bound and dispersion.
    grid = GridConfig(order=2)
    rows = []
    for fret in (0, 12, 24, 'box_corner'):
        if fret == 'box_corner':
            length = .1625
            case_mu = T / (2 * length * 1318.4)**2
            case_EI = .001984 * T * length**2 / math.pi**2
            case_radius = math.sqrt(case_mu / (math.pi * rho))
            case_young = 4 * case_EI / (math.pi * case_radius**4)
        else:
            length = L0 / 2**(fret/12)
            case_mu, case_EI, case_radius, case_young = mu, EI, radius, young
        # Off-centre pluck (0.45 L) and pickup (0.55 L) keep both position gradients nonzero.
        physical = torch.tensor([rho,T,case_radius,case_young,length,.002,1.,.001,.45,.55], dtype=torch.float64)
        theta = physical / nominal
        score = float(grid.physical_score(T/case_mu, case_EI/case_mu, .001, length, intervals=n, oversampling=args.oversampling))
        if score >= 1:
            if fret == 'box_corner':
                rows.append({'fret': fret, 'stability_score': score, 'status': 'outside_bound'})
                continue
            raise AssertionError(f'Outside strict grid bound: {score}')
        model = ForwardString(n, rate, excitation='raised_cosine', order=2)
        audio, jac = model.output_and_tangents(theta, seconds)
        amp = float(nominal[5]); expected_grad = 2*(audio[:, None]*jac).mean(dim=0)/amp**2
        cases = {}
        for count in (1, 3):
            x = theta.to('cuda').detach().requires_grad_()
            raw = simulate(x, n, rate, steps, 'raised_cosine', 256, count, order=2)
            (raw.square().mean()/amp**2).backward()
            gpu, grad = raw.detach().cpu(), x.grad.detach().cpu()
            torch.testing.assert_close(gpu, audio, rtol=1e-6, atol=1e-10)
            torch.testing.assert_close(grad, expected_grad, rtol=5e-4, atol=1e-7)
            cases[str(count)] = {
                'waveform_relative_l2': float((gpu-audio).norm()/audio.norm()),
                'gradient_relative_l2': float((grad-expected_grad).norm()/expected_grad.norm()),
            }
        row = {'fret': fret, 'length_m': length, 'stability_score': score,
               'max_abs_displacement_m': float(audio.abs().max()), 'cached': cases}
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {'status': 'passed', 'rate_hz': rate, 'intervals': n, 'seconds': seconds,
              'excitation': 'raised_cosine', 'physical_assumptions':
              'nylon high-E measured open f1 and B; fixed tension, mass and EI; also independent box corner; sigma0=1, sigma1=.001; amplitude=2mm; pluck 0.45 L, pickup 0.55 L',
              'cases': rows, 'gpu': torch.cuda.get_device_name()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
