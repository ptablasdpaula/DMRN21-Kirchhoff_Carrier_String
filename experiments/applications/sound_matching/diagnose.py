"""Loss landscape over f0 around a fit: does the loss prefer the right pitch, or did the optimiser miss it?"""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from cels import Direction, STFTCumulativeEnergyLoss
from experiments.applications.common.render import BatchRenderer
from experiments.applications.common.media import read_wave
from experiments.applications.sound_matching.boxes import refinement_box, family_box, SAMPLE_RATE
from experiments.applications.sound_matching.loss import unit_rms, log_spectral_distance
from src.grid import GridConfig

ORDER = ('material_density', 'tension', 'radius', 'young_modulus', 'length', 'amplitude', 'sigma_0', 'sigma_1', 'pluck_position', 'pickup_position')


def losses_for(audio, target):
    cand, targ = unit_rms(audio), unit_rms(target)[None].expand(audio.shape[0], -1)
    out = {}
    crit = {
        'tlogcel': STFTCumulativeEnergyLoss(sample_rate=SAMPLE_RATE, directions=(Direction.RIGHT_UP, Direction.RIGHT_DOWN), log_weighting=True, reduction='none'),
        'cel': STFTCumulativeEnergyLoss(sample_rate=SAMPLE_RATE, reduction='none'),
        'logcel': STFTCumulativeEnergyLoss(sample_rate=SAMPLE_RATE, log_weighting=True, reduction='none'),
        'deccel': STFTCumulativeEnergyLoss(sample_rate=SAMPLE_RATE, time_decay_seconds=1.0, frequency_decay_hz=1000.0, reduction='none'),
        'updown': STFTCumulativeEnergyLoss(sample_rate=SAMPLE_RATE, directions=(Direction.UP, Direction.DOWN), log_weighting=True, reduction='none'),
    }
    with torch.no_grad():
        for name, c in crit.items():
            out[name] = c.to(audio.device)(cand, targ).cpu().tolist()
        out['lsd_db'] = [log_spectral_distance(cand[i].cpu(), targ[0].cpu()) for i in range(audio.shape[0])]
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', type=Path, required=True, help='fit.json of a (note, family)')
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--label-f0', type=float, required=True)
    parser.add_argument('--family', default='steel')
    parser.add_argument('--points', type=int, default=25)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    fit = json.load(open(args.fit))
    target, rate = read_wave(args.target)
    target = torch.as_tensor(target[:3*rate], dtype=torch.float64, device='cuda')
    base = torch.tensor([fit['physical'][k] for k in ORDER], dtype=torch.float64)
    f0_fit = fit['derived']['f0_hz']
    # Sweep f0 by scaling the tension (f0 ∝ sqrt(T)); keep everything else fixed. Grid: stable for the whole sweep.
    f0s = torch.linspace(0.75*args.label_f0, 1.3*args.label_f0, args.points, dtype=torch.float64)
    report = {'fit_f0_hz': f0_fit, 'label_f0_hz': args.label_f0, 'f0_hz': f0s.tolist(), 'variants': {}}
    # (pluck, pickup, excitation, sigma1 override or None, output derivative order)
    variants = {}
    for sigma1 in (2e-3, 5e-3, 1e-2):
        for pluck in (float(base[8]), 0.3, 0.2, 0.12):
            variants[f'sigma1_{sigma1:g}_pluck_{pluck:.2f}'] = (pluck, float(base[9]), 'raised_cosine:0.08', sigma1, 0)
    for name, (pluck, pickup, excitation, sigma1, derivative) in variants.items():
        rows = base.repeat(len(f0s), 1)
        rows[:, 1] = base[1]*(f0s/f0_fit)**2
        rows[:, 8], rows[:, 9] = pluck, pickup
        if sigma1 is not None:
            rows[:, 7] = sigma1
        # grid from a box covering the sweep
        lo, hi = rows.min(0).values, rows.max(0).values
        from src.parameters import MaterialParameterBox
        ranges = tuple((float(l)*0.999, float(h)*1.001+1e-12) for l, h in zip(lo, hi))
        ranges = ranges[:6] + ((0.0, max(float(hi[6]), 1e-3)), (0.0, max(float(hi[7]), 1e-6)), (0.05, 0.95), (0.05, 0.95))
        box = MaterialParameterBox(ranges=ranges)
        grid = None
        for oversampling in range(6, 13):
            from experiments.applications.common.boxes import largest_stable_intervals
            n = largest_stable_intervals(box, oversampling, SAMPLE_RATE)
            if n and n >= 128:
                grid = GridConfig(box, sample_rate=SAMPLE_RATE, oversampling=oversampling, intervals=n); break
        renderer = BatchRenderer(grid, 3.0, excitation, device='cuda')
        with torch.no_grad():
            audio = renderer(rows.cuda())
            if derivative:
                audio = torch.diff(audio, dim=-1, prepend=audio[:, :1])*SAMPLE_RATE
        res = losses_for(audio, target)
        res['grid'] = renderer.describe()
        best = {k: float(f0s[int(np.argmin(v))]) for k, v in res.items() if isinstance(v, list)}
        res['argmin_f0_hz'] = best
        report['variants'][name] = res
        print(name, 'grid', grid.intervals, grid.integration_rate, 'argmin f0 per loss:', {k: round(v, 1) for k, v in best.items()}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1))
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        names = ['tlogcel', 'cel', 'logcel', 'deccel', 'updown', 'lsd_db']
        fig, axes = plt.subplots(1, len(names), figsize=(3.2*len(names), 3), constrained_layout=True)
        for ax, key in zip(axes, names):
            for vname, res in report['variants'].items():
                ax.plot(f0s, res[key], label=vname, lw=1.5)
            ax.axvline(args.label_f0, color='k', lw=0.8, ls='--'); ax.axvline(f0_fit, color='gray', lw=0.8, ls=':')
            ax.set_title(key, fontsize=10); ax.set_xlabel('f0 (Hz)', fontsize=9); ax.tick_params(labelsize=8)
        axes[0].legend(fontsize=7)
        fig.savefig(args.output.with_suffix('.png'), dpi=110)
    except Exception as error:
        print('plot failed', error)


if __name__ == '__main__':
    main()
