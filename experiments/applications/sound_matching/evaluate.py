"""Aggregate sound-matching fits into page data: metrics, derived quantities, catalogue matches, asset copies."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.applications.sound_matching import catalogue as cat
from experiments.applications.sound_matching.fit import cents

ASSETS = ('fit.wav', 'target_matched_level.wav', 'spectrograms.png', 'fit.gif')


def summarise(run_dir, manifest_path, assets_dir=None, top=3):
    run_dir = Path(run_dir)
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    catalogue = cat.load_catalogue()
    results = {'manifest': {k: manifest[k] for k in ('filter', 'statistics', 'pitches', 'velocities')}, 'notes': []}
    for note in manifest['notes']:
        folder = run_dir/note['note_str']
        if not (folder/'summary.json').exists():
            continue
        with open(folder/'summary.json') as handle:
            summary = json.load(handle)
        entry = {'note': {k: note[k] for k in ('note_str', 'instrument_str', 'pitch', 'velocity', 'qualities_str', 'label_f0_hz', 'onset_sample')},
                 'winner': summary['winner'], 'families': {}}
        for family in summary['families']:
            with open(folder/family/'fit.json') as handle:
                fit = json.load(handle)
            physical = [fit['physical'][k] for k in ('material_density', 'tension', 'radius', 'young_modulus', 'length',
                                                     'amplitude', 'sigma_0', 'sigma_1', 'pluck_position', 'pickup_position')]
            fitted = cat.fitted_quantities(physical)
            matches = cat.match(fitted, fitted['f0_hz'], catalogue, top=top)
            names = ('material_density', 'tension', 'radius', 'young_modulus', 'length', 'amplitude', 'sigma_0', 'sigma_1', 'pluck_position', 'pickup_position')
            family_box = fit['stages']['A']['box']
            at_bounds = [n for n, v, (lo, hi) in zip(names, physical, family_box)
                         if (hi-lo) > 0 and (abs(v-lo) <= 0.01*(hi-lo) or abs(v-hi) <= 0.01*(hi-lo))]
            record = {'loss': fit['loss'], 'at_bounds': at_bounds, 'signal': fit.get('signal', 'displacement'), 'loss_variant': fit.get('loss', 'tlogcel'), 'log_spectral_distance_db': fit['log_spectral_distance_db'],
                      'f1_cents_vs_label': fit['f1_cents_vs_label'], 'derived': fit['derived'], 'physical': fit['physical'],
                      'fitted': fitted, 'catalogue': matches,
                      'stages': {s: {k: fit['stages'][s][k] for k in ('grid', 'evaluations', 'seconds', 'best_loss')} for s in ('A', 'B')},
                      'assets': {}}
            if assets_dir is not None:
                target_dir = Path(assets_dir)/note['note_str']/family
                target_dir.mkdir(parents=True, exist_ok=True)
                for name in ASSETS:
                    if (folder/family/name).exists():
                        shutil.copy(folder/family/name, target_dir/name)
                        record['assets'][name] = str(Path(note['note_str'])/family/name)
            entry['families'][family] = record
        results['notes'].append(entry)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='fit.py output directory')
    parser.add_argument('--manifest', type=Path, default=Path('/data/scratch/acw794/data/nsynth/selected/manifest.json'))
    parser.add_argument('--assets', type=Path, default=None, help='copy WAV/PNG/GIF assets here (e.g. docs/supplement/sound_matching)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    results = summarise(args.run, args.manifest, args.assets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as handle:
        json.dump(results, handle, indent=1)
    for entry in results['notes']:
        best = entry['families'][entry['winner']]
        top = best['catalogue'][0] if best['catalogue'] else None
        print(f"{entry['note']['note_str']}: winner {entry['winner']}, CeL {best['loss']:.4f}, LSD {best['log_spectral_distance_db']:.1f} dB, "
              f"f1 {best['derived']['f1_hz']:.1f} Hz ({best['f1_cents_vs_label']:+.0f} cents vs label), "
              + (f"nearest string {top['code']} {top['material']} on {top['scale']} fret {top['fret']} (T {top['tension_n']:.0f} N)" if top else 'no catalogue match'))
    print(f'results -> {args.output}')


if __name__ == '__main__':
    main()
