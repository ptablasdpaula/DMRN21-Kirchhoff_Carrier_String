"""CPU checks for the sound-matching application: boxes and grids, the per-item loss, data filtering."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiments.applications.common.boxes import f0_range
from experiments.applications.sound_matching import data
from experiments.applications.sound_matching.boxes import family_box, refinement_box, stage_a_grid, choose_grid, FAMILIES
from experiments.applications.sound_matching.loss import make_criterion, per_item_loss, unit_rms, log_spectral_distance
from src.grid import GridConfig

torch.manual_seed(0)


def check_grids():
    expected_a = {'steel': 68, 'nylon': 60}
    for family in FAMILIES:
        box = family_box(family)
        grid = stage_a_grid(family)
        assert grid.oversampling == 10 and grid.intervals == expected_a[family], (family, grid.intervals)
        try:
            GridConfig(box, sample_rate=16000, oversampling=10, intervals=grid.intervals+2)
        except ValueError:
            pass
        else:
            raise AssertionError('stage-A grid is not the largest stable one')
        low, high = f0_range(box)
        assert low < 82.4 < 659.3 < high, (family, low, high)   # E2 to E5 inside both families
    centers = {'E2': ('steel', [7800., 120., .42e-3, 12e9, .65, 1e-3, 1., 5e-5, .2, .5]),
               'B3': ('nylon', [1150., 60., .40e-3, 4e9, .65, 1e-3, 1., 5e-5, .2, .5]),
               'E5': ('steel', [7800., 90., .18e-3, 200e9, .33, 1e-3, 1., 5e-5, .2, .5])}
    intervals = {}
    for name, (family, center) in centers.items():
        box = refinement_box(torch.tensor(center, dtype=torch.float64), family)
        grid = choose_grid(box)
        assert grid.oversampling == 6 and grid.corner_score <= 1, (name, grid.oversampling, grid.corner_score)
        intervals[name] = grid.intervals
    assert intervals['E2'] > intervals['B3'] > intervals['E5'] >= 56, intervals
    print('grids ok', intervals)


def check_loss():
    criterion = make_criterion(16000)
    target = torch.randn(8000, dtype=torch.float64)
    candidate = torch.randn(3, 8000, dtype=torch.float64, requires_grad=True)
    losses = per_item_loss(criterion, candidate, target)
    assert losses.shape == (3,)
    rows = torch.stack([per_item_loss(criterion, candidate[i:i+1].detach(), target)[0] for i in range(3)])
    torch.testing.assert_close(losses.detach(), rows, rtol=1e-10, atol=0)
    (grad,) = torch.autograd.grad(losses.sum(), candidate)
    grads = []
    for i in range(3):
        c = candidate[i:i+1].detach().requires_grad_()
        grads.append(torch.autograd.grad(per_item_loss(criterion, c, target).sum(), c)[0][0])
    torch.testing.assert_close(grad, torch.stack(grads), rtol=1e-9, atol=1e-14)
    scaled = per_item_loss(criterion, 37.0*candidate.detach()+0.2, target)
    torch.testing.assert_close(scaled, losses.detach(), rtol=1e-9, atol=1e-12)        # level and offset invariant
    assert float(per_item_loss(criterion, target[None], target)[0]) < 1e-12
    assert log_spectral_distance(target, target) < 1e-9 and log_spectral_distance(target, torch.randn(8000, dtype=torch.float64)) > 1
    u = unit_rms(torch.randn(2, 500, dtype=torch.float64)*5+3)
    torch.testing.assert_close(u.pow(2).mean(-1), torch.ones(2, dtype=torch.float64), rtol=1e-9, atol=1e-9)
    print('loss ok')


def check_data():
    def entry(name, family='guitar', source='acoustic', pitch=60, velocity=100, qualities=()):
        flags = [1 if q in qualities else 0 for q in data.QUALITIES]
        return {'note_str': name, 'instrument_str': 'x', 'instrument_family_str': family, 'instrument_source_str': source,
                'pitch': pitch, 'velocity': velocity, 'qualities': flags, 'qualities_str': list(qualities)}
    examples = {
        'a': entry('a', pitch=40, velocity=25), 'b': entry('b', pitch=50, velocity=50), 'c': entry('c', pitch=59, velocity=75),
        'd': entry('d', pitch=67, velocity=100), 'e': entry('e', pitch=76, velocity=127), 'e2': entry('e2', pitch=76, velocity=127),
        'r': entry('r', pitch=40, qualities=('reverb',)), 'dist': entry('dist', pitch=50, qualities=('distortion', 'bright')),
        'm': entry('m', pitch=59, qualities=('multiphonic',)), 't': entry('t', pitch=67, qualities=('tempo-synced',)),
        'bright': entry('bright', pitch=48, qualities=('bright',)),
        'elec': entry('elec', source='electronic', pitch=40), 'bass': entry('bass', family='bass', pitch=40),
    }
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/'examples.json'
        path.write_text(json.dumps(examples))
        kept, stats = data.pool(path)
        assert sorted(e['note_str'] for e in kept) == ['a', 'b', 'bright', 'c', 'd', 'e', 'e2'], [e['note_str'] for e in kept]
        assert stats['acoustic_guitar'] == 11 and stats['removed_by_quality'] == {'reverb': 1, 'distortion': 1, 'multiphonic': 1, 'tempo-synced': 1}
        chosen = data.select_notes(kept)
        assert [c['pitch'] for c in chosen] == [40, 50, 59, 67, 76] and len({c['velocity'] for c in chosen}) == 5
        assert all(c['pitch_offset'] == 0 for c in chosen)
        try:
            data.select_notes([e for e in kept if e['pitch'] != 59])
        except LookupError:
            pass
        else:
            raise AssertionError('select_notes must fail loudly on a missing pitch')
        near = data.select_notes([e for e in kept if e['pitch'] != 59] + [entry('c1', pitch=60)], max_offset=2)
        assert [c for c in near if c['requested_pitch'] == 59][0]['pitch_offset'] == 1
        bad = dict(entry('bad', qualities=('reverb',)), qualities_str=[])
        try:
            data.eligible(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('qualities/qualities_str mismatch must raise')
    tail = np.sin(np.linspace(0, 100, 64000))
    audio = np.concatenate([np.zeros(200), 0.001*np.ones(50), tail])   # the 0.001 plateau is below 2 % of the peak
    target, onset = data.prepare_target(audio, 16000, duration=3.0)
    expected = 250 + int(np.argmax(np.abs(tail) >= 0.02))
    assert onset == expected and target.shape == (48000,) and abs(target.mean()) < 1e-12, (onset, expected)
    print('data ok')


if __name__ == '__main__':
    check_grids(); check_loss(); check_data()
    print('sound_matching_cpu: all checks passed')


def check_catalogue():
    from experiments.applications.sound_matching import catalogue as cat
    catalogue = cat.load_catalogue()
    densities = {'plain_steel': 7800.0, 'nylon_plain': 1140.0}
    tolerance = {'plain_steel': 0.02, 'nylon_plain': 0.25}   # nylon gauges are nominal and the polymer density varies
    worst_chart, worst_geometry = 0.0, {}
    for entry in catalogue['entries']:
        check = entry['chart_check']
        scale = catalogue['chart_scale_in'][check['table']]
        predicted = cat.chart_tension_lb(entry['unit_weight_lb_in'], scale, check['pitch_hz'])
        worst_chart = max(worst_chart, abs(predicted/check['tension_lb']-1))
        if entry['class'] in densities:   # unit weight agrees with gauge x density for solid strings
            mu_geometry = densities[entry['class']]*math.pi*(entry['diameter_in']*cat.IN_TO_M/2)**2
            error = abs(cat.mass_per_length(entry)/mu_geometry-1)
            worst_geometry[entry['class']] = max(worst_geometry.get(entry['class'], 0.0), error)
            assert error < tolerance[entry['class']], (entry['code'], error)
        assert cat.effective_EI(catalogue, entry) > 0
    assert worst_chart < 0.01, worst_chart
    # A synthetic string built from an entry at a given scale and fret is matched first.
    entry = next(e for e in catalogue['entries'] if e['code'] == 'PB032')
    length = SCALES[1]*2**(-3/12)
    p = cat.predict(catalogue, entry, 146.8*2**(3/12), length)
    fitted = {'mu_kg_per_m': p['mu_kg_per_m']*1.02, 'tension_n': p['tension_n']*0.98, 'B': p['B']*1.1}
    best = cat.match(fitted, 146.8*2**(3/12), catalogue, top=3)
    # scale/fret pairs are nearly degenerate (650 mm at fret 4 is within 2 % of 24.75 in at fret 3): check the string and the length
    assert best[0]['code'] == 'PB032' and abs(best[0]['length_m']/length-1) < 0.03, best[0]
    q = cat.fitted_quantities([8050., 75., 3.556e-4, 174e9, 1.0, 2e-3, 1., 1e-4, .5, .5])   # the paper's nominal string
    assert abs(q['f0_hz']-76.6) < 0.5 and abs(q['B']-2.88e-4) < 1e-5
    print('catalogue ok', {'chart_max_rel_err': round(worst_chart, 4), 'geometry_max_rel_err': {k: round(v, 3) for k, v in worst_geometry.items()}})


SCALES = list(__import__('experiments.applications.sound_matching.catalogue', fromlist=['SCALES_M']).SCALES_M.values())

if __name__ == '__main__':
    import math
    check_catalogue()
    print('sound_matching_cpu: catalogue checks passed')
