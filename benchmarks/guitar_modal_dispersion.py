#!/usr/bin/env python3
"""Linear modal frequency/stability survey of the reference FDTD grid.

This is a small-amplitude, undamped analytical calculation, not a nonlinear
render or listening test.  It follows the no-SAV reference finite differences
and uses only the Python standard library.
"""
import csv
import gzip
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'benchmarks/guitar_modal_dispersion_results'
N = 176
FS = 132300.0
OUTPUT_RATE = 44100.0
CUTOFF = 20000.0
BS = (13e-6, 50e-6, 124e-6)
# String order E2 through E4; Woodhouse et al., Table I, new sets.
STRINGS = ('E2', 'A2', 'D3', 'G3', 'B3', 'E4')
OPEN_F1 = (82.4, 110.0, 146.8, 196.0, 246.9, 329.6)
OPEN_B = {'nylon': (19, 13, 17, 124, 70, 43),
          'steel': (106, 45, 32, 30, 58, 21)}


def frequency(n, f0, b):
    return n * f0 * math.sqrt(1 + b * n * n)


def spatial_lambda(n):
    return (2 * N * math.sin(math.pi * n / (2 * N)))**2


def omega(n, f0, b):
    lam = spatial_lambda(n)
    return 2 * f0 * math.sqrt(lam + b / math.pi**2 * lam**2)


def discrete_frequency(n, f0, b):
    ratio = omega(n, f0, b) / (2 * FS)
    return FS / math.pi * math.asin(ratio) if ratio <= 1 else None


def write_csv(path, fields, rows):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'wt', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def survey_case(label, f1, b):
    f0 = f1 / math.sqrt(1 + b)
    highest_ratio = omega(N-1, f0, b) / (2 * FS)
    grid_stable = highest_ratio <= 1
    rows = []
    n = 1
    while frequency(n, f0, b) <= CUTOFF:
        cont = frequency(n, f0, b)
        if n >= N:
            status, fd, cents = 'not_represented', None, None
        elif not grid_stable:
            status, fd, cents = 'full_grid_unstable', None, None
        else:
            fd = discrete_frequency(n, f0, b)
            status = 'represented' if fd < OUTPUT_RATE/2 else 'output_nyquist_exceeded'
            cents = 1200 * math.log2(fd/cont)
        rows.append({'case': label, 'first_partial_hz': f1, 'tension_f0_hz': f0,
                     'B': b, 'full_grid_stable_undamped': grid_stable,
                     'stability_ratio': highest_ratio, 'mode': n,
                     'continuous_hz': cont, 'discrete_hz': fd,
                     'error_cents': cents, 'status': status})
        n += 1
    valid = [r for r in rows if r['status'] == 'represented']
    summary = {'case': label, 'first_partial_hz': f1, 'tension_f0_hz': f0,
               'B': b, 'full_grid_stable_undamped': grid_stable,
               'stability_ratio': highest_ratio, 'physical_modes_to_20khz': len(rows),
               'missing_modes_to_20khz': sum(r['status'] == 'not_represented' for r in rows),
               'output_nyquist_exceeded_modes': sum(r['status'] == 'output_nyquist_exceeded' for r in rows),
               'max_abs_error_cents': max((abs(r['error_cents']) for r in valid), default=None)}
    for mode in (1, 3, 5, 10, 20, 50, 100, 175):
        found = next((r for r in valid if r['mode'] == mode), None)
        summary[f'mode_{mode}_error_cents'] = found['error_cents'] if found else None
    return rows, summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_modes, summaries = [], []
    # A0 to E6 at quarter-semitone resolution. E2--E6 is the conventional
    # 24-fret six-string envelope; lower notes are extended/bass sensitivity.
    for q in range(0, (88-21)*4 + 1):
        midi = 21 + q/4
        f1 = 440 * 2**((midi-69)/12)
        band = 'standard_six_string_envelope' if midi >= 40 else 'extended_low_sensitivity'
        for b in BS:
            modes, summary = survey_case(f'{band}_midi_{midi:.2f}', f1, b)
            all_modes.extend(modes)
            summaries.append(summary)

    # Correlated fretting scenarios derived from measured open-string values.
    # Hold tension, effective stiffness, and mass/length fixed, and assume
    # L_fret = L_open * 2^(-fret/12); do not conflate with measured fretted data.
    correlated = []
    for material, bs in OPEN_B.items():
        for string, open_f1, open_micro_b in zip(STRINGS, OPEN_F1, bs):
            b0 = open_micro_b * 1e-6
            f0_open = open_f1 / math.sqrt(1+b0)
            for fret in range(25):
                scale = 2**(fret/12)
                b = b0 * scale**2
                f1 = f0_open * scale * math.sqrt(1+b)
                _, summary = survey_case(f'{material}_{string}_fret_{fret}', f1, b)
                summary.update(material=material, string=string, fret=fret,
                               open_f1_hz=open_f1, open_B=b0)
                correlated.append(summary)

    write_csv(OUT/'modes.csv.gz', list(all_modes[0]), all_modes)
    write_csv(OUT/'envelope_summary.csv', list(summaries[0]), summaries)
    write_csv(OUT/'measured_strings_fret_scenarios.csv', list(correlated[0]), correlated)
    for desc, seq in [('envelope', summaries), ('correlated', correlated)]:
        print(desc, 'cases', len(seq), 'stable', sum(r['full_grid_stable_undamped'] for r in seq),
              'unstable', sum(not r['full_grid_stable_undamped'] for r in seq))
    print('Example stable measured scenarios:')
    for item in correlated:
        if item['fret'] in (0, 12, 24):
            print(item['case'], f"f1={item['first_partial_hz']:.3f}", f"B={item['B']:.7g}",
                  'stable=', item['full_grid_stable_undamped'], 'missing=',item['missing_modes_to_20khz'],
                  'cents(1,3,5,10,20)=', [round(item[f'mode_{i}_error_cents'],3) if item[f'mode_{i}_error_cents'] is not None else None for i in (1,3,5,10,20)])
    print('Wrote', OUT)


if __name__ == '__main__':
    main()
