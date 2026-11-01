#!/usr/bin/env python3
"""Plan a shared fixed KC grid over a documented guitar-string family.

The bounds are sufficient in exact arithmetic for the damped non-SAV update;
they do not claim accuracy or cover arbitrary strings. Uses only Python stdlib.
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'benchmarks/guitar_stability_results'
N, FS, LENGTH, SIGMA1_MAX, MARGIN = 176, 44100, .65, .001, 1.01
PITCHES = [329.6, 246.9, 196., 146.8, 110., 82.4]
B_MICRO = {
    'nylon': [43, 70, 124, 17, 13, 19],
    'steel_new': [21, 58, 30, 32, 45, 106],
}


def score(c2, kappa2, length, sigma1, factor):
    """Conservative condition: <= 1 is safe with the grid-spacing margin."""
    k = 1 / (FS * factor)
    inverse_h = MARGIN * N / length
    return ((c2 * k*k + 4 * sigma1*k)*inverse_h**2
            + 4*kappa2*k*k*inverse_h**4)


def minimum_factor(c2, kappa2, length, sigma1):
    for factor in range(1, 10001):
        if score(c2, kappa2, length, sigma1, factor) < 1:
            return factor
    raise ValueError('Factor exceeds 10000')


def main():
    rows = []
    for material, b_values in B_MICRO.items():
        for string, (pitch, b_micro) in enumerate(zip(PITCHES, b_values), 1):
            b_open = b_micro*1e-6
            f0_open = pitch / math.sqrt(1+b_open)
            c2 = (2*LENGTH*f0_open)**2
            kappa2 = b_open*c2*LENGTH**2/math.pi**2
            for fret in range(25):
                length = LENGTH * 2**(-fret/12)
                f0 = f0_open * 2**(fret/12)
                B = b_open * 2**(fret/6)
                factor = minimum_factor(c2, kappa2, length, SIGMA1_MAX)
                rows.append(dict(material=material, string=string, fret=fret,
                                 open_first_partial_hz=pitch, open_B=b_open,
                                 length_m=length, tension_frequency_hz=f0, B=B,
                                 score_at_3x=score(c2,kappa2,length,SIGMA1_MAX,3),
                                 min_factor=factor, integration_rate_hz=FS*factor,
                                 margin_score=score(c2,kappa2,length,SIGMA1_MAX,factor)))
    # A conservative independent-parameter box. These corners are wider than
    # the observed correlated 12-string/fret sample and not all are physical.
    # Extend the low-frequency and long-scale side as an explicit sensitivity
    # margin for extended-range / bass-like strings. These are assumed bounds,
    # not a claim that every instrument or wound-string construction fits.
    box = dict(f0_min_hz=27.5, f0_max_hz=329.6*4,
               B_min=1e-6, B_max=124e-6*16, L_min_m=LENGTH/4,
               L_max_m=.9, sigma1_min_m2_s=0., sigma1_max_m2_s=SIGMA1_MAX)
    f0, B, length = box['f0_max_hz'], box['B_max'], box['L_min_m']
    c2 = (2*length*f0)**2
    kappa2 = B*c2*length**2/math.pi**2
    box_factor = minimum_factor(c2,kappa2,length,SIGMA1_MAX)
    summary = {
        'intervals': N, 'output_rate_hz': FS, 'spacing_margin': MARGIN,
        'sigma1_max_m2_s_assumption': SIGMA1_MAX,
        'correlated_case_count': len(rows),
        'correlated_global_factor': max(r['min_factor'] for r in rows),
        'correlated_worst_cases': [r for r in rows if r['min_factor'] == max(t['min_factor'] for t in rows)],
        'cases_failing_3x_margin': sum(r['score_at_3x'] >= 1 for r in rows),
        'cases_passing_3x_margin': sum(r['score_at_3x'] < 1 for r in rows),
        'adaptive_factor_histogram': {str(i):sum(r['min_factor']==i for r in rows) for i in sorted({r['min_factor'] for r in rows})},
        'box_bounds': box, 'box_global_factor': box_factor,
        'box_worst_margin_score_at_factor': score(c2,kappa2,length,SIGMA1_MAX,box_factor),
        'box_worst_margin_score_at_previous_factor': score(c2,kappa2,length,SIGMA1_MAX,box_factor-1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'cases.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
