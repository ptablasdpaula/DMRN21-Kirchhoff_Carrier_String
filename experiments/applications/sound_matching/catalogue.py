"""Match fitted string parameters to real strings from a manufacturer's specifications.

The waveform determines f0 = sqrt(T/mu)/(2L), B = pi^2 EI/(T L^2), the two
decays, the pluck and pickup fractions and only weakly the Kirchhoff--Carrier
strength; (rho, r, E, T, L) are not individually identifiable. For each
catalogue string (mass per length from D'Addario's unit weight, bending
stiffness from Woodhouse's measurements of the same construction class) and
each candidate scale length and fret, the string's tension at the note's f0
and its inharmonicity are predicted and compared with the fitted (mu, T, B).
"""
import json
import math
from pathlib import Path

LB_PER_IN_TO_KG_PER_M = 0.45359237/0.0254     # 17.858
LB_TO_N = 4.4482216
IN_TO_M = 0.0254
DEFAULT_PATH = Path(__file__).with_name('catalogue.json')
SCALES_M = {'25.5 in (steel acoustic)': 25.5*IN_TO_M, '24.75 in (Gibson)': 24.75*IN_TO_M, '650 mm (classical)': 0.650}


def load_catalogue(path=DEFAULT_PATH):
    with open(path) as handle:
        return json.load(handle)


def chart_tension_lb(unit_weight, scale_in, frequency_hz):
    return unit_weight*(2*scale_in*frequency_hz)**2/386.4


def mass_per_length(entry):
    return entry['unit_weight_lb_in']*LB_PER_IN_TO_KG_PER_M


def effective_EI(catalogue, entry):
    """Bending stiffness (N m^2) interpolated log-log in mass per length within the entry's construction class."""
    cls = catalogue['classes'][entry['class']]
    table = catalogue['woodhouse_table1'][cls['woodhouse_set']]
    points = sorted((table['mass_g_per_m'][i-1]*1e-3, table['EI_1e-6_Nm2'][i-1]*1e-6) for i in cls['strings'])
    mu = mass_per_length(entry)
    if mu <= points[0][0]:
        (m0, e0), (m1, e1) = points[0], points[1]
    elif mu >= points[-1][0]:
        (m0, e0), (m1, e1) = points[-2], points[-1]
    else:
        for (m0, e0), (m1, e1) in zip(points, points[1:]):
            if m0 <= mu <= m1:
                break
    slope = math.log(e1/e0)/math.log(m1/m0)
    return e0*(mu/m0)**slope


def predict(catalogue, entry, f0_hz, length_m):
    """Mass per length, tension and inharmonicity of ``entry`` tuned to ``f0_hz`` over ``length_m``."""
    mu = mass_per_length(entry)
    tension = mu*(2*length_m*f0_hz)**2
    EI = effective_EI(catalogue, entry)
    return {'mu_kg_per_m': mu, 'tension_n': tension, 'B': math.pi**2*EI/(tension*length_m**2), 'EI_Nm2': EI}


def match(fitted, f0_hz, catalogue=None, scales_m=SCALES_M, frets=range(13), weights=(1.0, 1.0, 0.5),
          tension_range_n=(30.0, 250.0), top=3):
    """Rank (string, scale, fret) by log-distance to the fitted (mu, T, B).

    ``fitted`` is a dict with ``mu_kg_per_m``, ``tension_n`` and ``B`` (from the
    fitted physical parameters); ``f0_hz`` is the fitted fundamental.
    """
    catalogue = catalogue or load_catalogue()
    w_mu, w_t, w_b = weights
    ranked = []
    for entry in catalogue['entries']:
        for scale_name, scale in scales_m.items():
            for fret in frets:
                length = scale*2**(-fret/12)
                p = predict(catalogue, entry, f0_hz, length)
                if not tension_range_n[0] <= p['tension_n'] <= tension_range_n[1]:
                    continue
                distance = math.sqrt(w_mu*math.log(p['mu_kg_per_m']/fitted['mu_kg_per_m'])**2
                                     + w_t*math.log(p['tension_n']/fitted['tension_n'])**2
                                     + w_b*math.log(p['B']/fitted['B'])**2)
                ranked.append({'code': entry['code'], 'class': entry['class'], 'material': entry.get('material', 'plain steel' if entry['class'] == 'plain_steel' else ''),
                               'set': entry.get('set'), 'diameter_in': entry['diameter_in'], 'scale': scale_name, 'fret': fret,
                               'length_m': length, 'mu_kg_per_m': p['mu_kg_per_m'], 'tension_n': p['tension_n'], 'B': p['B'],
                               'distance': distance})
    ranked.sort(key=lambda r: r['distance'])
    return ranked[:top]


def fitted_quantities(physical):
    """(mu, T, B, f0) from the ten physical parameters (sequence or tensor)."""
    rho, tension, radius, modulus, length = [float(v) for v in list(physical)[:5]]
    mu = rho*math.pi*radius**2
    return {'mu_kg_per_m': mu, 'tension_n': tension,
            'B': math.pi**2*modulus*(math.pi*radius**4/4)/(tension*length**2),
            'f0_hz': math.sqrt(tension/mu)/(2*length)}
