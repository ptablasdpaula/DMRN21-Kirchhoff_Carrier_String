"""Guitar-string parameter boxes and grids for fitting 16 kHz NSynth notes.

Two string families (homogeneous equivalents of steel and nylon-core strings)
span the open-to-12th-fret range; a narrow refinement box around a stage-A
winner lets the fine stage use many more intervals. Stable interval counts are
set by a box's highest f0, the counts needed by its lowest, so each stage and
each note gets its own fixed grid.
"""
import math
import torch
from src.grid import GridConfig
from src.parameters import MaterialParameterBox
from experiments.applications.common.boxes import largest_stable_intervals, f0_range

SAMPLE_RATE = 16000
EXCITATION = 'raised_cosine:0.08'
# amplitude (m), sigma_0 (1/s), sigma_1 (m^2/s), pluck (fraction of L, one half only), pickup
COMMON = ((2e-4, 5e-3), (0.0, 5.0), (0.0, 3e-4), (0.05, 0.5), (0.05, 0.95))
# rho (kg/m^3), T (N), r (m), E (Pa), L (m)
FAMILIES = {
    'steel': ((7000., 8100.), (45., 180.), (0.15e-3, 0.45e-3), (10e9, 220e9), (0.325, 0.65)),
    'nylon': ((1000., 6500.), (40., 100.), (0.25e-3, 0.60e-3), (1e9, 8e9), (0.325, 0.65)),
}
STAGE_A_OVERSAMPLING = 10
REFINEMENT_FACTORS = {'rho': 1.1, 'T': 1.15, 'r': 1.15, 'E': 1.5, 'L': 1.05}
POSITION_MARGIN = 0.1


def family_box(family):
    if family not in FAMILIES:
        raise ValueError(f'Unknown family {family!r}; choose from {sorted(FAMILIES)}.')
    return MaterialParameterBox(ranges=FAMILIES[family] + COMMON)


def refinement_box(center, family, factors=REFINEMENT_FACTORS, position_margin=POSITION_MARGIN):
    """Multiplicative box around a stage-A winner, intersected with its family box."""
    outer = family_box(family)
    center = [float(v) for v in center]
    ranges = []
    for i, key in enumerate(('rho', 'T', 'r', 'E', 'L')):
        low, high = outer.ranges[i]
        ranges.append((max(low, center[i]/factors[key]), min(high, center[i]*factors[key])))
    ranges += list(COMMON[:3])
    for i in (8, 9):
        low, high = outer.ranges[i]
        ranges.append((max(low, center[i]-position_margin), min(high, center[i]+position_margin)))
    return MaterialParameterBox(ranges=tuple(ranges))


def stage_a_grid(family, oversampling=STAGE_A_OVERSAMPLING, sample_rate=SAMPLE_RATE):
    box = family_box(family)
    intervals = largest_stable_intervals(box, oversampling, sample_rate)
    if intervals is None:
        raise ValueError(f'{family} family box is not stable at {oversampling}x.')
    return GridConfig(box, sample_rate=sample_rate, oversampling=oversampling, intervals=intervals)


def choose_grid(box, *, sample_rate=SAMPLE_RATE, min_oversampling=6, max_oversampling=12, cap=256,
                headroom=1.5, minimum=32):
    """Smallest oversampling whose stable interval count reaches ~headroom x the partials below Nyquist."""
    f_low, _ = f0_range(box)
    target = min(cap, max(minimum, 2*math.ceil(headroom*(sample_rate/2)/f_low/2)))
    best = None
    for oversampling in range(min_oversampling, max_oversampling+1):
        intervals = largest_stable_intervals(box, oversampling, sample_rate, cap=cap)
        if intervals is None:
            continue
        if best is None or intervals > best[1]:
            best = (oversampling, intervals)
        if intervals >= target:
            return GridConfig(box, sample_rate=sample_rate, oversampling=oversampling, intervals=intervals)
    if best is None:
        raise ValueError('No stable grid in the oversampling range.')
    return GridConfig(box, sample_rate=sample_rate, oversampling=best[0], intervals=best[1])
