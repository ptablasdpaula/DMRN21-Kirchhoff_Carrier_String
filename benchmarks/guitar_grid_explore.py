"""Illustrative modal counts and stability versus accuracy for the current box."""
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.grid import GridConfig


def modes_below(f0, B, cutoff):
    """Largest n with n*f0*sqrt(1+B*n**2) <= cutoff."""
    x = cutoff/f0
    root = x if B == 0 else math.sqrt(2*x*x / (1+math.sqrt(1+4*B*x*x)))
    n = math.floor(root)
    while (n+1)*f0*math.sqrt(1+B*(n+1)**2) <= cutoff: n += 1
    while n*f0*math.sqrt(1+B*n*n) > cutoff: n -= 1
    return n


def minimum_even_intervals(mode):
    return 2*math.ceil((mode+1)/2)


def stable_factor(grid, intervals):
    for factor in range(1, 10000):
        if grid.score(100., .001, .001, intervals=intervals, oversampling=factor) <= 1:
            return factor
    raise ValueError('Search limit exceeded')


def spatial_resolution(mode, power):
    intervals = minimum_even_intervals(mode)
    while True:
        x = math.pi*mode/(2*intervals)
        ratio = math.sin(x)/x
        if ratio**power >= .99: return intervals
        intervals += 2


def discrete_frequency(mode, f0, B, intervals, oversampling):
    k = 1/(44100*oversampling)
    c = 2*f0  # L = 1 metre in this example.
    stiffness = c*math.sqrt(B)/math.pi
    spatial_eigenvalue = 4*math.sin(math.pi*mode/(2*intervals))**2 * intervals**2
    omega = math.sqrt(c*c*spatial_eigenvalue + stiffness**2*spatial_eigenvalue**2)
    if k*omega/2 > 1: raise ValueError('Mode is unstable')
    return math.asin(k*omega/2)/(math.pi*k)


def main():
    grid = GridConfig(order=2)  # this study analyses the three-point scheme
    cutoff = 20000
    max_mode = modes_below(60., 1e-5, cutoff)
    intervals = minimum_even_intervals(max_mode)
    factor = stable_factor(grid, intervals)
    mode = 257  # An odd mode is visible at the midpoint pickup.
    continuous = mode*60*math.sqrt(1+1e-5*mode*mode)
    discrete = discrete_frequency(mode, 60., 1e-5, intervals, factor)
    spacing_1pct = spatial_resolution(max_mode, 1)
    squared_1pct = spatial_resolution(max_mode, 2)
    report = {'assumptions': {'cutoff_hz': cutoff, 'box_f0_hz': [60,100],
              'box_B': [1e-5,.001], 'box_sigma1_max':.001, 'length_m':1.,
              'grid_spacing_margin':1.01, 'excitation_for_frequency_example':'linear undamped single mode'},
              'max_mode_low_corner':max_mode, 'minimum_even_intervals':intervals,
              'stability_oversampling':factor, 'worst_corner_score':grid.score(100.,.001,.001,intervals=intervals,oversampling=factor),
              'visible_mode':mode, 'continuous_frequency_hz':continuous,
              'discrete_frequency_hz':discrete, 'relative_frequency_error':discrete/continuous-1,
              'illustrative_spatial_wavenumber_1pct': {'intervals':spacing_1pct, 'stability_oversampling':stable_factor(grid,spacing_1pct)},
              'illustrative_squared_wavenumber_1pct': {'intervals':squared_1pct, 'stability_oversampling':stable_factor(grid,squared_1pct)},
              'nylon_e2_measured_example_modes':modes_below(82.4,19e-6,cutoff),
              'steel_e2_measured_example_modes':modes_below(82.4,106e-6,cutoff)}
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__': main()
