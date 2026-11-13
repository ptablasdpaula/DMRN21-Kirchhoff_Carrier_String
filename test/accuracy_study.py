"""Refinement tables, decimation equivalence, pluck width and a short-run smoke for the accuracy study."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import math
import torch

from benchmarks.accuracy_study import (BENCHMARK_INTERVALS, BENCHMARK_NAME, BENCHMARK_ORDER, BENCHMARK_OVERSAMPLING,
                                       OUTPUT_RATE, fdtd_audio, modes_below_nyquist, refinement, string_constants)
from src import kc_string as reference
from src.grid import GridConfig
from src.kc_string_forward import PARAMETER_COUNT, excitation_shape, nominal_parameters, raised_cosine_fraction
from src.synthesis import Renderer, decimate


def main():
    torch.set_num_threads(1)
    assert (BENCHMARK_INTERVALS, BENCHMARK_OVERSAMPLING, BENCHMARK_ORDER, BENCHMARK_NAME) == (176, 2, 4, 'fdtd-176-2x')
    c, K, sigma1, length = string_constants(nominal_parameters())
    for actual, expected in ((c, reference.wave_speed), (K, reference.stiffness),
                             (sigma1, reference.sigma_1), (length, reference.length)):
        assert math.isclose(actual, expected, rel_tol=1e-12), (actual, expected)
    tables = {2: [272, 390, 556, 788, 1118], 4: [236, 338, 480, 682, 968]}
    for order, expected in tables.items():
        table = [refinement(factor, c, K, sigma1, length, order)[0] for factor in (3, 6, 12, 24, 48)]
        assert table == expected, (order, table)
        grid = GridConfig(order=order)
        for factor, intervals in zip((3, 6, 12, 24, 48), table):
            assert grid.physical_score(c*c, K*K, sigma1, length, intervals=intervals, oversampling=factor) <= 1
    assert modes_below_nyquist(nominal_parameters()) == 123

    grid = GridConfig()
    renderer = Renderer(grid, .01)
    raw = torch.randn(round(.01*grid.integration_rate)+2, dtype=torch.float64)
    assert torch.equal(renderer._filter_audio(raw), decimate(raw, grid.oversampling, renderer.samples))
    assert decimate(raw[:441], 1, 441).shape == (441,)

    node_based = excitation_shape(176, 'raised_cosine')
    physical = excitation_shape(176, f'raised_cosine:{raised_cosine_fraction(176)!r}')
    assert (node_based-physical).abs().max() < 1e-15
    support = torch.nonzero(node_based).flatten()
    assert node_based[87] == 1 and support.min() == 45 and support.max() == 129

    theta = torch.ones(PARAMETER_COUNT, dtype=torch.float64)
    width = f'raised_cosine:{raised_cosine_fraction(176)!r}'
    for order in (2, 4):
        for factor in (3, 6):
            intervals = refinement(factor, c, K, sigma1, length, order)[0]
            audio = fdtd_audio(theta, intervals, factor, width, .05, order=order)
            assert audio.shape == (round(.05*OUTPUT_RATE),) and bool(torch.isfinite(audio).all())
    print('PASS: refinement tables for both stencils, 123 sub-Nyquist modes, decimation, pluck width and short refined runs')


if __name__ == '__main__':
    main()
