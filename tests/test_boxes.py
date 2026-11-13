"""Box transforms, grid stability and the benchmark workload."""

import pytest
import torch

from kc_string import physics, workload
from kc_string.boxes import MaterialParameterBox
from kc_string.grid import Grid, largest_stable_intervals


def test_logits_round_trip():
    values = workload.batch(5)
    torch.testing.assert_close(workload.BOX.decode(workload.BOX.encode(values)), values, rtol=1e-9, atol=0)


def test_decode_stays_in_the_box():
    low, high = workload.BOX.bounds()
    values = workload.BOX.decode(torch.linspace(-40, 40, 7, dtype=torch.float64)[:, None].expand(7, 10))
    assert bool(((values >= low) & (values <= high)).all())


def test_workload_batch_is_nested_and_inside_the_box():
    torch.testing.assert_close(workload.batch(4), workload.batch(8)[:4], rtol=0, atol=0)
    low, high = workload.BOX.bounds()
    values = workload.batch(128)
    assert bool(((values >= low) & (values <= high)).all())


def test_paper_grid():
    assert (workload.GRID.intervals, workload.GRID.integration_rate, workload.STEPS) == (176, 88200, 88200)


def test_unstable_grid_is_rejected():
    with pytest.raises(ValueError, match="unstable"):
        Grid(workload.BOX, sample_rate=44100, oversampling=1, intervals=176)


def test_largest_stable_intervals_is_stable_and_maximal():
    intervals = largest_stable_intervals(workload.BOX, oversampling=2, sample_rate=44100)
    Grid(workload.BOX, 44100, 2, intervals)
    with pytest.raises(ValueError):
        Grid(workload.BOX, 44100, 2, intervals + 2)


def test_closed_forms_of_the_nominal_string():
    density, tension, radius, _, length = physics.NOMINAL[:5]
    f0 = physics.fundamental(density, tension, radius, length)
    assert f0 == pytest.approx(physics.WAVE_SPEED / (2 * length))
    box = MaterialParameterBox(ranges=tuple((0.9 * v, 1.1 * v) for v in physics.NOMINAL))
    assert box.corners().shape == (2**10, 10)
