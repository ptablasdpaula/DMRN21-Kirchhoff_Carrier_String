"""Choose one discretisation for an entire parameter box, before fitting."""
from dataclasses import dataclass, field
import math
from . import kc_string as reference
from .parameters import ParameterBox


def reference_intervals():
    h = 1.01 * reference.minimum_grid_spacing(1/44100, reference.wave_speed,
                                             reference.stiffness, reference.sigma_1)
    return 2 * math.floor(reference.length / h / 2)


@dataclass(frozen=True)
class GridConfig:
    box: ParameterBox = field(default_factory=ParameterBox)
    sample_rate: int = 44100
    max_oversampling: int = 128
    max_intervals: int = 4096
    oversampling: int = field(init=False)
    intervals: int = field(init=False)
    minimum_intervals: int = field(init=False)

    def __post_init__(self):
        for name in ('sample_rate', 'max_oversampling', 'max_intervals'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        minimum = reference_intervals()
        f0, B, sigma1 = self.box.ranges[0][1], self.box.ranges[1][1], self.box.ranges[5][1]
        c = 2*self.box.length*f0
        K = c*self.box.length*math.sqrt(B)/math.pi
        for factor in range(1, self.max_oversampling+1):
            h = 1.01 * reference.minimum_grid_spacing(1/(self.sample_rate*factor), c, K, sigma1)
            intervals = 2*math.floor(self.box.length/h/2)
            if intervals >= minimum:
                if intervals > self.max_intervals:
                    raise ValueError(f'Grid requires {intervals} intervals, above limit {self.max_intervals}.')
                object.__setattr__(self, 'oversampling', factor)
                object.__setattr__(self, 'intervals', intervals)
                object.__setattr__(self, 'minimum_intervals', minimum)
                if self.corner_score > 1+1e-12:
                    raise ValueError('Computed grid violates the spacing-margin condition.')
                return
        raise ValueError(f'Box needs oversampling above ×{self.max_oversampling} to preserve {minimum} intervals. '
                         'Raise the resource limit or narrow the ranges.')

    @property
    def integration_rate(self):
        return self.sample_rate*self.oversampling

    @property
    def spacing(self):
        return self.box.length/self.intervals

    def score(self, f0, B, sigma1, *, intervals=None, oversampling=None):
        N = self.intervals if intervals is None else intervals
        factor = self.oversampling if oversampling is None else oversampling
        k = 1/(self.sample_rate*factor)
        # H=h/1.01: bending has the fourth power of the margin, not the second.
        inverse_H = N*1.01/self.box.length
        c2 = (2*self.box.length*f0)**2
        K2 = B*c2*self.box.length**2/math.pi**2
        return (c2*k*k+4*sigma1*k)*inverse_H**2 + 4*K2*k*k*inverse_H**4

    @property
    def corner_score(self):
        return self.score(self.box.ranges[0][1], self.box.ranges[1][1], self.box.ranges[5][1])

    def describe(self):
        return (f'Grid: {self.intervals} intervals (reference ×1 minimum {self.minimum_intervals}); '
                f'h={self.spacing:.7g} m\nOversampling ×{self.oversampling}; '
                f'internal rate {self.integration_rate:,} Hz; output {self.sample_rate:,} Hz\n'
                f'Worst-corner margin-adjusted stability utilisation: {self.corner_score:.6f} ≤ 1')
