"""Choose one discretisation for an entire parameter box, before fitting."""
from dataclasses import dataclass, field
import math
from . import kc_string as reference
from .parameters import ParameterBox, MaterialParameterBox


def reference_intervals():
    h = 1.01 * reference.minimum_grid_spacing(1/44100, reference.wave_speed,
                                             reference.stiffness, reference.sigma_1)
    return 2 * math.floor(reference.length / h / 2)


@dataclass(frozen=True)
class GridConfig:
    box: ParameterBox | MaterialParameterBox = field(default_factory=ParameterBox)
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
        cases = self.box.stability_cases()
        for factor in range(1, self.max_oversampling+1):
            intervals = min(2*math.floor(length/(1.01*reference.minimum_grid_spacing(
                1/(self.sample_rate*factor), c, K, sigma1))/2) for c,K,sigma1,length in cases)
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
        """Minimum spacing over the box; actual material-mode spacing is L/N."""
        return self.box.length_range[0]/self.intervals

    def score(self, f0, B, sigma1, *, intervals=None, oversampling=None):
        if self.box.coordinate_system != 'perceptual':
            raise ValueError('Use score_values with direct material coordinates.')
        N = self.intervals if intervals is None else intervals
        factor = self.oversampling if oversampling is None else oversampling
        k = 1/(self.sample_rate*factor)
        # H=h/1.01: bending has the fourth power of the margin, not the second.
        inverse_H = N*1.01/self.box.length
        c2 = (2*self.box.length*f0)**2
        K2 = B*c2*self.box.length**2/math.pi**2
        return (c2*k*k+4*sigma1*k)*inverse_H**2 + 4*K2*k*k*inverse_H**4

    def score_values(self, values):
        rho,tension,radius,modulus,length,_,_,sigma1=self.box.to_material(values).unbind()
        return self.physical_score(tension/(rho*math.pi*radius**2),modulus*radius**2/(4*rho),sigma1,length)

    def physical_score(self,c2,K2,sigma1,length,*,intervals=None,oversampling=None):
        N=self.intervals if intervals is None else intervals
        factor=self.oversampling if oversampling is None else oversampling
        k=1/(self.sample_rate*factor); inverse_H=N*1.01/length
        return (c2*k*k+4*sigma1*k)*inverse_H**2+4*K2*k*k*inverse_H**4

    @property
    def corner_score(self):
        return max(self.physical_score(c*c,K*K,sigma1,length) for c,K,sigma1,length in self.box.stability_cases())

    def describe(self):
        low,high=self.box.length_range
        spacing_text=f'h={low/self.intervals:.7g} m' if low==high else f'h=L/N ({low/self.intervals:.7g}–{high/self.intervals:.7g} m over length range)'
        return (f'Grid: {self.intervals} intervals (reference ×1 minimum {self.minimum_intervals}); '
                f'{spacing_text}\nOversampling ×{self.oversampling}; '
                f'internal rate {self.integration_rate:,} Hz; output {self.sample_rate:,} Hz\n'
                f'Worst-corner margin-adjusted stability utilisation: {self.corner_score:.6f} ≤ 1')
