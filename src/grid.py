"""Choose one discretisation for an entire parameter box, before fitting."""
from dataclasses import dataclass, field
import math
from . import kc_string as reference
from .kc_string_forward import SPATIAL_ORDER, STENCILS, eigenvalue_factor
from .parameters import ParameterBox, MaterialParameterBox


def reference_intervals(order=SPATIAL_ORDER):
    """Largest stable even grid for the nominal string at 44.1 kHz: 148 (order 2), 128 (order 4).

    The five-point stencil's largest eigenvalue is 4/3 of the three-point value,
    which tightens the spacing bound by sqrt(4/3) at a fixed time step.
    """
    h = 1.01 * math.sqrt(eigenvalue_factor(order)) * reference.minimum_grid_spacing(
        1/44100, reference.wave_speed, reference.stiffness, reference.sigma_1)
    return 2 * math.floor(reference.length / h / 2)


@dataclass(frozen=True)
class GridConfig:
    """Spatial order, oversampling and interval count for one parameter box.

    Without overrides the smallest oversampling factor whose box-corner-stable
    grid keeps at least ``reference_intervals(order)`` intervals is chosen.
    Giving both ``oversampling`` and ``intervals`` fixes the grid explicitly;
    it must still satisfy the margin-adjusted bound at the box corner.
    """
    box: ParameterBox | MaterialParameterBox = field(default_factory=ParameterBox)
    sample_rate: int = 44100
    max_oversampling: int = 128
    max_intervals: int = 4096
    order: int = SPATIAL_ORDER
    oversampling: int | None = None
    intervals: int | None = None
    minimum_intervals: int = field(init=False)
    explicit: bool = field(init=False)

    def __post_init__(self):
        for name in ('sample_rate', 'max_oversampling', 'max_intervals'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if self.order not in STENCILS:
            raise ValueError('Spatial order must be 2 or 4.')
        minimum = reference_intervals(self.order)
        object.__setattr__(self, 'minimum_intervals', minimum)
        if (self.oversampling is None) != (self.intervals is None):
            raise ValueError('Give both oversampling and intervals, or neither.')
        if self.oversampling is not None:
            for name in ('oversampling', 'intervals'):
                value = getattr(self, name)
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    raise ValueError(f'{name} must be a positive integer.')
            if self.intervals % 2 or self.intervals < 4 or self.intervals > self.max_intervals:
                raise ValueError(f'intervals must be even, >= 4 and <= {self.max_intervals}.')
            object.__setattr__(self, 'explicit', True)
            if self.corner_score > 1+1e-12:
                raise ValueError(f'{self.intervals} intervals at x{self.oversampling} violate the margin-adjusted '
                                 f'stability bound at the box corner ({self.corner_score:.3f} > 1).')
            return
        object.__setattr__(self, 'explicit', False)
        margin = 1.01 * math.sqrt(eigenvalue_factor(self.order))
        cases = self.box.stability_cases()
        for factor in range(1, self.max_oversampling+1):
            intervals = min(2*math.floor(length/(margin*reference.minimum_grid_spacing(
                1/(self.sample_rate*factor), c, K, sigma1))/2) for c,K,sigma1,length in cases)
            if intervals >= minimum:
                if intervals > self.max_intervals:
                    raise ValueError(f'Grid requires {intervals} intervals, above limit {self.max_intervals}.')
                object.__setattr__(self, 'oversampling', factor)
                object.__setattr__(self, 'intervals', intervals)
                if self.corner_score > 1+1e-12:
                    raise ValueError('Computed grid violates the spacing-margin condition.')
                return
        raise ValueError(f'Box needs oversampling above x{self.max_oversampling} to preserve {minimum} intervals. '
                         'Raise the resource limit or narrow the ranges.')

    @property
    def integration_rate(self):
        return self.sample_rate*self.oversampling

    @property
    def spacing(self):
        """Minimum spacing over the box; actual material-mode spacing is L/N."""
        return self.box.length_range[0]/self.intervals

    def score(self, f0, B, sigma1, *, intervals=None, oversampling=None, order=None):
        if self.box.coordinate_system != 'perceptual':
            raise ValueError('Use score_values with direct material coordinates.')
        c2 = (2*self.box.length*f0)**2
        K2 = B*c2*self.box.length**2/math.pi**2
        return self.physical_score(c2, K2, sigma1, self.box.length, intervals=intervals, oversampling=oversampling, order=order)

    def score_values(self, values):
        rho,tension,radius,modulus,length,_,_,sigma1=self.box.to_material(values)[:8].unbind()  # positions do not affect stability
        return self.physical_score(tension/(rho*math.pi*radius**2),modulus*radius**2/(4*rho),sigma1,length)

    def physical_score(self,c2,K2,sigma1,length,*,intervals=None,oversampling=None,order=None):
        """Margin-adjusted stability utilisation; the grid is stable when this is <= 1.

        H = h/1.01: bending carries the fourth power of the margin, not the
        second. The stencil's eigenvalue factor gamma multiplies the wave and
        loss term once and the bending term twice (h -> h/sqrt(gamma)).
        """
        N=self.intervals if intervals is None else intervals
        factor=self.oversampling if oversampling is None else oversampling
        gamma=eigenvalue_factor(self.order if order is None else order)
        k=1/(self.sample_rate*factor); inverse_H=N*1.01/length
        return gamma*(c2*k*k+4*sigma1*k)*inverse_H**2+4*K2*k*k*(gamma*inverse_H**2)**2

    @property
    def corner_score(self):
        return max(self.physical_score(c*c,K*K,sigma1,length) for c,K,sigma1,length in self.box.stability_cases())

    def describe(self):
        low,high=self.box.length_range
        spacing_text=f'h={low/self.intervals:.7g} m' if low==high else f'h=L/N ({low/self.intervals:.7g}–{high/self.intervals:.7g} m over length range)'
        origin='explicit' if self.explicit else f'reference x1 minimum {self.minimum_intervals}'
        return (f'Spatial order {self.order} (eigenvalue factor {eigenvalue_factor(self.order):g}); '
                f'grid: {self.intervals} intervals ({origin}); {spacing_text}\n'
                f'Oversampling x{self.oversampling}; internal rate {self.integration_rate:,} Hz; output {self.sample_rate:,} Hz\n'
                f'Worst-corner margin-adjusted stability utilisation: {self.corner_score:.6f} <= 1')
