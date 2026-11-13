"""One fixed, stable grid for a whole parameter box, chosen before optimising."""

from dataclasses import dataclass

from .scheme import ORDER, eigenvalue_factor


@dataclass(frozen=True)
class Grid:
    """Interval count and oversampling, checked against the box's worst stability case.

    The internal rate is ``sample_rate * oversampling``. Stability uses Russo et
    al.'s 1 % spacing margin and the stencil's eigenvalue factor.
    """

    box: object
    sample_rate: int
    oversampling: int
    intervals: int
    order: int = ORDER

    def __post_init__(self):
        if self.intervals % 2 or self.intervals < 4:
            raise ValueError("The interval count must be even and at least 4.")
        score = max(self.score(*case) for case in self.box.stability_cases())
        if score > 1 + 1e-12:
            raise ValueError(
                f"{self.intervals} intervals at x{self.oversampling} are unstable at the box corner ({score:.3f} > 1)."
            )

    @property
    def integration_rate(self):
        """Internal sample rate of the scheme (Hz)."""
        return self.sample_rate * self.oversampling

    def score(self, wave_speed, stiffness, sigma_1, length):
        """Stability utilisation for one string; the grid is stable when it is at most 1.

        With H = h / 1.01, the stencil's eigenvalue factor gamma multiplies the wave
        and loss terms once and the bending term twice.
        """
        gamma = eigenvalue_factor(self.order)
        k = 1 / self.integration_rate
        inverse_h = self.intervals * 1.01 / length
        wave_and_loss = gamma * (wave_speed**2 * k * k + 4 * sigma_1 * k) * inverse_h**2
        return wave_and_loss + 4 * stiffness**2 * k * k * (gamma * inverse_h**2) ** 2


def largest_stable_intervals(box, oversampling, sample_rate, cap=256, order=ORDER):
    """Largest even interval count up to ``cap`` that is stable over the whole box, or None."""
    for intervals in range(cap - cap % 2, 3, -2):
        try:
            Grid(box, sample_rate, oversampling, intervals, order)
        except ValueError:
            continue
        return intervals
    return None
