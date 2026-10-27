"""Fixed-grid perceptual synthesis, filtered downsampling and linear diagnostics."""
import math
import cmath
import torch
from .kc_string_forward import ForwardString
from .kc_string_forward import nominal_parameters


class Renderer:
    def __init__(self, grid, duration=4., excitation='multimode', max_steps=10_000_000):
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('Duration must be finite and positive.')
        if round(duration*grid.integration_rate) > max_steps:
            raise ValueError(f'Render exceeds {max_steps:,} internal steps; shorten duration or narrow the box.')
        self.grid, self.duration, self.excitation = grid, duration, excitation
        self.samples = round(duration*grid.sample_rate)
        if self.samples < 1:
            raise ValueError('Duration must contain at least one output sample.')
        self.model = ForwardString(grid.intervals, grid.integration_rate, excitation=excitation)
        self.filter = None
        if grid.oversampling > 1:
            self.half_width = 16*grid.oversampling
            index = torch.arange(-self.half_width, self.half_width+1, dtype=torch.float64)
            cutoff = .45/grid.oversampling
            kernel = 2*cutoff*torch.sinc(2*cutoff*index)*torch.hann_window(index.numel(), periodic=False, dtype=torch.float64)
            self.filter = (kernel/kernel.sum()).reshape(1,1,-1)

    def __call__(self, values):
        # The fit's bounded parameter transform supplies the whole-box guarantee.
        # No per-evaluation stability test, projection or parameter-dependent grid.
        multipliers = self.grid.box.to_material(values) / nominal_parameters()
        audio = self.model(multipliers, self.duration)
        if self.filter is not None:
            padded = torch.nn.functional.pad(audio.reshape(1,1,-1), (self.half_width,self.half_width), mode='replicate')
            audio = torch.nn.functional.conv1d(padded,self.filter,stride=self.grid.oversampling).flatten()
        return audio[:self.samples]

    def synthesize(self, values):
        self.grid.box.validate(values)
        with torch.no_grad():
            audio = self(values)
        if not bool(torch.isfinite(audio).all()):
            raise RuntimeError('Simulation produced nonfinite output.')
        return audio


def modal_diagnostics(values, grid, count=8):
    """Continuous and discrete *linear* damped poles; not a nonlinear accuracy claim.

    q² is the positive eigenvalue of -Dxx. The discrete recurrence is
    A z² + D z + C = 0; frequencies come from arg(z), decay from -log|z|/k.
    Nonoscillatory poles have no cents metric.
    """
    rho,tension,radius,modulus,L,_,sigma0,sigma1=grid.box.to_material(values).detach().tolist()
    h=L/grid.intervals; k=1/grid.integration_rate
    c2=tension/(rho*math.pi*radius**2); K2=modulus*radius**2/(4*rho)
    rows=[]
    for mode in range(1,min(count,grid.intervals-1)+1):
        beta2=(mode*math.pi/L)**2
        decay=sigma0+sigma1*beta2
        continuous_pole=-decay+cmath.sqrt(complex(decay**2-c2*beta2-K2*beta2**2))
        continuous=abs(continuous_pole.imag)/(2*math.pi)
        q2=(2/h*math.sin(mode*math.pi/(2*grid.intervals)))**2
        A=1+sigma0*k
        D=-2+(c2*q2+K2*q2*q2)*k*k+2*sigma1*k*q2
        C=1-sigma0*k-2*sigma1*k*q2
        discriminant=D*D-4*A*C
        roots=((-D+cmath.sqrt(discriminant))/(2*A),(-D-cmath.sqrt(discriminant))/(2*A))
        pole=max(roots,key=abs)
        oscillatory=discriminant < 0 and abs(pole)>0
        numerical=abs(cmath.phase(pole))/(2*math.pi*k) if oscillatory else None
        numerical_decay=-math.log(abs(pole))/k if abs(pole)>0 else math.inf
        rows.append(dict(mode=mode,continuous_hz=continuous if continuous>0 else None,
                         discrete_hz=numerical, cents=1200*math.log2(numerical/continuous) if oscillatory and continuous>0 else None,
                         continuous_decay=-continuous_pole.real, discrete_decay=numerical_decay))
    return rows


def nonlinear_strength(values):
    """First-mode tension-modulation indicator; not a bound for other shapes."""
    return float((values[1]*(values[2]/values[3])**2).detach())
