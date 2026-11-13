"""Fixed-grid perceptual synthesis, filtered downsampling and linear diagnostics."""
import math
import cmath
import torch
from .kc_string_forward import ForwardString
from .kc_string_forward import discrete_eigenvalues, nominal_parameters


def antialias_kernel(oversampling, device='cpu'):
    """Hann-windowed sinc low-pass with cutoff .45 of the output rate, 32·OF+1 taps.

    Returned as a (1, 1, taps) conv1d weight with unit DC gain.
    """
    half_width = 16*oversampling
    index = torch.arange(-half_width, half_width+1, dtype=torch.float64)
    cutoff = .45/oversampling
    kernel = 2*cutoff*torch.sinc(2*cutoff*index)*torch.hann_window(index.numel(), periodic=False, dtype=torch.float64)
    return (kernel/kernel.sum()).reshape(1,1,-1).to(device)


def decimate(audio, oversampling, samples, kernel=None):
    """Replicate-pad, filter and keep every ``oversampling``-th sample, then the first ``samples``.

    ``Renderer`` skips this at 1×; calling it with ``oversampling=1`` still
    applies the 33-tap filter, which lets two signals at the output rate be
    compared in the same band.
    """
    if kernel is None:
        kernel = antialias_kernel(oversampling, audio.device)
    half_width = (kernel.shape[-1]-1)//2
    padded = torch.nn.functional.pad(audio.reshape(1,1,-1), (half_width,half_width), mode='replicate')
    return torch.nn.functional.conv1d(padded, kernel, stride=oversampling).flatten()[:samples]


class Renderer:
    def __init__(self, grid, duration=4., excitation='multimode', max_steps=10_000_000,
                 device='cpu', backend='native', chunk_steps=8192, threads=256):
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        if self.device.type not in ('cpu','cuda'):
            raise ValueError('Only CPU and CUDA devices are supported.')
        self.backend = backend
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('Duration must be finite and positive.')
        if round(duration*grid.integration_rate) > max_steps:
            raise ValueError(f'Render exceeds {max_steps:,} internal steps; shorten duration or narrow the box.')
        self.grid, self.duration, self.excitation = grid, duration, excitation
        self.samples = round(duration*grid.sample_rate)
        if self.samples < 1:
            raise ValueError('Duration must contain at least one output sample.')
        self.model = ForwardString(grid.intervals, grid.integration_rate, excitation=excitation, backend=backend,
                                   chunk_steps=chunk_steps, threads=threads, order=grid.order)
        self.filter = None
        if grid.oversampling > 1:
            self.half_width = 16*grid.oversampling
            self.filter = antialias_kernel(grid.oversampling, self.device)

    def __call__(self, values):
        # The fit's bounded parameter transform supplies the whole-box guarantee.
        # No per-evaluation stability test, projection or parameter-dependent grid.
        values = values.to(self.device)
        multipliers = self.grid.box.to_material(values) / nominal_parameters(self.device)
        audio = self.model(multipliers, self.duration)
        return self._filter_audio(audio)

    def forward_seeded(self, values, mask):
        """Experimental exact recurrence using selected coordinate sensitivities."""
        values = values.to(self.device)
        if not torch.is_grad_enabled() or not values.requires_grad or not any(mask):
            return self(values)
        from .seeded import seeded_audio
        audio = seeded_audio(values, self.grid.box, self.model, self.duration, mask)
        return self._filter_audio(audio)

    def _filter_audio(self, audio):
        if self.filter is not None:
            return decimate(audio, self.grid.oversampling, self.samples, self.filter)
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
    rho,tension,radius,modulus,L,_,sigma0,sigma1=grid.box.to_material(values).detach().tolist()[:8]
    h=L/grid.intervals; k=1/grid.integration_rate
    c2=tension/(rho*math.pi*radius**2); K2=modulus*radius**2/(4*rho)
    eigenvalues=discrete_eigenvalues(grid.intervals, grid.order).tolist()
    rows=[]
    for mode in range(1,min(count,grid.intervals-1)+1):
        beta2=(mode*math.pi/L)**2
        decay=sigma0+sigma1*beta2
        continuous_pole=-decay+cmath.sqrt(complex(decay**2-c2*beta2-K2*beta2**2))
        continuous=abs(continuous_pole.imag)/(2*math.pi)
        q2=-eigenvalues[mode-1]/h**2
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
