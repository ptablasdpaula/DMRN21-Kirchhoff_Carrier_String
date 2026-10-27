"""Renderer shape, resampling and six-parameter gradient checks."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.grid import GridConfig
from src.parameters import reference_parameters,material_multipliers
from src.synthesis import Renderer,modal_diagnostics
from src.kc_string_forward import ForwardString

def main():
    torch.set_num_threads(1)
    grid=GridConfig(); p=reference_parameters().requires_grad_()
    for excitation in ('first_mode','multimode','raised_cosine'):
        renderer=Renderer(grid,.003,excitation)
        audio=renderer(p)
        assert audio.numel()==132 and torch.isfinite(audio).all()
        relative=torch.ones(6,dtype=torch.float64,requires_grad=True)
        assert torch.autograd.gradcheck(lambda x:renderer(x*p.detach()),(relative,),eps=1e-5,atol=1e-7,rtol=2e-3)
        print(excitation, 'gradient passed')
    # Unextended low-level API retains its first-mode default and waveform.
    old=ForwardString(grid.intervals,grid.integration_rate)
    explicit=ForwardString(grid.intervals,grid.integration_rate,excitation='first_mode')
    theta=material_multipliers(p)
    torch.testing.assert_close(old(theta,.003),explicit(theta,.003),rtol=0,atol=0)
    rows=modal_diagnostics(p,grid)
    assert len(rows)==8 and all(row['discrete_hz']>0 for row in rows)
    print('Excitation, filtered output, six-coordinate gradients and linear diagnostics passed.')

if __name__=='__main__':main()
