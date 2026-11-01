"""Check selected strings within an independently differentiated batch of 128."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from benchmarks.exact_methods import BOX,GRID,audio_for,varied_parameters
from src.synthesis import Renderer
from src.optimization import criterion_for

def main():
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    duration=.25;batch=128
    values=varied_parameters(batch)
    cpu=Renderer(GRID,duration)
    target=cpu.synthesize(BOX.defaults())
    gpu=Renderer(GRID,duration,device='cuda')
    loss=criterion_for(target.cuda(),GRID.sample_rate)
    for method in ('cuda-modal','cuda-seeded','cuda-recompute','cuda-adjoint'):
        x=values.cuda().detach().clone().requires_grad_()
        outputs=audio_for(x,gpu,method,tangents=True)
        losses=torch.stack([loss(row) for row in outputs])
        (grads,)=torch.autograd.grad(losses.sum(),x)
        reference_loss=criterion_for(target,GRID.sample_rate)
        for i in (0,17,63,127):
            p=values[i].detach().clone().requires_grad_()
            ref=cpu(p);ref_loss=reference_loss(ref);(ref_grad,)=torch.autograd.grad(ref_loss,p)
            torch.testing.assert_close(outputs[i].detach().cpu(),ref.detach(),rtol=1e-6,atol=1e-10)
            torch.testing.assert_close(losses[i].detach().cpu(),ref_loss.detach(),rtol=5e-4,atol=1e-10)
            torch.testing.assert_close((grads[i].detach().cpu()*p.detach()),ref_grad.detach()*p.detach(),rtol=5e-4,atol=1e-8)
        print(method,'128-item isolated waveform/loss/gradient PASS',flush=True)
    with torch.no_grad():
        reg=audio_for(values.cuda(),gpu,'cuda-register',tangents=False)
        for i in (0,17,63,127):
            torch.testing.assert_close(reg[i].cpu(),cpu.synthesize(values[i]),rtol=1e-6,atol=1e-10)
    print('cuda-register 128-item isolated synthesis PASS',flush=True)
if __name__=='__main__':main()
