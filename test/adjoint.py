"""Checkpointed CUDA adjoint versus the independently implemented forward Jacobian."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.kc_string_forward import ForwardString,PARAMETER_COUNT
from src.adjoint import simulate

torch.set_num_threads(1)
assert torch.cuda.is_available()
# Off-centre pluck and pickup: the VJP covers all ten multipliers, including
# the pickup term accumulated from the per-mode weight gradient.
varied=[1.02,.98,1.01,1.03,1.01,.9,1.1,.95,1.02,.97]
for n in (4,32,128,176):
    rate=44100 if n<=128 else 132300
    for excitation in ('first_mode','multimode','raised_cosine'):
        model=ForwardString(n,rate,excitation=excitation,backend='modal')
        for steps,checkpoint in ((0,1),(1,1),(17,7),(35,64)):
            x=torch.tensor(varied,dtype=torch.float64,device='cuda',requires_grad=True)
            expected,jac=model.output_and_tangents(x,steps/rate)
            audio=simulate(x,n,rate,steps,excitation,checkpoint)
            probe=torch.linspace(-1,1,steps+2,dtype=torch.float64,device='cuda')
            actual=torch.autograd.grad(audio@probe,x)[0];ref=probe@jac
            torch.testing.assert_close(audio,expected,rtol=1e-8,atol=1e-12)
            torch.testing.assert_close(actual,ref,rtol=1e-6,atol=1e-9)
        print('PASS',n,excitation,flush=True)
    # Independent items exercise per-batch checkpoint and coefficient strides.
    x=torch.stack([torch.ones(PARAMETER_COUNT,device='cuda',dtype=torch.float64),torch.tensor(varied,device='cuda',dtype=torch.float64)]).requires_grad_()
    out=simulate(x,n,rate,35,'multimode',7)
    weights=torch.arange(out.numel(),dtype=torch.float64,device='cuda').reshape_as(out)/out.numel()
    grads=torch.autograd.grad((out*weights).sum(),x)[0]
    for i in range(2):
        y,j=ForwardString(n,rate,excitation='multimode',backend='modal').output_and_tangents(x[i],35/rate)
        torch.testing.assert_close(out[i],y,rtol=1e-8,atol=1e-12)
        torch.testing.assert_close(grads[i],weights[i]@j,rtol=1e-6,atol=1e-9)
    print('PASS batch',n,flush=True)
# Full-duration VJP on the exact default grid, pickup off the midpoint node.
x=torch.tensor([1.]*9+[.97],device='cuda',dtype=torch.float64,requires_grad=True);steps=4*132300
model=ForwardString(176,132300,excitation='multimode',backend='modal')
y,j=model.output_and_tangents(x,4.)
probe=torch.sin(torch.arange(y.numel(),dtype=torch.float64,device='cuda')*.001)
audio=simulate(x,176,132300,steps,'multimode',1024)
grad=torch.autograd.grad(audio@probe,x)[0]
torch.testing.assert_close(audio,y,rtol=1e-6,atol=1e-10)
torch.testing.assert_close(grad,probe@j,rtol=5e-4,atol=1e-8)
print('PASS four-second VJP',flush=True)
