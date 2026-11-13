"""Full filtered CeL+gradient through physical recurrence versus torch autograd."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from benchmarks.exact_methods import GRID
from experiments.parallel_segments.toy_scan import advance,coefficients
from experiments.structured_adjoint.physical import physical_problem
from experiments.structured_adjoint.structured import structured_vjp
from src.optimization import criterion_for
from src.synthesis import Renderer


def main():
    torch.set_num_threads(1);steps=3969;chunk=16;batch=1;device='cpu'
    problem,initial_derivs,_=physical_problem(batch,GRID.intervals,GRID.integration_rate,steps,device)
    x=torch.zeros(batch,problem['dm'].shape[-1],dtype=torch.float64,device=device,requires_grad=True)
    A,P,Q,m=coefficients(problem,x)
    prev=problem['p0']+torch.einsum('bni,bi->bn',initial_derivs[:,0],x)
    cur=problem['u0']+torch.einsum('bni,bi->bn',initial_derivs[:,1],x)
    # The readout moves with the pickup position through its tangent columns.
    w=problem['w']+torch.einsum('bni,bi->bn',problem['dw'],x)
    outputs=[(prev*w).sum(-1),(cur*w).sum(-1)]
    checks=[torch.stack((prev,cur),-2)]
    for t in range(steps):
        v=advance(prev,cur,A,P,Q,m);outputs.append((v*w).sum(-1));prev,cur=cur,v
        if (t+1)%chunk==0 or t+1==steps:checks.append(torch.stack((prev,cur),-2))
    raw=torch.stack(outputs,1)
    model=Renderer(GRID,steps/GRID.integration_rate,device=device)
    audio=model._filter_audio(raw[0])
    time=torch.arange(model.samples,dtype=torch.float64)/GRID.sample_rate
    target=1e-3*torch.sin(2*torch.pi*440*time)
    criterion=criterion_for(target,GRID.sample_rate);loss=criterion(audio)
    upstream,expected=torch.autograd.grad(loss,(raw,x))
    problem['upstream']=upstream[:,2:].detach()
    initial,got,gw,_=structured_vjp(problem,x.detach(),torch.stack(checks).detach(),chunk,4)
    n=GRID.intervals-1
    direct=torch.stack((upstream[:,0,None]*problem['w'],upstream[:,1,None]*problem['w']),dim=1)
    got=got+torch.einsum('bti,btid->bd',initial.reshape(batch,2,n)+direct,initial_derivs)
    # Pickup position: the two startup samples read the initial state directly.
    gw=gw+upstream[:,0,None]*problem['p0']+upstream[:,1,None]*problem['u0']
    got=got+torch.einsum('bi,bid->bd',gw,problem['dw'])
    torch.testing.assert_close(got,expected,rtol=5e-5,atol=1e-8)
    rel=((got-expected).norm()/expected.norm().clamp_min(1e-30)).item()
    print('FULL_FILTER_CEL_GRADIENT_CPU','steps',steps,'seconds',steps/GRID.integration_rate,'relative_l2',rel,flush=True)
if __name__=='__main__':main()
