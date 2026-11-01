"""Structured segment-adjoint algebra and physical recurrence validation."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from unittest.mock import patch
from experiments.parallel_segments.toy_scan import make_problem,forward,advance,coefficients,segment_scan
from experiments.structured_adjoint.physical import physical_problem
from experiments.structured_adjoint.structured import compose,matvec,local_terms,structured_vjp,checked_relative_per_item


def rel(a,b):return ((a-b).norm()/b.norm().clamp_min(1e-30)).item()


def test_compose():
    torch.manual_seed(18)
    B,n=3,5;dtype=torch.float64
    def rand(*shape):return torch.randn(*shape,dtype=dtype)
    # Block factors may be singular. Composition must not invert them.
    d1=rand(B,n,2,2);d1[:,0]=0
    d2=rand(B,n,2,2);d2[:,1]=0
    u1,v1=rand(B,n,2,3),rand(B,n,2,3)
    u2,v2=rand(B,n,2,4),rand(B,n,2,4)
    b1,b2=rand(B,n,2),rand(B,n,2);x=rand(B,n,2)
    d,u,v,b=compose((d1,u1,v1,b1),(d2,u2,v2,b2))
    got=matvec(d,u,v,x)+b
    ref=matvec(d1,u1,v1,matvec(d2,u2,v2,x)+b2)+b1
    torch.testing.assert_close(got,ref,rtol=1e-12,atol=1e-12)


def test_physical_local_jacobian(device):
    p,_,_=physical_problem(batch=1,intervals=176,steps=1,device=device)
    x=torch.stack((p['p0'],p['u0']),dim=-1).detach().requires_grad_()
    A,P,Q,m=p['A0'],p['P0'],p['Q0'],p['m0']
    def step(x):
        old,current=x.unbind(-1)
        return torch.stack((current,advance(old,current,A,P,Q,m)),dim=-1)
    y=step(x);target=torch.randn_like(y)
    expected,=torch.autograd.grad((y*target).sum(),x)
    old,current=x.detach().unbind(-1);v=y.detach()[...,1]
    q=Q*current;l=A*current+P*old
    alpha=((q*old).sum(-1)+m*(q*l).sum(-1))/(1+m*(q*q).sum(-1))
    D,a,b,*_=local_terms(old,current,v,alpha,A,P,Q,m)
    got=torch.einsum('bnij,bnj->bni',D,target)+a*(b*target).sum((-2,-1))[:,None,None]
    torch.testing.assert_close(got,expected,rtol=1e-11,atol=1e-11)
    print('PHYSICAL_LOCAL_JACOBIAN',device,'rel',rel(got,expected),flush=True)


def test_dense_reference(device):
    p=make_problem(4,2,64,device=device);theta=torch.zeros(2,8,dtype=torch.float64,device=device)
    _,check=forward(p,theta,16)
    initial,g,_,_=structured_vjp(p,theta,check,16,4)
    ref_initial,ref_g=segment_scan(p,theta,check,16)
    torch.testing.assert_close(initial,ref_initial,rtol=1e-10,atol=1e-12)
    torch.testing.assert_close(g,ref_g,rtol=1e-10,atol=1e-12)
    print('DENSE_SCAN_REFERENCE',device,'state',rel(initial,ref_initial),'parameters',rel(g,ref_g),flush=True)


def test_mixed_loudness_guards(device):
    # Global Frobenius comparisons would let the loud item hide this error.
    reference=torch.zeros(1,2,4,2,dtype=torch.float64,device=device)
    reference[0,0]=1.;reference[0,1]=1e-8
    corrupt=reference.clone();corrupt[0,1,0,0]+=1e-10
    assert rel(corrupt,reference)<1e-8
    for name in ('Checkpoint reconstruction','Segment boundary adjoint'):
        try:checked_relative_per_item(corrupt,reference,name)
        except RuntimeError as exc:assert '[0, 1]' in str(exc)
        else:raise AssertionError(f'{name} masked the quiet item')
    problem=make_problem(4,2,1,device=device)
    problem['p0'][1]*=1e-6;problem['u0'][1]*=1e-6
    theta=torch.zeros(2,8,dtype=torch.float64,device=device)
    _,checks=forward(problem,theta,1)
    damaged=checks.clone();damaged[-1,1,1,0]+=1e-11
    assert rel(damaged[-1],checks[-1])<1e-8
    try:structured_vjp(problem,theta,damaged,1,1)
    except RuntimeError as exc:assert 'Checkpoint reconstruction' in str(exc) and '[0, 1]' in str(exc)
    else:raise AssertionError('Corrupted quiet checkpoint was not rejected')
    print('MIXED_LOUDNESS_GUARDS',device,'passed',flush=True)


def test_parameter_and_mass_guards(device):
    problem=make_problem(4,2,4,device=device)
    theta=torch.zeros(2,8,dtype=torch.float64,device=device)
    _,checks=forward(problem,theta,4)
    bad=dict(problem);bad['dA']=bad['dA'].clone();bad['dA'][1,0,0]=float('nan')
    with patch('experiments.structured_adjoint.structured.coefficients',return_value=coefficients(problem,theta)):
        try:structured_vjp(bad,theta,checks,4,1)
        except RuntimeError as exc:assert 'Nonfinite parameter VJP' in str(exc) and '[0, 1]' in str(exc)
        else:raise AssertionError('Nonfinite parameter derivative was not rejected')
    bad=dict(problem);bad['m0']=bad['m0'].clone();bad['m0'][1]=0.
    try:structured_vjp(bad,theta,checks,4,1)
    except ValueError as exc:assert 'Singular mass coefficient' in str(exc)
    else:raise AssertionError('Singular mass was not rejected')
    print('PARAMETER_AND_MASS_GUARDS',device,'passed',flush=True)


def test_end_to_end(device,intervals,chunk,window,batch,segments):
    steps=chunk*segments
    if intervals:
        problem,initial_derivs,_=physical_problem(batch,intervals,132300,steps,device)
    else:
        problem=make_problem(4,batch,steps,device=device);initial_derivs=None
    # The parameter count is the problem's: ten for the physical string, eight for the toy.
    x=torch.zeros(batch,problem['dm'].shape[-1],dtype=torch.float64,device=device,requires_grad=True)
    A,P,Q,m=coefficients(problem,x)
    if initial_derivs is not None:
        prev=problem['p0']+torch.einsum('bni,bi->bn',initial_derivs[:,0],x)
        cur=problem['u0']+torch.einsum('bni,bi->bn',initial_derivs[:,1],x)
        # The readout moves with the pickup position through its tangent columns.
        w=problem['w']+torch.einsum('bni,bi->bn',problem['dw'],x)
        checkpoints=[torch.stack((prev,cur),dim=-2)];ys=[]
        for t in range(steps):
            v=advance(prev,cur,A,P,Q,m);ys.append((v*w).sum(-1));prev,cur=cur,v
            if (t+1)%chunk==0:checkpoints.append(torch.stack((prev,cur),dim=-2))
        audio=torch.stack(ys,1);check=torch.stack(checkpoints)
    else:
        audio,check=forward(problem,x,chunk)
    expected,=torch.autograd.grad((audio*problem['upstream']).sum(),x)
    initial,got,gw,detail=structured_vjp(problem,x.detach(),check.detach(),chunk,window)
    if initial_derivs is not None:
        got=got+torch.einsum('bti,btid->bd',initial.reshape(batch,2,intervals-1),initial_derivs)
        got=got+torch.einsum('bi,bid->bd',gw,problem['dw'])
    torch.testing.assert_close(got,expected,rtol=5e-6,atol=1e-9)
    print('VJP',device,intervals or 4,chunk,window,batch,'relative',rel(got,expected),'rank',max(x['rank'] for x in detail),flush=True)


def test_partial_last_segment(device,chunk):
    batch=2;intervals=32;steps=4*chunk-3
    problem,initial_derivs,_=physical_problem(batch,intervals,132300,steps,device)
    x=torch.zeros(batch,problem['dm'].shape[-1],dtype=torch.float64,device=device,requires_grad=True)
    A,P,Q,m=coefficients(problem,x)
    prev=problem['p0']+torch.einsum('bni,bi->bn',initial_derivs[:,0],x)
    cur=problem['u0']+torch.einsum('bni,bi->bn',initial_derivs[:,1],x)
    w=problem['w']+torch.einsum('bni,bi->bn',problem['dw'],x)
    checks=[torch.stack((prev,cur),-2)];ys=[]
    for t in range(steps):
        v=advance(prev,cur,A,P,Q,m);ys.append((v*w).sum(-1));prev,cur=cur,v
        if (t+1)%chunk==0 or t+1==steps:checks.append(torch.stack((prev,cur),-2))
    expected,=torch.autograd.grad((torch.stack(ys,1)*problem['upstream']).sum(),x)
    initial,got,gw,_=structured_vjp(problem,x.detach(),torch.stack(checks).detach(),chunk,4)
    got=got+torch.einsum('bti,btid->bd',initial.reshape(batch,2,intervals-1),initial_derivs)
    got=got+torch.einsum('bi,bid->bd',gw,problem['dw'])
    torch.testing.assert_close(got,expected,rtol=5e-6,atol=1e-9)
    print('PARTIAL_LAST_SEGMENT',device,chunk,steps,'relative',rel(got,expected),flush=True)


def main():
    torch.set_num_threads(1)
    device=sys.argv[1] if len(sys.argv)>1 else 'cpu'
    test_compose();test_physical_local_jacobian(device);test_dense_reference(device);test_mixed_loudness_guards(device);test_parameter_and_mass_guards(device)
    for k in (16,32,64):
        for w in (1,2,3,4):test_end_to_end(device,0,k,w,2,5)
        test_end_to_end(device,32,k,4,2,4)
        test_partial_last_segment(device,k)
    test_end_to_end(device,176,16,2,1,2)
    print('ALL STRUCTURED ADJOINT TESTS PASSED',flush=True)
if __name__=='__main__':main()
