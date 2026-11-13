"""Independent CPU check of conditional scan against actual modal recurrence."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.kc_string_forward import initialize,dual_multiply
from src.accelerated import basis

def test_physical_conditional_scan():
    torch.set_num_threads(1)
    for intervals in (4,32,176):
      for excitation in ('first_mode','multimode','raised_cosine'):
       for k in (1,7,64,256):
        theta=torch.tensor([1.02,.98,1.01,1.03,1.01,.9,1.1,.95,1.02,.97],dtype=torch.float64)
        state,c,_=initialize(theta,intervals,132300,excitation)
        h2=c[0].clone();c[2]=dual_multiply(c[2],h2);c[3]=dual_multiply(dual_multiply(c[3],h2),h2);c[5]=dual_multiply(c[5],h2);c[7]=dual_multiply(c[7],h2)
        transform,e,_,_=basis(intervals,'cpu',True);state=torch.einsum('ij,tjd->tid',transform.T,state)
        p,u=state[0,:,0],state[1,:,0];c=c[:,0];A=2*c[1]+e*c[2]-e*e*c[3];P=c[4]-e*c[5];Q=e*c[7];m=c[6]
        initial=torch.stack((u,p),dim=-1);primal=[];alpha=[]
        for _ in range(k):
            l=A*u+P*p;q=Q*u;z=(q@p+m*(q@l))/(1+m*(q@q));v=m*(l-q*z)
            alpha.append(z);primal.append(torch.stack((v,u),dim=-1));p,u=u,v
        alpha=torch.stack(alpha);M=torch.zeros((k,intervals-1,2,2),dtype=torch.float64)
        M[:,:,0,0]=m*(A[None,:]-Q[None,:]*alpha[:,None]);M[:,:,0,1]=m*P;M[:,:,1,0]=1
        stride=1
        while stride<k:
            old=M.clone();M[stride:]=old[stride:]@old[:-stride];stride*=2
        observed=(M@initial[None,:,:,None]).squeeze(-1);expected=torch.stack(primal)
        torch.testing.assert_close(observed,expected,rtol=1e-8,atol=1e-12)
    print('PASS physical conditional state scan vs serial recurrence')

if __name__=='__main__':test_physical_conditional_scan()
