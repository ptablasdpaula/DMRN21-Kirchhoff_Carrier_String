"""Construct the production modal string as a linearised toy-scan problem."""
import torch
from src.accelerated import basis
from src.kc_string_forward import PARAMETER_COUNT, dual_multiply, initialize, pickup_weights


def physical_problem(batch=1,intervals=176,rate=132300,steps=64,device='cpu'):
    theta=torch.ones(batch,PARAMETER_COUNT,dtype=torch.float64,device=device)
    # Deliberately nonidentical physical parameters, with positive damping. The
    # same mild rule moves the pluck and pickup positions a little off centre.
    index=torch.arange(batch,dtype=torch.float64,device=device)[:,None]
    cols=torch.arange(PARAMETER_COUNT,dtype=torch.float64,device=device)[None,:]
    theta=theta*torch.exp(.02*torch.sin((index+1)*(cols+1)*.73))
    state,coeff,_=torch.vmap(lambda row:initialize(row,intervals,rate,'multimode'))(theta)
    h2=coeff[:,0].clone()
    coeff[:,2]=dual_multiply(coeff[:,2],h2)
    coeff[:,3]=dual_multiply(dual_multiply(coeff[:,3],h2),h2)
    coeff[:,5]=dual_multiply(coeff[:,5],h2)
    coeff[:,7]=dual_multiply(coeff[:,7],h2)
    transform,eigen,_,_=basis(intervals,str(device),True)
    state=torch.einsum('ij,btjd->btid',transform.T,state)
    # Per-item modal readout: the value column drives the recurrence, the
    # tangent columns (pickup position only) carry d loss/d w into d loss/d theta.
    pickup=torch.vmap(lambda row:pickup_weights(row,intervals,False))(theta)
    e=eigen[None,:,None]
    Avec=2*coeff[:,1,None,:]+e*coeff[:,2,None,:]-e.square()*coeff[:,3,None,:]
    Pvec=coeff[:,4,None,:]-e*coeff[:,5,None,:]
    Qvec=e*coeff[:,7,None,:]
    t=torch.arange(steps,dtype=theta.dtype,device=device)
    upstream=(torch.sin(.013*t)+.27*torch.cos(.0037*t))[None,:].expand(batch,-1)/max(1,steps)
    problem=dict(p0=state[:,0,:,0],u0=state[:,1,:,0],A0=Avec[...,0],P0=Pvec[...,0],Q0=Qvec[...,0],m0=coeff[:,6,0],
                 dA=Avec[...,1:],dP=Pvec[...,1:],dQ=Qvec[...,1:],dm=coeff[:,6,1:],
                 w=pickup[...,0],dw=pickup[...,1:],upstream=upstream)
    return problem,state[...,1:],theta
