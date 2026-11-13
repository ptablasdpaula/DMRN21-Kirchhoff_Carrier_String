"""Experimental end-to-end interface for the structured discrete adjoint.

The primal and its checkpoints are exactly the existing replay kernel. The
backward is a bounded-window, untruncated structured suffix scan and is much
slower than fused CUDA until demonstrated otherwise. First order only.
"""
import torch
from torch.autograd.function import once_differentiable
from src.adjoint import _Adjoint as _Primal
from src.kc_string_forward import PARAMETER_COUNT, SPATIAL_ORDER
from experiments.structured_adjoint.structured import structured_vjp


class _Structured(torch.autograd.Function):
    @staticmethod
    def forward(ctx,theta,intervals,rate,steps,excitation,checkpoint,window):
        if not isinstance(window,int) or isinstance(window,bool) or window<1:
            raise ValueError('window must be a positive integer')
        out=_Primal.forward(ctx,theta,intervals,rate,steps,excitation,checkpoint,SPATIAL_ORDER)
        ctx.window=window
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        state,coeff,eigen,pickup,checks=ctx.saved_tensors
        grad=(grad[None] if ctx.single else grad).contiguous()
        n=eigen.numel();e=eigen[None,:,None]
        A=2*coeff[:,1,None,:]+e*coeff[:,2,None,:]-e.square()*coeff[:,3,None,:]
        P=(coeff[:,4,None,:]-e*coeff[:,5,None,:])
        Q=e*coeff[:,7,None,:]
        # Per-item modal readout (value column); its tangent columns carry the
        # pickup-position chain rule below.
        weights=pickup[...,0]
        problem=dict(p0=state[:,0,:,0],u0=state[:,1,:,0],
          A0=A[...,0],P0=P[...,0],Q0=Q[...,0],m0=coeff[:,6,0],
          dA=A[...,1:],dP=P[...,1:],dQ=Q[...,1:],dm=coeff[:,6,1:],
          w=weights,upstream=grad[:,2:])
        zeros=coeff.new_zeros((coeff.shape[0],PARAMETER_COUNT))
        initial,g,gw,_=structured_vjp(problem,zeros,checks,ctx.checkpoint,ctx.window)
        initial=initial.reshape(coeff.shape[0],2,n)
        direct=torch.stack((grad[:,0,None]*weights,grad[:,1,None]*weights),dim=1)
        g=g+torch.einsum('bti,btid->bd',initial+direct,state[...,1:])
        # Pickup position: the two startup samples read the initial state, the
        # rest comes from the scan; both pass through the weights' tangents.
        gw=gw+grad[:,0,None]*state[:,0,:,0]+grad[:,1,None]*state[:,1,:,0]
        g=g+torch.einsum('bi,bid->bd',gw,pickup[...,1:])
        return (g[0] if ctx.single else g),None,None,None,None,None,None


def simulate(theta,intervals=176,rate=132300,steps=13230,excitation='multimode',checkpoint=32,window=4):
    return _Structured.apply(theta,intervals,rate,steps,excitation,checkpoint,window)
