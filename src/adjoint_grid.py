"""Hand-written discrete adjoint of the grid scheme with checkpointed replay.

The grid counterpart of ``src.adjoint``: the same scheme, the same 256-step
checkpoints and the same hand-derived adjoint in one CUDA kernel, but stepped on
the grid values instead of the sine modes. Where the modal kernel multiplies each
mode by its eigenvalue, ``kc_adjoint_grid.cu`` applies the five-point stencil
across neighbouring threads. It exists to separate what the hand-written kernel
buys from what mode stepping buys. Restricted to CUDA, float64 and N <= 256.
"""
from functools import lru_cache
from pathlib import Path
import torch
from torch.autograd.function import once_differentiable
from .kc_string_forward import SPATIAL_ORDER, STENCILS, PARAMETER_COUNT, initialize, dual_multiply, pickup_weights

@lru_cache(maxsize=1)
def kernel():
    from torch.utils.cpp_extension import load
    return load(name='kc_checkpoint_adjoint_grid',sources=[str(Path(__file__).with_name('kc_adjoint_grid.cu'))],
                extra_cuda_cflags=['-O3','--fmad=false','-lineinfo'],with_cuda=True,verbose=False)

class _AdjointGrid(torch.autograd.Function):
    @staticmethod
    def forward(ctx,theta,intervals,rate,steps,excitation,checkpoint,order):
        if theta.dtype != torch.float64 or not theta.is_cuda or theta.ndim not in (1,2):
            raise ValueError(f'Expected CUDA float64 ({PARAMETER_COUNT},) or (B,{PARAMETER_COUNT}).')
        single=theta.ndim==1
        x=theta[None] if single else theta
        if x.shape[1]!=PARAMETER_COUNT or x.shape[0]==0 or intervals<4 or intervals%2 or intervals>256 or steps<0 or checkpoint<1:
            raise ValueError('Invalid batch, grid or checkpoint configuration.')
        if x.shape[0]==1:
            state,coeff,_=initialize(x[0],intervals,rate,excitation,order);state=state[None];coeff=coeff[None]
            pickup=pickup_weights(x[0],intervals,True)[None]
        else:
            state,coeff,_=torch.vmap(lambda row:initialize(row,intervals,rate,excitation,order))(x)
            pickup=torch.vmap(lambda row:pickup_weights(row,intervals,True))(x)
        # Grid states and nodal pickup weights: value column for the kernels,
        # tangent columns for the chain rule in backward.
        state=state.contiguous();pickup=pickup.contiguous()
        h2=coeff[:,0].clone()
        coeff[:,2]=dual_multiply(coeff[:,2],h2)
        coeff[:,3]=dual_multiply(dual_multiply(coeff[:,3],h2),h2)
        coeff[:,5]=dual_multiply(coeff[:,5],h2)
        coeff[:,7]=dual_multiply(coeff[:,7],h2)
        weights=STENCILS[order];pad=(5-len(weights))//2
        stencil=torch.tensor((0.,)*pad+tuple(weights)+(0.,)*pad,dtype=torch.float64,device=theta.device)
        values,checks=kernel().forward(state[...,0].contiguous(),coeff[...,0].contiguous(),stencil,pickup[...,0].contiguous(),steps,checkpoint)
        ctx.save_for_backward(state,coeff,stencil,pickup,checks)
        ctx.steps=steps;ctx.checkpoint=checkpoint;ctx.single=single
        return values[0] if single else values

    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        state,coeff,stencil,pickup,checks=ctx.saved_tensors
        grad=(grad[None] if ctx.single else grad).contiguous()
        statebar,g,gw=kernel().backward(coeff[...,0].contiguous(),stencil,pickup[...,0].contiguous(),grad,checks,ctx.steps,ctx.checkpoint)
        n=state.shape[2]
        dc=coeff[...,1:]
        # Per-point inner products of the kernel, summed over the grid: the gradient
        # with respect to c1, c2, c3, c4, c5 and c7 (l = 2 c1 u + c2 Du - c3 DDu + c4 p - c5 Dp, q = c7 Du).
        inner=g[:,:6*n].reshape(-1,6,n).sum(dim=2)
        result=torch.einsum('bti,btid->bd',statebar,state[...,1:])
        result=result+2*inner[:,0,None]*dc[:,1]+inner[:,1,None]*dc[:,2]-inner[:,2,None]*dc[:,3]
        result=result+inner[:,3,None]*dc[:,4]-inner[:,4,None]*dc[:,5]+inner[:,5,None]*dc[:,7]+g[:,6*n,None]*dc[:,6]
        # Pickup position: d loss/d w (per grid point) through the weights' tangent columns.
        result=result+torch.einsum('bi,bid->bd',gw,pickup[...,1:])
        return (result[0] if ctx.single else result),None,None,None,None,None,None

def simulate(theta,intervals=176,rate=132300,steps=13230,excitation='multimode',checkpoint=256,order=SPATIAL_ORDER):
    return _AdjointGrid.apply(theta,intervals,rate,steps,excitation,checkpoint,order)
