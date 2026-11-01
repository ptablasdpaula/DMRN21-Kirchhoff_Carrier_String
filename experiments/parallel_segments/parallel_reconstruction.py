"""Time-parallel checkpoint-segment reconstruction with sequential adjoint.

This is an experimental exact-recurrence prototype: reconstruction of up to
``group`` segments is launched in parallel, while gradients are propagated
across segment boundaries sequentially. It does not implement a parallel scan.
"""
from functools import lru_cache
from pathlib import Path
import torch
from torch.autograd.function import once_differentiable
from experiments.reconstruction.reconstruct import _Adjoint as Sequential

@lru_cache(maxsize=1)
def kernel():
    from torch.utils.cpp_extension import load
    return load(name='kc_parallel_segment_reconstruct',
                sources=[str(Path(__file__).with_name('parallel_reconstruction.cu'))],
                extra_cuda_cflags=['-O3','--fmad=false','-lineinfo'],with_cuda=True,verbose=False)

class _Parallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx,theta,intervals,rate,steps,excitation,checkpoint,group):
        if not isinstance(group,int) or isinstance(group,bool) or not 1<=group<=65535:
            raise ValueError('group must be an integer between 1 and 65535')
        ctx.group=group
        return Sequential.forward(ctx,theta,intervals,rate,steps,excitation,checkpoint)
    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        state,coeff,eigen,pickup,checks=ctx.saved_tensors
        grad=(grad[None] if ctx.single else grad).contiguous()
        # Per-item modal pickup weights (value column) for the kernel; the
        # tangent columns (pickup position only) feed the chain rule below.
        bars,g,gw,flags,drift=kernel().backward(coeff[...,0].contiguous(),eigen,pickup[...,0].contiguous(),grad,checks,ctx.steps,ctx.checkpoint,ctx.group)
        if bool(torch.any(flags != 0)):
            raise RuntimeError(f'Parallel state reconstruction failed safety checks: flags={flags.cpu().tolist()}, maximum relative boundary errors={drift.max().item()}')
        n=eigen.numel();e=eigen[None,:,None];dc=coeff[...,1:]
        dA=2*dc[:,1,None,:]+e*dc[:,2,None,:]-e.square()*dc[:,3,None,:]
        dP=dc[:,4,None,:]-e*dc[:,5,None,:];dQ=e*dc[:,7,None,:]
        result=torch.einsum('bti,btid->bd',bars,state[...,1:])
        result+=torch.einsum('bi,bid->bd',g[:,:n],dA)+torch.einsum('bi,bid->bd',g[:,n:2*n],dP)
        result+=torch.einsum('bi,bid->bd',g[:,2*n:3*n],dQ)+g[:,3*n,None]*dc[:,6,:]
        result+=torch.einsum('bi,bid->bd',gw,pickup[...,1:])
        return (result[0] if ctx.single else result),None,None,None,None,None,None

def simulate(theta,intervals=176,rate=132300,steps=13230,excitation='multimode',checkpoint=256,group=64):
    return _Parallel.apply(theta,intervals,rate,steps,excitation,checkpoint,group)
