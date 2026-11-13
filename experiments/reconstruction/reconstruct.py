"""Experimental discrete adjoint with checkpoint-anchored state reconstruction.

Unlike ``output_and_tangents``, this computes only the vector-Jacobian product
required by a scalar loss. Restricted to CUDA, float64, stride 1 and N <= 256.
"""
from functools import lru_cache
from pathlib import Path
import torch
from torch.autograd.function import once_differentiable
from src.accelerated import basis
from src.kc_string_forward import PARAMETER_COUNT, initialize, dual_multiply, pickup_weights

@lru_cache(maxsize=1)
def kernel():
    from torch.utils.cpp_extension import load
    return load(name='kc_checkpoint_reconstruct',sources=[str(Path(__file__).with_name('kc_reconstruct.cu'))],
                extra_cuda_cflags=['-O3','--fmad=false','-lineinfo'],with_cuda=True,verbose=False)

class _Adjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx,theta,intervals,rate,steps,excitation,checkpoint):
        if theta.dtype != torch.float64 or not theta.is_cuda or theta.ndim not in (1,2):
            raise ValueError(f'Expected CUDA float64 ({PARAMETER_COUNT},) or (B,{PARAMETER_COUNT}).')
        single=theta.ndim==1
        x=theta[None] if single else theta
        if x.shape[1]!=PARAMETER_COUNT or x.shape[0]==0 or intervals<4 or intervals%2 or intervals>256 or steps<0 or checkpoint<1:
            raise ValueError('Invalid batch, grid or checkpoint configuration.')
        if x.shape[0]==1:
            state,coeff,_=initialize(x[0],intervals,rate,excitation);state=state[None];coeff=coeff[None]
            pickup=pickup_weights(x[0],intervals,False)[None]
        else:
            state,coeff,_=torch.vmap(lambda row:initialize(row,intervals,rate,excitation))(x)
            pickup=torch.vmap(lambda row:pickup_weights(row,intervals,False))(x)
        # Modal pickup weights per item: value column for the kernels, tangent
        # columns (pickup position only) for the chain rule in backward.
        pickup=pickup.contiguous()
        h2=coeff[:,0].clone()
        coeff[:,2]=dual_multiply(coeff[:,2],h2)
        coeff[:,3]=dual_multiply(dual_multiply(coeff[:,3],h2),h2)
        coeff[:,5]=dual_multiply(coeff[:,5],h2)
        coeff[:,7]=dual_multiply(coeff[:,7],h2)
        transform,eigen,weights,ready=basis(intervals,str(theta.device),True)
        if ready is not None:
            stream=torch.cuda.current_stream(theta.device);stream.wait_event(ready)
            for item in (transform,eigen,weights): item.record_stream(stream)
        state=torch.einsum('ij,btjd->btid',transform.T,state).contiguous()
        values,checks=kernel().forward(state[...,0].contiguous(),coeff[...,0].contiguous(),eigen,pickup[...,0].contiguous(),steps,checkpoint)
        ctx.save_for_backward(state,coeff,eigen,pickup,checks)
        ctx.steps=steps;ctx.checkpoint=checkpoint;ctx.single=single
        return values[0] if single else values

    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        state,coeff,eigen,pickup,checks=ctx.saved_tensors
        grad=(grad[None] if ctx.single else grad).contiguous()
        statebar,g,gw,flags,drift=kernel().backward(coeff[...,0].contiguous(),eigen,pickup[...,0].contiguous(),grad,checks,ctx.steps,ctx.checkpoint)
        if bool(torch.any(flags != 0)):
            raise RuntimeError(f"State reconstruction failed safety checks: flags={flags.cpu().tolist()}, maximum relative boundary errors={drift.cpu().tolist()}")
        n=eigen.numel();e=eigen[None,:,None]
        dc=coeff[...,1:]
        dA=2*dc[:,1,None,:]+e*dc[:,2,None,:]-e.square()*dc[:,3,None,:]
        dP=dc[:,4,None,:]-e*dc[:,5,None,:]
        dQ=e*dc[:,7,None,:]
        result=torch.einsum('bti,btid->bd',statebar,state[...,1:])
        result=result+torch.einsum('bi,bid->bd',g[:,:n],dA)+torch.einsum('bi,bid->bd',g[:,n:2*n],dP)
        result=result+torch.einsum('bi,bid->bd',g[:,2*n:3*n],dQ)+g[:,3*n,None]*dc[:,6,:]
        # Pickup position: d loss/d w (per mode) through the weights' tangent columns.
        result=result+torch.einsum('bi,bid->bd',gw,pickup[...,1:])
        return (result[0] if ctx.single else result),None,None,None,None,None

def simulate(theta,intervals=176,rate=132300,steps=13230,excitation='multimode',checkpoint=256):
    return _Adjoint.apply(theta,intervals,rate,steps,excitation,checkpoint)
