"""Experimental whole-batch replay fallback for unsafe inverse reconstruction.

Fallback recomputes the primal checkpoints from the saved initial state and
verifies them before using the replay VJP. It is deliberately conservative and
may cost an extra forward pass plus a replay; a passing boundary check is not a
proof that every reconstructed interior gradient is accurate. The public
solver is unchanged. ``force_fallback`` is provided for tests and benchmarking.
"""
import torch
from torch.autograd.function import once_differentiable
from experiments.scalar_cache.reconstruct import _Adjoint
from src.adjoint import kernel as replay_kernel
from src.kc_string_forward import SPATIAL_ORDER


class _Fallback(_Adjoint):
    @staticmethod
    def forward(ctx,theta,intervals,rate,steps,excitation,checkpoint,cache_count,force_fallback):
        out=_Adjoint.forward(ctx,theta,intervals,rate,steps,excitation,checkpoint,cache_count,SPATIAL_ORDER)
        ctx.force_fallback=force_fallback
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        if not ctx.force_fallback:
            try:
                result=_Adjoint.backward(ctx,grad)
                # One gradient per input of this function: theta plus seven constants.
                return (result[0],)+(None,)*7
            except RuntimeError as exc:
                if not str(exc).startswith('Cached state reconstruction failed safety checks:'):
                    raise
        state,coeff,eigen,pickup,checks,_scalars=ctx.saved_tensors
        grad=(grad[None] if ctx.single else grad).contiguous()
        if not all(bool(torch.isfinite(t).all()) for t in (state,coeff,eigen,pickup,checks,grad)):
            raise RuntimeError('Replay fallback refused nonfinite forward state, coefficient, checkpoint or upstream gradient')
        kernel=replay_kernel()
        # Per-item modal pickup weights: value column for the kernels, tangent
        # columns (pickup position only) for the chain rule below.
        weights=pickup[...,0].contiguous()
        audio,recomputed=kernel.forward(state[...,0].contiguous(),coeff[...,0].contiguous(),eigen,weights,ctx.steps,ctx.checkpoint)
        if not bool(torch.isfinite(audio).all()) or not bool(torch.isfinite(recomputed).all()):
            raise RuntimeError('Replay fallback produced a nonfinite forward trajectory')
        try:
            torch.testing.assert_close(checks,recomputed,rtol=1e-10,atol=1e-13)
        except AssertionError as exc:
            raise RuntimeError('Replay fallback refused inconsistent forward checkpoints') from exc
        statebar,g,gw=kernel.backward(coeff[...,0].contiguous(),eigen,weights,grad,recomputed,ctx.steps,ctx.checkpoint)
        n=eigen.numel();e=eigen[None,:,None];dc=coeff[...,1:]
        dA=2*dc[:,1,None,:]+e*dc[:,2,None,:]-e.square()*dc[:,3,None,:]
        dP=dc[:,4,None,:]-e*dc[:,5,None,:];dQ=e*dc[:,7,None,:]
        result=torch.einsum('bti,btid->bd',statebar,state[...,1:])
        result=result+torch.einsum('bi,bid->bd',g[:,:n],dA)+torch.einsum('bi,bid->bd',g[:,n:2*n],dP)
        result=result+torch.einsum('bi,bid->bd',g[:,2*n:3*n],dQ)+g[:,3*n,None]*dc[:,6,:]
        # Pickup position: d loss/d w (per mode) through the weights' tangent columns.
        result=result+torch.einsum('bi,bid->bd',gw,pickup[...,1:])
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError('Replay fallback produced nonfinite gradients')
        return (result[0] if ctx.single else result),None,None,None,None,None,None,None


def simulate(theta,intervals=176,rate=132300,steps=13230,excitation='multimode',checkpoint=256,cache_count=3,force_fallback=False):
    if cache_count not in (1,2,3):raise ValueError('fallback requires cache_count 1, 2 or 3')
    return _Fallback.apply(theta,intervals,rate,steps,excitation,checkpoint,cache_count,force_fallback)
