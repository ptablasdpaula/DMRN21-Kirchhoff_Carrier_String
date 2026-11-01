"""Fused CPU/CUDA evaluators; batching is an internal measurement interface."""
from functools import lru_cache
from pathlib import Path
import os
import platform
import torch
if __package__:
    from .kc_string_forward import SPATIAL_ORDER, PARAMETER_COUNT, discrete_eigenvalues, pickup_weights
else:
    from kc_string_forward import SPATIAL_ORDER, PARAMETER_COUNT, discrete_eigenvalues, pickup_weights

@lru_cache(maxsize=2)
def kernel(cuda=False):
    from torch.utils.cpp_extension import load
    base=Path(__file__).parent
    flags=['/O2','/fp:strict'] if os.name=='nt' else ['-O3','-ffp-contract=off', '-mcpu=native' if platform.machine().lower() in ('arm64','aarch64') else '-march=native']
    if cuda: flags+=['-DWITH_CUDA']
    return load(name='kc_fused_cuda' if cuda else 'kc_fused_cpu',
                sources=[str(base/'kc_fused.cpp')]+([str(base/'kc_fused.cu')] if cuda else []),
                extra_cflags=flags,extra_cuda_cflags=['-O3','--fmad=false','-lineinfo'],
                with_cuda=cuda,verbose=False)

@lru_cache(maxsize=32)
def basis(intervals,device,modal,order=SPATIAL_ORDER):
    """Sine transform, dimensionless eigenvalues of h²·D_xx and midpoint pickup weights.

    Every kernel reads its spatial operator from ``eigen``; the caller must use
    the same ``order`` here and in ``initialize``. The returned weights are the
    nominal midpoint readout kept for the older experiments; the parameterised
    pickup comes from ``kc_string_forward.pickup_weights`` per batch item.
    """
    n=intervals-1
    indices=torch.arange(1,intervals,dtype=torch.float64,device=device)
    eigen=discrete_eigenvalues(intervals,order,device)
    if modal:
        transform=(2/intervals)**.5*torch.sin(torch.pi*indices[:,None]*indices[None,:]/intervals)
        weights=transform[n//2].contiguous()
    else:
        transform=None
        weights=torch.zeros(n,dtype=torch.float64,device=device);weights[n//2]=1
    ready=None
    if torch.device(device).type=='cuda':
        ready=torch.cuda.Event()
        ready.record(torch.cuda.current_stream(device))
    return transform,eigen,weights,ready

@torch.no_grad()
def evaluate(theta,intervals,rate,steps,stride,excitation,backend,tangents=True,chunk_steps=8192,threads=256,return_states=False,seeds=None,variant=0,order=SPATIAL_ORDER):
    if __package__:
        from .kc_string_forward import initialize, dual_multiply
    else:
        from kc_string_forward import initialize, dual_multiply
    single=theta.ndim==1
    parameters=theta[None] if single else theta
    if parameters.ndim!=2 or parameters.shape[1]!=PARAMETER_COUNT or parameters.shape[0]==0:
        raise ValueError(f'Expected ({PARAMETER_COUNT},) or internal batch (B,{PARAMETER_COUNT}) parameter multipliers.')
    spatial=backend!='modal'
    if parameters.shape[0]==1:
        state,coefficient,_=initialize(parameters[0],intervals,rate,excitation,order)
        states,coeff=state[None],coefficient[None]
        pickup=pickup_weights(parameters[0],intervals,spatial)[None]
    else:
        states,coeff,_=torch.vmap(lambda row: initialize(row,intervals,rate,excitation,order))(parameters)
        pickup=torch.vmap(lambda row: pickup_weights(row,intervals,spatial))(parameters)
    if seeds is not None:
        if not tangents:
            raise ValueError('Directional seeds require tangents=True.')
        seeds=torch.as_tensor(seeds,device=theta.device,dtype=theta.dtype)
        if seeds.ndim==2:
            seeds=seeds.unsqueeze(0).expand(parameters.shape[0],-1,-1)
        if seeds.ndim!=3 or seeds.shape[:2]!=(parameters.shape[0],PARAMETER_COUNT) or not 1<=seeds.shape[2]<=PARAMETER_COUNT:
            raise ValueError(f'Seeds must have shape ({PARAMETER_COUNT},S) or (B,{PARAMETER_COUNT},S), with 1 <= S <= {PARAMETER_COUNT}.')
        if not bool(torch.isfinite(seeds).all()):
            raise ValueError('Seeds must be finite.')
        # The eleven-column startup is inexpensive. Project its parameter columns
        # before the recurrence, so each kernel step advances only S directions.
        states=torch.cat((states[...,:1],torch.einsum('btid,bds->btis',states[...,1:],seeds)),dim=-1)
        coeff=torch.cat((coeff[...,:1],torch.einsum('bid,bds->bis',coeff[...,1:],seeds)),dim=-1)
        pickup=torch.cat((pickup[...,:1],torch.einsum('bid,bds->bis',pickup[...,1:],seeds)),dim=-1)
    # Absorb fixed spacing factors once, retaining their length derivatives.
    h2=coeff[:,0].clone()
    coeff[:,2]=dual_multiply(coeff[:,2],h2)
    coeff[:,3]=dual_multiply(dual_multiply(coeff[:,3],h2),h2)
    coeff[:,5]=dual_multiply(coeff[:,5],h2)
    coeff[:,7]=dual_multiply(coeff[:,7],h2)
    transform,eigen,weights,ready=basis(intervals,str(theta.device),backend=='modal',order)
    if ready is not None:
        stream=torch.cuda.current_stream(theta.device)
        stream.wait_event(ready)
        # Cached tensors can be evicted while another stream is using them.
        for tensor in (transform,eigen,weights):
            if tensor is not None: tensor.record_stream(stream)
    if not tangents:
        states=states[...,:1];coeff=coeff[...,:1];pickup=pickup[...,:1]
    if transform is not None:
        states=torch.einsum('ij,btjd->btid',transform.T,states)
    audio,jac,final=kernel(theta.is_cuda).simulate(states.contiguous(),coeff.contiguous(),eigen,pickup.contiguous(),
                    steps,stride,backend=='modal',chunk_steps,threads,order,variant)
    if return_states:
        if transform is not None:
            final=torch.einsum('ij,btjd->btid',transform,final)
        return (audio[0],jac[0],final[0]) if single else (audio,jac,final)
    return (audio[0],jac[0]) if single else (audio,jac)
