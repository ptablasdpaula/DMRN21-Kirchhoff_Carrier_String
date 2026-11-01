"""Acceptance checks for checkpointed state reconstruction (Slurm GPU worker).

Compares the candidate against the replay discrete adjoint and explicit native
forward Jacobian, then checks selected gradients by finite differences. Stress
cases are reported individually: an explicitly rejected unsafe inverse is not
reported as a successful comparison and cannot pass the benchmark gate.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from src.adjoint import simulate as replay
from src.kc_string_forward import ForwardString, nominal_parameters
from src.parameters import ParameterBox
from src.grid import GridConfig
from experiments.reconstruction.reconstruct import simulate as reconstruct, kernel
from benchmarks.exact_methods import digest as source_digest
from benchmarks.reconstruction import reconstruction_digest



def simulate_vjp(method, x, n, rate, steps, excitation, checkpoint, probe):
    x=x.detach().clone().requires_grad_()
    audio=method(x, n, rate, steps, excitation, checkpoint)
    grad=torch.autograd.grad((audio*probe).sum(), x)[0]
    return audio.detach(),grad.detach()


def probe_for(x,steps):
    t=torch.arange(steps+2,device=x.device,dtype=x.dtype)
    weights=(torch.sin(.013*t)+.27*torch.cos(.0037*t))/(steps+2)
    if x.ndim==2:
        weights=weights[None,:]*(1+torch.arange(x.shape[0],device=x.device,dtype=x.dtype)[:,None]*.031)
    return weights.contiguous()


def relative(a,b):
    return ((a-b).norm()/b.norm().clamp_min(1e-30)).item()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    parser.add_argument('--quick',action='store_true',help='Skip full-length and batch stress tests')
    args=parser.parse_args()
    assert torch.cuda.is_available(), 'Run on a Slurm GPU worker.'
    torch.set_num_threads(1)
    source_hash=source_digest();experiment_hash=reconstruction_digest();test_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();checks=[]
    device='cuda';dtype=torch.float64
    box=ParameterBox();grid=GridConfig(box);nominal=nominal_parameters(device)
    def multiplier(physical):
        return box.to_material(physical.to(device))/nominal
    def physical(amplitude=None,sigma0=None,sigma1=None,frequency=None,inharmonicity=None):
        p=box.defaults().clone()
        for index,value in ((2,amplitude),(4,sigma0),(5,sigma1),(0,frequency),(1,inharmonicity)):
            if value is not None:p[index]=value
        box.validate(p)
        return multiplier(p)

    def check(name, theta, n, rate, steps, excitation, checkpoint, native=False):
        w=probe_for(theta,steps)
        a,g=simulate_vjp(reconstruct,theta,n,rate,steps,excitation,checkpoint,w)
        b,h=simulate_vjp(replay,theta,n,rate,steps,excitation,checkpoint,w)
        # Current fused modal implementation independently propagates forward AD.
        if theta.ndim==1:
            model=ForwardString(n,rate,excitation=excitation,backend='modal')
            f,j=model.output_and_tangents(theta,steps/rate)
            direct=w@j
            torch.testing.assert_close(b,f,rtol=1e-8,atol=1e-12)
            torch.testing.assert_close(h,direct,rtol=5e-4,atol=1e-9)
            if native:
                cpu=ForwardString(n,rate,excitation=excitation)
                f,j=cpu.output_and_tangents(theta.cpu(),steps/rate)
                torch.testing.assert_close(a.cpu(),f,rtol=1e-6,atol=1e-10)
                torch.testing.assert_close(g.cpu(),w.cpu()@j,rtol=5e-4,atol=1e-8)
        torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-12)
        torch.testing.assert_close(g,h,rtol=5e-4,atol=1e-9)
        item=dict(name=name,intervals=n,rate=rate,steps=steps,checkpoint=checkpoint,
                  batch=theta.shape[0] if theta.ndim==2 else 1,excitation=excitation,
                  waveform_relative_l2=relative(a,b),gradient_relative_l2=relative(g,h),status='passed')
        checks.append(item);print(json.dumps(item),flush=True)
        return g

    # The perceptual box leaves the pluck and pickup at their nominal centre;
    # the off-centre case moves both so their gradients are nonzero.
    off_centre=physical();off_centre[8]=.9;off_centre[9]=1.1
    cases=(('reference',physical()),('low_amplitude',physical(amplitude=.0005)),
           ('high_amplitude',physical(amplitude=.003)),
           ('undamped',physical(sigma0=0.,sigma1=0.)),
           ('high_damping',physical(sigma0=2.,sigma1=.001)),
           ('stability_corner',physical(amplitude=.003,sigma0=2.,sigma1=.001,frequency=100.,inharmonicity=.001)),
           ('off_centre_pluck_pickup',off_centre))
    base=off_centre
    for n,rate in ((4,44100),(32,44100),(128,44100),(176,132300)):
        for excitation in ('first_mode','multimode','raised_cosine'):
            for steps,checkpoint in ((0,1),(1,1),(17,7),(35,64)):
                check('boundaries',base,n,rate,steps,excitation,checkpoint,native=(n==32 and steps==17))
    # Probe higher damping and nonlinear amplitudes for each selected interval.
    for name,x in cases:
        for checkpoint in (16,64,256,1024):
            try:
                check(name,x,grid.intervals,grid.integration_rate,2049,'multimode',checkpoint)
            except RuntimeError as error:
                if checkpoint<1024 or 'State reconstruction failed safety checks' not in str(error):raise
                rejected=dict(name=name,checkpoint=checkpoint,status='guard_rejected',reason=str(error))
                checks.append(rejected);print(json.dumps(rejected),flush=True)
    # Non-identical items exercise batch checkpoint strides and isolation.
    selected=cases if not args.quick else cases[:3]
    batch=torch.stack([item[1] for item in selected])
    check('mixed_batch',batch,grid.intervals,grid.integration_rate,1025,'multimode',256)
    if not args.quick:
        for name,x in cases:
            check('full_'+name,x,grid.intervals,grid.integration_rate,4*grid.integration_rate,
                  'multimode',256,native=(name in ('reference','high_damping')))
        check('full_mixed_batch',batch,grid.intervals,grid.integration_rate,4*grid.integration_rate,
              'multimode',256)
        batch128=torch.stack([cases[i%len(cases)][1] for i in range(128)])
        check('full_batch_128',batch128,grid.intervals,grid.integration_rate,4*grid.integration_rate,
              'multimode',256)
    # Exercise low-level singularity, nonfinite and boundary-drift checks.
    # These tests deliberately corrupt internal input after valid forward runs.
    # Per-item (B, n) pickup weights; backward returns (bar, grads, gw, status, drift).
    k=kernel();n=3;steps=7;chunk=7
    e=torch.ones(n,dtype=dtype,device=device);wt=torch.ones(1,n,dtype=dtype,device=device)
    state=torch.tensor([[[.001,.002,-.001],[.002,-.001,.003]]],dtype=dtype,device=device)
    c=torch.zeros(1,8,dtype=dtype,device=device);c[0,1]=.9;c[0,4]=-1.;c[0,6]=1.
    _,checkpoints=k.forward(state,c,e,wt,steps,chunk)
    upstream=torch.ones(1,steps+2,dtype=dtype,device=device)
    _,_,gw,flags,drift=k.backward(c,e,wt,upstream,checkpoints,steps,chunk)
    assert int(flags[0])==0 and torch.isfinite(drift).all() and gw.shape==(1,n) and torch.isfinite(gw).all()
    corrupted=checkpoints.clone();corrupted[-1,0,0,0]+=.01
    _,_,_,flags,_=k.backward(c,e,wt,upstream,corrupted,steps,chunk)
    assert int(flags[0]) & 8, f'Boundary drift was not detected: {flags}'
    corrupted=checkpoints.clone();corrupted[-1,0,0,0]=float('nan')
    _,_,_,flags,_=k.backward(c,e,wt,upstream,corrupted,steps,chunk)
    assert int(flags[0]) & 4, f'Nonfinite state was not detected: {flags}'
    singular=c.clone();singular[0,4]=0.
    _,singular_checkpoints=k.forward(state,singular,e,wt,steps,chunk)
    _,_,_,flags,_=k.backward(singular,e,wt,upstream,singular_checkpoints,steps,chunk)
    assert int(flags[0]) & 1, f'Singular inverse diagonal was not detected: {flags}'
    # q^T P^-1 q=1 => rank-one inverse has zero denominator.
    singular=c.clone();singular[0,1]=0.;singular[0,4]=1.;singular[0,7]=1.
    special=torch.tensor([[[0.,0.,0.],[1.,0.,0.]]],dtype=dtype,device=device)
    _,singular_checkpoints=k.forward(special,singular,e,wt,1,1)
    _,_,_,flags,_=k.backward(singular,e,wt,torch.ones(1,3,dtype=dtype,device=device),singular_checkpoints,1,1)
    assert int(flags[0]) & 2, f'Singular rank-one denominator was not detected: {flags}'
    checks.append(dict(name='guard_singular_nonfinite_and_drift',status='passed'))
    # Independent scalar finite differences and explicit higher derivative rejection.
    theta=off_centre.detach().clone().requires_grad_();steps=2049;direction=theta.new_tensor([.1,-.2,.15,-.1,.12,.08,.05,-.04,.06,-.05])
    w=probe_for(theta,steps)
    def objective(v): return (reconstruct(v,grid.intervals,grid.integration_rate,steps,'multimode',256)*w).sum()
    first=torch.autograd.grad(objective(theta),theta,create_graph=True)[0]
    eps=1e-5
    with torch.no_grad(): fd=(objective(theta.detach()+eps*direction)-objective(theta.detach()-eps*direction))/(2*eps)
    torch.testing.assert_close(first@direction,fd,rtol=5e-4,atol=1e-9)
    try:torch.autograd.grad(first.sum(),theta)
    except RuntimeError:pass
    else:raise AssertionError('Second derivatives must be explicitly rejected')
    checks.append(dict(name='finite_difference_and_higher_derivative',status='passed',finite=float(fd),analytic=float(first@direction)))
    output=dict(status='passed',source_sha256=source_hash,reconstruction_sha256=experiment_hash,test_sha256=test_hash,
                source_changed_during_run=(source_digest()!=source_hash or reconstruction_digest()!=experiment_hash or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=test_hash),
                device=torch.cuda.get_device_name(),torch=torch.__version__,quick=args.quick,checks=checks)
    if output['source_changed_during_run']:raise RuntimeError('Source changed during reconstruction validation.')
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print('ALL RECONSTRUCTION CHECKS PASSED',flush=True)

if __name__=='__main__':main()
