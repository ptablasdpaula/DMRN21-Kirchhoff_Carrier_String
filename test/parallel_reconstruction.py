"""Parallel checkpoint segment reconstruction: states, VJPs and safety.

The experiment parallelises state reconstruction but keeps adjoint propagation
between segments sequential. This suite checks both steps together against the
existing exact forward-Jacobian and replay-based reference implementations.
"""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from src.adjoint import simulate as replay
from src.kc_string_forward import ForwardString,PARAMETER_COUNT,initialize,dual_multiply,pickup_weights
from src.accelerated import basis
from src.parameters import ParameterBox
from src.grid import GridConfig
from experiments.reconstruction.reconstruct import simulate as sequential, kernel as seq_kernel
from experiments.parallel_segments.parallel_reconstruction import simulate as parallel, kernel as par_kernel
from benchmarks.exact_methods import digest as source_digest
from benchmarks.reconstruction import reconstruction_digest


def parallel_digest():
    h=hashlib.sha256()
    for name in ('parallel_reconstruction.cu','parallel_reconstruction.py'):
        path=ROOT/'experiments'/'parallel_segments'/name
        h.update(name.encode()+path.read_bytes())
    return h.hexdigest()


def probe(x,steps):
    t=torch.arange(steps+2,device=x.device,dtype=x.dtype)
    w=(torch.sin(.013*t)+.27*torch.cos(.0037*t))/(steps+2)
    if x.ndim==2:w=w[None,:]*(1+torch.arange(x.shape[0],device=x.device,dtype=x.dtype)[:,None]*.031)
    return w.contiguous()


def vjp(method,x,n,rate,steps,exc,checkpoint,weight,group=None):
    x=x.detach().clone().requires_grad_()
    if group is None:y=method(x,n,rate,steps,exc,checkpoint)
    else:y=method(x,n,rate,steps,exc,checkpoint,group)
    grad=torch.autograd.grad((y*weight).sum(),x)[0]
    return y.detach(),grad.detach()


def rel(a,b):return ((a-b).norm()/b.norm().clamp_min(1e-30)).item()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path)
    parser.add_argument('--quick',action='store_true',help='Skip full-length batch 128 case')
    args=parser.parse_args();assert torch.cuda.is_available(),'Run using an allocated Slurm GPU.'
    torch.set_num_threads(1);device='cuda';dtype=torch.float64
    src=source_digest();seq=reconstruction_digest();par=parallel_digest();test=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rows=[];box=ParameterBox();grid=GridConfig(box)
    nominal=torch.ones(PARAMETER_COUNT,dtype=dtype,device=device)
    # Off-centre pluck and pickup exercise the two position gradients throughout.
    varied=nominal.new_tensor([1.02,.98,1.01,1.03,1.01,.9,1.1,.95,1.02,.97])
    def check(name,theta,n,rate,steps,exc,checkpoint,group):
        w=probe(theta,steps)
        a,g=vjp(parallel,theta,n,rate,steps,exc,checkpoint,w,group)
        b,h=vjp(sequential,theta,n,rate,steps,exc,checkpoint,w)
        c,r=vjp(replay,theta,n,rate,steps,exc,checkpoint,w)
        torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-12)
        torch.testing.assert_close(a,c,rtol=1e-9,atol=1e-12)
        torch.testing.assert_close(g,h,rtol=5e-4,atol=1e-9)
        torch.testing.assert_close(g,r,rtol=5e-4,atol=1e-9)
        if theta.ndim==1:
            model=ForwardString(n,rate,excitation=exc,backend='modal')
            y,j=model.output_and_tangents(theta,steps/rate)
            torch.testing.assert_close(a,y,rtol=1e-8,atol=1e-12)
            torch.testing.assert_close(g,w@j,rtol=5e-4,atol=1e-9)
        row=dict(name=name,intervals=n,steps=steps,checkpoint=checkpoint,group=group,
                 batch=theta.shape[0] if theta.ndim==2 else 1,excitation=exc,
                 waveform_relative_l2=rel(a,c),gradient_relative_l2=rel(g,r),status='passed')
        rows.append(row);print(json.dumps(row),flush=True)
    # Zero steps, odd segment lengths, partial trailing segment, groups crossing
    # multiple blocks and group > number of segments.
    for n,rate in ((4,44100),(32,44100),(128,44100),(176,132300)):
        for exc in ('first_mode','multimode','raised_cosine'):
            for steps,k,g in ((0,7,3),(1,7,2),(17,7,2),(35,7,3),(37,7,4),(37,7,64)):
                check('segments',varied,n,rate,steps,exc,k,g)
    # Mixed batch verifies independent gradient propagation across every item.
    batch=torch.stack([nominal,varied,nominal*.99,varied*1.001])
    for group in (1,2,3,8):check('batch_isolation',batch,176,132300,101,'multimode',7,group)
    if not args.quick:
        batch128=torch.stack([nominal if i%2==0 else varied for i in range(128)])
        check('four_seconds_batch128',batch128,176,132300,4*132300,'multimode',256,8)
    for invalid in (0,65536):
        try:parallel(nominal,176,132300,1,'multimode',7,invalid)
        except ValueError:pass
        else:raise AssertionError(f'Invalid group {invalid} must be rejected')
    # Two independent streams exercise checkpoint/tape allocator lifetime.
    streams=[torch.cuda.Stream(),torch.cuda.Stream()];outputs=[]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            x=varied.detach().clone().requires_grad_();y=parallel(x,128,44100,35,'multimode',7,3)
            weights=probe(x,35);g=torch.autograd.grad((y*weights).sum(),x)[0]
            outputs.append((y.detach(),g.detach()))
    torch.cuda.synchronize()
    for left,right in zip(outputs[0],outputs[1]):torch.testing.assert_close(left,right,rtol=0,atol=0)
    rows.append(dict(name='independent_cuda_streams',status='passed'))
    # Directly compare sampled intermediate reconstructed states against a
    # forward trajectory checkpointed EVERY timestep, not just end boundaries.
    n=32;rate=44100;steps=37;k=7;B=3
    parameters=torch.stack([nominal,varied,nominal*.99])
    states,coeff,_=torch.vmap(lambda row: initialize(row,n,rate,'multimode'))(parameters)
    h2=coeff[:,0].clone();coeff[:,2]=dual_multiply(coeff[:,2],h2)
    coeff[:,3]=dual_multiply(dual_multiply(coeff[:,3],h2),h2)
    coeff[:,5]=dual_multiply(coeff[:,5],h2);coeff[:,7]=dual_multiply(coeff[:,7],h2)
    transform,eigen,_,event=basis(n,str(parameters.device),True)
    if event is not None:torch.cuda.current_stream(parameters.device).wait_event(event)
    states=torch.einsum('ij,btjd->btid',transform.T,states)
    primal=states[...,0].contiguous();cp=coeff[...,0].contiguous()
    # The replay kernel reads each item's own modal pickup weights (value column).
    pickup=torch.vmap(lambda row:pickup_weights(row,n,False))(parameters)[...,0].contiguous()
    _,checkpoints=seq_kernel().forward(primal,cp,eigen,pickup,steps,k)
    _,every=seq_kernel().forward(primal,cp,eigen,pickup,steps,1)
    total=(steps+k-1)//k
    for first,number in ((0,1),(1,3),(4,2)):
        tape,flags,drift=par_kernel().debug_segments(cp,eigen,checkpoints,steps,k,first,number)
        assert not bool(flags.any()) and bool(torch.isfinite(drift).all())
        for local in range(number):
            start=(first+local)*k;end=min(start+k,steps);length=end-start
            for j in range(length+1):
                torch.testing.assert_close(tape[local,:,j],every[start+j,:,0],rtol=1e-8,atol=1e-12)
            torch.testing.assert_close(tape[local,:,length+1],every[end,:,1],rtol=1e-8,atol=1e-12)
    rows.append(dict(name='all_intermediate_states_sampled_groups',status='passed',sample_groups=3))
    print('PASS intermediate reconstructed states',flush=True)
    # Inject low-level malformed states to verify explicit safety status bits.
    n=3;steps=7;k=7;B=1
    # Both the replay forward and the parallel backward take per-item (B, n)
    # pickup weights; the backward also returns d loss/d w as its third output.
    e=torch.ones(n,device=device,dtype=dtype);wt=torch.ones(1,n,device=device,dtype=dtype)
    state=torch.tensor([[[.001,.002,-.001],[.002,-.001,.003]]],device=device,dtype=dtype)
    c=torch.zeros(1,8,device=device,dtype=dtype);c[0,1]=.9;c[0,4]=-1.;c[0,6]=1.
    _,cp=seq_kernel().forward(state,c,e,wt,steps,k)
    upstream=torch.ones(1,steps+2,device=device,dtype=dtype)
    _,_,_,flags,drift=par_kernel().backward(c,e,wt,upstream,cp,steps,k,4)
    assert not bool(flags.any()) and bool(torch.isfinite(drift).all())
    bad=cp.clone();bad[-1,0,0,0]+=.01
    _,_,_,flags,_=par_kernel().backward(c,e,wt,upstream,bad,steps,k,4);assert int(flags[0,0])&8
    bad=cp.clone();bad[-1,0,0,0]=float('nan')
    _,_,_,flags,_=par_kernel().backward(c,e,wt,upstream,bad,steps,k,4);assert int(flags[0,0])&4
    badc=c.clone();badc[0,4]=0.;_,bad=seq_kernel().forward(state,badc,e,wt,steps,k)
    _,_,_,flags,_=par_kernel().backward(badc,e,wt,upstream,bad,steps,k,4);assert int(flags[0,0])&1
    badc=c.clone();badc[0,1]=0.;badc[0,4]=1.;badc[0,7]=1.
    special=torch.tensor([[[0.,0.,0.],[1.,0.,0.]]],device=device,dtype=dtype)
    _,bad=seq_kernel().forward(special,badc,e,wt,1,1)
    _,_,_,flags,_=par_kernel().backward(badc,e,wt,torch.ones(1,3,device=device,dtype=dtype),bad,1,1,2)
    assert int(flags[0,0])&2
    rows.append(dict(name='guard_singular_nonfinite_and_drift',status='passed'))
    theta=varied.detach().clone().requires_grad_();steps=2049;w=probe(theta,steps)
    direction=theta.new_tensor([.1,-.2,.15,-.1,.12,.08,.05,-.04,.06,-.05])
    def objective(t):return (parallel(t,176,132300,steps,'multimode',64,3)*w).sum()
    gradient=torch.autograd.grad(objective(theta),theta,create_graph=True)[0];eps=1e-5
    with torch.no_grad():fd=(objective(theta.detach()+eps*direction)-objective(theta.detach()-eps*direction))/(2*eps)
    torch.testing.assert_close(gradient@direction,fd,rtol=5e-4,atol=1e-9)
    try:torch.autograd.grad(gradient.sum(),theta)
    except RuntimeError:pass
    else:raise AssertionError('Second derivative must be rejected')
    rows.append(dict(name='finite_difference_and_higher_derivative',status='passed'))
    changed=(src!=source_digest() or seq!=reconstruction_digest() or par!=parallel_digest() or test!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if changed:raise RuntimeError('Source changed during parallel validation')
    report=dict(status='passed',source_sha256=src,reconstruction_sha256=seq,parallel_sha256=par,
                test_sha256=test,source_changed_during_run=changed,gpu=torch.cuda.get_device_name(),quick=args.quick,checks=rows)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('ALL PARALLEL RECONSTRUCTION CHECKS PASSED',flush=True)
if __name__=='__main__':main()
