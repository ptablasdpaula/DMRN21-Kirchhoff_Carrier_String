"""GPU correctness tests for the recorded-alpha conditional state scan."""
import argparse, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.adjoint import simulate as replay
from src.kc_string_forward import PARAMETER_COUNT,initialize,dual_multiply,pickup_weights
from src.accelerated import basis
from experiments.scalar_cache.reconstruct import kernel as forward_kernel
from experiments.parallel_alpha_scan_one.reconstruct import kernel,simulate
from benchmarks.exact_methods import digest

def rel(a,b):return (torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-30)).item()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--long-batch',action='store_true');a=p.parse_args()
    assert torch.cuda.is_available();torch.set_num_threads(1)
    def implementation_hash():
        h=hashlib.sha256()
        for path in ('experiments/parallel_alpha_scan_one/alpha_scan.cu','experiments/parallel_alpha_scan_one/reconstruct.py','experiments/scalar_cache/kc_scalar_cache.cu','experiments/scalar_cache/reconstruct.py'):
            h.update((ROOT/path).read_bytes())
        return h.hexdigest()
    src=digest();impl=implementation_hash();test_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    nominal=torch.ones(PARAMETER_COUNT,device='cuda',dtype=torch.float64);varied=nominal.new_tensor([1.02,.98,1.01,1.03,1.01,.9,1.1,.95,1.02,.97])
    rows=[]
    def check(name,x,n,steps,k,group):
        t=torch.arange(steps+2,device=x.device,dtype=x.dtype);w=(torch.sin(.013*t)+.27*torch.cos(.0037*t))/(steps+2)
        if x.ndim==2:w=w[None,:]*(1+torch.arange(x.shape[0],device=x.device,dtype=x.dtype)[:,None]*.001)
        def run(fn):
            v=x.detach().clone().requires_grad_();y=fn(v);g,=torch.autograd.grad((y*w).sum(),v);return y.detach(),g.detach()
        y,g=run(lambda v:simulate(v,n,132300,steps,'multimode',k,group));yr,gr=run(lambda v:replay(v,n,132300,steps,'multimode',k))
        torch.testing.assert_close(y,yr,rtol=1e-9,atol=1e-12);torch.testing.assert_close(g,gr,rtol=5e-4,atol=1e-9)
        row=dict(name=name,batch=x.shape[0] if x.ndim==2 else 1,steps=steps,checkpoint=k,group=group,waveform_relative_l2=rel(y,yr),gradient_relative_l2=rel(g,gr),status='passed');rows.append(row);print(json.dumps(row),flush=True)
    for n in (4,32,176):
      for steps,k,g in ((0,7,3),(1,7,2),(17,7,2),(35,7,3),(37,7,4),(257,64,3)):
        check('odd_and_small',varied,n,steps,k,g)
    batch=torch.stack([nominal,varied,nominal*.99,varied*1.001]);check('mixed_batch',batch,176,1025,256,4)
    check('four_seconds_single',varied,176,4*132300,256,16)
    if a.long_batch:
        batch128=torch.stack([nominal if i%2==0 else varied for i in range(128)])
        check('four_seconds_batch128',batch128,176,4*132300,256,8)
    # Compare every recovered state with forward checkpoints at each time step.
    n=32;steps=37;k=7;parameters=torch.stack([nominal,varied,nominal*.99]);B=len(parameters)
    states,coeff,_=torch.vmap(lambda x:initialize(x,n,132300,'multimode'))(parameters)
    h2=coeff[:,0].clone();coeff[:,2]=dual_multiply(coeff[:,2],h2);coeff[:,3]=dual_multiply(dual_multiply(coeff[:,3],h2),h2);coeff[:,5]=dual_multiply(coeff[:,5],h2);coeff[:,7]=dual_multiply(coeff[:,7],h2)
    transform,eigen,weights,event=basis(n,str(parameters.device),True)
    if event is not None:torch.cuda.current_stream(parameters.device).wait_event(event)
    states=torch.einsum('ij,btjd->btid',transform.T,states);state=states[...,0].contiguous();coeff=coeff[...,0].contiguous()
    # Both the scalar-cache forward and the alpha-scan backward read each item's
    # own modal pickup weights (value column); the backward also returns d loss/d w.
    pickup=torch.vmap(lambda row:pickup_weights(row,n,False))(parameters)[...,0].contiguous()
    _,cp,cache=forward_kernel().forward(state,coeff,eigen,pickup,steps,k,1)
    _,every,_=forward_kernel().forward(state,coeff,eigen,pickup,steps,1,1)
    for first,count in ((0,1),(1,3),(4,2)):
        tape,flags,drift=kernel().debug_segments(coeff,eigen,cp,cache,steps,k,first,count)
        assert not bool(flags.any()) and bool(torch.isfinite(drift).all())
        for local in range(count):
            start=(first+local)*k;length=min(k,steps-start)
            for j in range(length+1):torch.testing.assert_close(tape[local,:,j],every[start+j,:,0],rtol=1e-8,atol=1e-12)
            torch.testing.assert_close(tape[local,:,length+1],every[start+length,:,1],rtol=1e-8,atol=1e-12)
    bad=cache.clone();bad[0,2,0]=float('nan');_,flags,_=kernel().debug_segments(coeff,eigen,cp,bad,steps,k,0,1);assert int(flags[0,0])&4
    bad=cp.clone();bad[1,0,0,0]+=.01;_,flags,_=kernel().debug_segments(coeff,eigen,bad,cache,steps,k,0,1);assert int(flags[0,0])&8
    bad_grad=torch.ones(B,steps+2,device=state.device,dtype=state.dtype);bad_grad[0,3]=float('nan')
    _,_,_,flags,_=kernel().backward(coeff,eigen,pickup,bad_grad,cp,cache,steps,k,3);assert bool(((flags[:,0]&4)!=0).any())
    rows.append(dict(name='interior_states_and_guards',status='passed'))
    for k,g in ((0,1),(257,1),(256,0)):
        try:simulate(nominal,176,132300,1,'multimode',k,g)
        except ValueError:pass
        else:raise AssertionError('invalid bounds')
    current_impl=implementation_hash()
    result=dict(status='passed',source_sha256=src,implementation_sha256=impl,test_sha256=test_hash,source_changed=(digest()!=src or current_impl!=impl or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=test_hash),rows=rows)
    assert not result['source_changed'];a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print('ALL ALPHA SCAN VALIDATION PASSED',flush=True)

if __name__=='__main__':main()
