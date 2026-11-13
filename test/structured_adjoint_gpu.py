"""Physical-grid structured suffix-adjoint checks and bounded-scale timings."""
import argparse, hashlib, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.kc_string_forward import initialize,dual_multiply,pickup_weights
from src.accelerated import basis
from src.adjoint import simulate,kernel
from benchmarks.exact_methods import metadata
from experiments.structured_adjoint.physical import physical_problem
from experiments.structured_adjoint.structured import structured_vjp
from experiments.structured_adjoint.autograd import simulate as structured_simulate


def rel(a,b):return ((a-b).norm()/b.norm().clamp_min(1e-30)).item()


def run_case(batch,chunk,window,repeats,steps=None):
    device='cuda';intervals=176;rate=132300
    steps=chunk*window if steps is None else steps
    p,state_derivs,theta=physical_problem(batch,intervals,rate,steps,device)
    state,coeff,_=torch.vmap(lambda row:initialize(row,intervals,rate,'multimode'))(theta)
    h2=coeff[:,0].clone()
    coeff[:,2]=dual_multiply(coeff[:,2],h2)
    coeff[:,3]=dual_multiply(dual_multiply(coeff[:,3],h2),h2)
    coeff[:,5]=dual_multiply(coeff[:,5],h2)
    coeff[:,7]=dual_multiply(coeff[:,7],h2)
    transform,eigen,_,ready=basis(intervals,device,True)
    if ready is not None:torch.cuda.current_stream().wait_event(ready)
    state=torch.einsum('ij,btjd->btid',transform.T,state)
    # The replay kernel reads each item's own modal pickup weights (value column).
    pickup=torch.vmap(lambda row:pickup_weights(row,intervals,False))(theta)[...,0].contiguous()
    audio,checks=kernel().forward(state[...,0].contiguous(),coeff[...,0].contiguous(),eigen,pickup,steps,chunk)
    x=theta.clone().detach().requires_grad_()
    probe=torch.nn.functional.pad(p['upstream'],(2,0))
    ref_audio=simulate(x,intervals,rate,steps,'multimode',chunk)
    (ref_grad,)=torch.autograd.grad((ref_audio*probe).sum(),x)
    zero=torch.zeros_like(theta)
    def candidate():
        initial,g,gw,detail=structured_vjp(p,zero,checks,chunk,window)
        g=g+torch.einsum('bti,btid->bd',initial.reshape(batch,2,intervals-1),state_derivs)
        # Pickup position through the readout tangents; the probe is zero on the
        # two startup samples, so the scan's d loss/d w is complete here.
        g=g+torch.einsum('bi,bid->bd',gw,p['dw'])
        return g,detail
    got,detail=candidate()
    torch.testing.assert_close(audio,ref_audio,rtol=1e-9,atol=1e-12)
    torch.testing.assert_close(got,ref_grad,rtol=5e-4,atol=1e-9)
    wrapper_error=None
    if batch==1 or chunk==16:
        x2=theta.clone().detach().requires_grad_()
        # Nonzero initial-sample gradients exercise the initial-state chain.
        probe2=probe.clone();probe2[:,0]=.17;probe2[:,1]=-.11
        structured_audio=structured_simulate(x2,intervals,rate,steps,'multimode',chunk,window)
        (grad2,)=torch.autograd.grad((structured_audio*probe2).sum(),x2)
        x3=theta.clone().detach().requires_grad_()
        ref2=simulate(x3,intervals,rate,steps,'multimode',chunk)
        (grad3,)=torch.autograd.grad((ref2*probe2).sum(),x3)
        torch.testing.assert_close(grad2,grad3,rtol=5e-4,atol=1e-9)
        wrapper_error=rel(grad2,grad3)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
    for _ in range(repeats):
        t=time.perf_counter();g,details=candidate();torch.cuda.synchronize();times.append(time.perf_counter()-t)
    return dict(batch=batch,chunk=chunk,window=window,steps=steps,status='passed',
      audio_relative_l2=rel(audio,ref_audio.detach()),gradient_relative_l2=rel(got,ref_grad.detach()),
      boundary_error=max(z['boundary_error'] for z in detail),adjoint_error=max(z['adjoint_error'] for z in detail),wrapper_gradient_relative_l2=wrapper_error,
      rank=max(z['rank'] for z in detail),seconds=times,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
      estimated_four_second_backward=sorted(times)[len(times)//2]*529200/steps,
      estimate_note='Linear runtime extrapolation of eager-PyTorch short prototype only; not a measurement')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--batches',default='1,128');p.add_argument('--chunks',default='16,32,64');p.add_argument('--window',type=int,default=4);p.add_argument('--repeats',type=int,default=2);p.add_argument('--max-full-seconds',type=float,default=180.);a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('Requires allocated GPU')
    torch.set_num_threads(1)
    files=[ROOT/'experiments/structured_adjoint/structured.py',ROOT/'experiments/structured_adjoint/autograd.py',
           ROOT/'experiments/structured_adjoint/physical.py',ROOT/'experiments/parallel_segments/toy_scan.py',
           ROOT/'test/structured_adjoint.py',Path(__file__)]
    def fingerprint():
        h=hashlib.sha256()
        for file in files:h.update(str(file.relative_to(ROOT)).encode()+file.read_bytes())
        return h.hexdigest()
    base_meta=metadata(); signature=fingerprint()
    result={'metadata':{**base_meta,'structured_suite_sha256':signature,'status':'running'},'cases':[]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def save():a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    save()
    for batch in map(int,a.batches.split(',')):
        for chunk in map(int,a.chunks.split(',')):
            try: row=run_case(batch,chunk,a.window,a.repeats)
            except (RuntimeError,AssertionError,ValueError) as exc:
                row=dict(batch=batch,chunk=chunk,window=a.window,status='failed',error=str(exc))
                if isinstance(exc,torch.cuda.OutOfMemoryError):torch.cuda.empty_cache()
            result['cases'].append(row);save();print(json.dumps(row),flush=True)
            if row['status']=='passed':
                estimate=row['estimated_four_second_backward']
                if estimate<=a.max_full_seconds:
                    full=dict(batch=batch,chunk=chunk,window=a.window,kind='four_second',status='eligible',
                              reason='Short timing permits a bounded full-loss comparison in the follow-up worker',estimated_seconds=estimate)
                else:
                    full=dict(batch=batch,chunk=chunk,window=a.window,kind='four_second',status='skipped',
                              reason='Short-window estimate exceeds runtime guard; full result not measured',
                              estimated_seconds=estimate)
                result['cases'].append(full);save();print(json.dumps(full),flush=True)
    result['metadata']['status']='complete' if all(row['status'] in ('passed','skipped','eligible') for row in result['cases']) else 'failed'
    result['metadata']['source_changed']=(fingerprint()!=signature or metadata()['source_sha256']!=base_meta['source_sha256'])
    save()
    if result['metadata']['status']!='complete' or result['metadata']['source_changed']:raise RuntimeError('Failed validation or source changed')
if __name__=='__main__':main()
