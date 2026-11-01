"""JAX modal forward-mode versus fused CUDA, same loss; bounded HPC benchmark.

JAX does the physical setup and recurrence; PyTorch supplies the identical
anti-alias filter and CeL loss. DLPack shares device buffers, and the timed JAX
path includes this hybrid bridge and the loss/backward. This is therefore not a
pure JAX loss benchmark. Output/coordinate gradients are checked versus replay.
"""
import argparse, hashlib, json, os, platform, statistics, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('JAX_ENABLE_X64','true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import jax
import jax.numpy as jnp
import torch
from experiments.jax_study.jax_modal import renderer, forward_jacobian
from benchmarks.exact_methods import BOX, GRID, filter_audio, varied_parameters, audio_for, measure, metadata, coordinates_and_seeds
from src.accelerated import evaluate
from benchmarks.reconstruction import render, relative_error
from src.optimization import criterion_for
from src.synthesis import Renderer

torch.set_num_threads(1)

class JaxAudio(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, executable, count):
        x=jax.dlpack.from_dlpack(values.detach().contiguous())
        audio,tangents=executable(x)
        audio.block_until_ready(); tangents.block_until_ready()
        output=torch.from_dlpack(audio)
        jac=torch.from_dlpack(tangents)
        ctx.save_for_backward(jac)
        ctx.num_values=values.shape[1]
        return output
    @staticmethod
    def backward(ctx, grad):
        jac,=ctx.saved_tensors
        a=torch.einsum('bt,btd->bd',grad.contiguous(),jac)
        result=grad.new_zeros((grad.shape[0],ctx.num_values))
        result[:,:a.shape[1]]=a
        return result,None,None

class SeededAudio(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, steps, count):
        theta,seeds=coordinates_and_seeds(values)
        audio,jac=evaluate(theta,GRID.intervals,GRID.integration_rate,steps,1,
                           "multimode","modal",True,seeds=seeds[:,:,:count])
        ctx.save_for_backward(jac);ctx.num_values=values.shape[1]
        return audio
    @staticmethod
    def backward(ctx, grad):
        jac,=ctx.saved_tensors
        a=torch.einsum("bt,btd->bd",grad.contiguous(),jac)
        result=grad.new_zeros((grad.shape[0],ctx.num_values));result[:,:a.shape[1]]=a
        return result,None,None

def render_jax(values, render_config, executable, count):
    return filter_audio(JaxAudio.apply(values,executable,count),render_config)

def check(values, renderer_, criterion, method, executable, count):
    p=values.detach().clone().requires_grad_()
    if method=='jax': audio=render_jax(p,renderer_,executable,count)
    else: audio=render(p,renderer_,'replay',256)
    losses=torch.stack([criterion(row) for row in audio])
    grad,=torch.autograd.grad(losses.sum(),p)
    return audio.detach(),losses.detach(),grad.detach()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--batches',default='1,8')
    parser.add_argument('--durations',default='.025,.25,4')
    parser.add_argument('--directions',default='2,6')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--max-predicted-seconds',type=float,default=60.)
    args=parser.parse_args()
    batches=[int(v) for v in args.batches.split(',')]
    durations=[float(v) for v in args.durations.split(',')]
    directions=[int(v) for v in args.directions.split(',')]
    assert torch.cuda.is_available() and jax.default_backend()=='gpu', 'Requires both JAX and PyTorch CUDA'
    result={'metadata':{**metadata(),'jax':jax.__version__,'jax_backend':jax.default_backend(),
         'method':'JAX physical setup + modal recurrence/JVP, PyTorch filter and CeL loss, DLPack bridge',
         'benchmark_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'implementation_sha256':hashlib.sha256((ROOT/'experiments/jax_study/jax_modal.py').read_bytes()).hexdigest(),
         'jax_peak_memory':'not measured; torch allocator stats exclude JAX allocations',
         'status':'running'},'checks':[],'rows':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save(): args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    def report(record): result['rows'].append(record);save(); print(json.dumps(record),flush=True)
    save()
    previous={}
    for duration in durations:
      target=Renderer(GRID,duration).synthesize(BOX.defaults()).detach().cuda()
      model=Renderer(GRID,duration,device='cuda')
      criterion=criterion_for(target,GRID.sample_rate)
      steps=round(duration*GRID.integration_rate)
      for batch in batches:
        values=varied_parameters(batch).cuda()
        for count in directions:
          key=(batch,count)
          estimate=previous[key]*duration/previous[(key,'duration')] if key in previous else 0
          if estimate>args.max_predicted_seconds:
            report(dict(method='jax_forward',batch=batch,duration=duration,directions=count,status='skipped',reason='estimated per-evaluation runtime exceeds bound',estimated_seconds=estimate))
            continue
          try:
            x=jax.dlpack.from_dlpack(values.contiguous())
            t=time.perf_counter()
            exe=jax.jit(lambda v:forward_jacobian(v,steps,count)).lower(x).compile()
            compile_seconds=time.perf_counter()-t
            t=time.perf_counter(); aa,gg=exe(x); gg.block_until_ready(); warmup=time.perf_counter()-t
            del aa,gg
            previous[key]=warmup;previous[(key,'duration')]=duration
            ref_audio,ref_loss,ref_grad=check(values,model,criterion,'replay',exe,count)
            out,loss,grad=check(values,model,criterion,'jax',exe,count)
            errors={'audio_relative_l2':relative_error(out,ref_audio),'loss_relative_l2':relative_error(loss,ref_loss),
              'scaled_gradient_relative_l2':relative_error((values*grad)[:,:count],(values*ref_grad)[:,:count])}
            torch.testing.assert_close(out,ref_audio,rtol=1e-5,atol=1e-9)
            torch.testing.assert_close(loss,ref_loss,rtol=1e-3,atol=1e-10)
            torch.testing.assert_close((values*grad)[:,:count],(values*ref_grad)[:,:count],rtol=1e-3,atol=1e-8)
            result['checks'].append(dict(batch=batch,duration=duration,directions=count,status='passed',**errors));save()
            def jaxfn():
                p=values.detach().clone().requires_grad_()
                audio=render_jax(p,model,exe,count)
                torch.stack([criterion(row) for row in audio]).sum().backward()
            timing=measure(jaxfn,torch.device('cuda'),args.repeats)
            timing.pop('peak_cuda_bytes',None)
            report(dict(method='jax_forward',batch=batch,duration=duration,directions=count,status='passed',compile_seconds=compile_seconds,first_run_seconds=warmup,**timing))
            def seededfn():
                p=values.detach().clone().requires_grad_()
                audio=filter_audio(SeededAudio.apply(p,steps,count),model)
                torch.stack([criterion(row) for row in audio]).sum().backward()
            timing=measure(seededfn,torch.device('cuda'),args.repeats)
            report(dict(method='cuda_seeded',batch=batch,duration=duration,directions=count,status='passed',**timing))
          except Exception as exc:
            result['checks'].append(dict(batch=batch,duration=duration,directions=count,status='failed',error=repr(exc)));save()
            print('FAILED',batch,duration,count,repr(exc),flush=True)
            continue
        # Same hardware, duration, loss, params and replay implementation.
        def replayfn():
          p=values.detach().clone().requires_grad_()
          torch.stack([criterion(row) for row in render(p,model,'replay',256)]).sum().backward()
        timing=measure(replayfn,torch.device('cuda'),args.repeats)
        report(dict(method='cuda_replay',batch=batch,duration=duration,status='passed',**timing))
    result['metadata']['status']='complete';save()

if __name__=='__main__': main()
