"""Full JAX recurrence + filter + CeL, tested against the existing CUDA path.

FDTD uses JAX forward-mode for selected parameter directions; CeL uses JAX
reverse-mode for its output gradient. Prepared target is cached once like the
project's bound loss. Every timed case must pass waveform/loss/gradient checks.
"""
import argparse, hashlib, json, os, statistics, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('JAX_ENABLE_X64','true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import jax
import jax.numpy as jnp
import torch
from experiments.jax_study.jax_modal import forward_jacobian
from experiments.jax_loss_benchmark import jax_loss,prepare_target,anti_alias
from benchmarks.exact_methods import BOX, GRID, varied_parameters, measure, metadata,filter_audio
from benchmarks.reconstruction import render,relative_error
from benchmarks.jax_compare import SeededAudio
from src.optimization import criterion_for
from src.synthesis import Renderer

torch.set_num_threads(1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--durations',default='.025,.25,4');p.add_argument('--batches',default='1,8')
    p.add_argument('--directions',default='2,6');p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--max-predicted-seconds',type=float,default=20.)
    a=p.parse_args();batches=[int(x) for x in a.batches.split(',')];durations=[float(x) for x in a.durations.split(',')];counts=[int(x) for x in a.directions.split(',')]
    assert torch.cuda.is_available() and jax.default_backend()=='gpu', 'Requires CUDA on both JAX and PyTorch'
    result={'metadata':{**metadata(),'jax':jax.__version__,'jax_backend':jax.default_backend(),
      'method':'pure JAX modal recurrence + forward-mode tangents + filter + CeL reverse-mode',
      'jax_peak_memory':'not measured; torch allocation does not include JAX allocation',
      'benchmark_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'implementation_sha256':hashlib.sha256((ROOT/'experiments/jax_study/jax_modal.py').read_bytes()).hexdigest(),
      'loss_sha256':hashlib.sha256((ROOT/'experiments/jax_loss_benchmark.py').read_bytes()).hexdigest(),
      'status':'running'},'checks':[],'rows':[]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def save():a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    def add(row):result['rows'].append(row);save();print(json.dumps(row),flush=True)
    previous={};save()
    for duration in durations:
      target=Renderer(GRID,duration).synthesize(BOX.defaults()).detach().cuda()
      model=Renderer(GRID,duration,device='cuda'); criterion=criterion_for(target,GRID.sample_rate)
      tgt=jax.dlpack.from_dlpack(target.contiguous());prepared=jax.jit(prepare_target)(tgt)
      kernel=jax.dlpack.from_dlpack(model.filter.reshape(-1).contiguous())
      steps=round(duration*GRID.integration_rate)
      for batch in batches:
       values=varied_parameters(batch).cuda();x=jax.dlpack.from_dlpack(values.contiguous())
       for count in counts:
        key=(batch,count);estimate=previous[key]*duration/previous[(key,'duration')] if key in previous else 0
        if estimate>a.max_predicted_seconds:
         add(dict(method='pure_jax_forward',batch=batch,duration=duration,directions=count,status='skipped',estimated_seconds=estimate,reason='estimated evaluation time exceeds bound'));continue
        try:
         def eval_one(v,prepared,kernel):
          raw,jac=forward_jacobian(v,steps,count)
          loss,upstream=jax.value_and_grad(lambda waveform:jax_loss(waveform,prepared,kernel)*batch)(raw)
          gradient=jnp.einsum('bt,btd->bd',upstream,jac)
          return loss,gradient,raw
         t=time.perf_counter(); exe=jax.jit(eval_one).lower(x,prepared,kernel).compile(); compile_s=time.perf_counter()-t
         t=time.perf_counter();loss,g,raw=exe(x,prepared,kernel);g.block_until_ready();jax.block_until_ready((loss,raw));first=time.perf_counter()-t
         previous[key]=first;previous[(key,'duration')]=duration
         ref=values.detach().clone().requires_grad_();ref_audio=render(ref,model,'replay',256)
         ref_loss=torch.stack([criterion(row) for row in ref_audio]).sum();ref_grad,=torch.autograd.grad(ref_loss,ref)
         candidate=filter_audio(torch.from_dlpack(raw),model);actual_loss=torch.from_dlpack(loss);actual_g=torch.from_dlpack(g)
         errors={'audio_relative_l2':relative_error(candidate,ref_audio.detach()),
           'loss_relative_l2':relative_error(actual_loss,ref_loss.detach()),
           'scaled_gradient_relative_l2':relative_error(values[:,:count]*actual_g,(values*ref_grad)[:,:count])}
         torch.testing.assert_close(candidate,ref_audio.detach(),rtol=1e-5,atol=1e-9)
         torch.testing.assert_close(actual_loss,ref_loss.detach(),rtol=1e-3,atol=1e-10)
         torch.testing.assert_close(values[:,:count]*actual_g,(values*ref_grad)[:,:count],rtol=1e-3,atol=1e-8)
         result['checks'].append(dict(batch=batch,duration=duration,directions=count,status='passed',**errors));save()
         samples=[]
         for _ in range(a.repeats):
          t=time.perf_counter();out=exe(x,prepared,kernel);jax.block_until_ready(out);samples.append(time.perf_counter()-t)
         add(dict(method='pure_jax_forward',batch=batch,duration=duration,directions=count,status='passed',compile_seconds=compile_s,first_run_seconds=first,median_seconds=statistics.median(samples),repetitions_seconds=samples))
         def cuda_seeded():
          p=values.detach().clone().requires_grad_();raw=SeededAudio.apply(p,steps,count);audio=filter_audio(raw,model)
          torch.stack([criterion(row) for row in audio]).sum().backward()
         timing=measure(cuda_seeded,torch.device('cuda'),a.repeats)
         add(dict(method='cuda_seeded',batch=batch,duration=duration,directions=count,status='passed',**timing))
        except Exception as exc:
         result['checks'].append(dict(batch=batch,duration=duration,directions=count,status='failed',error=repr(exc)));save();print('FAILED',batch,duration,count,repr(exc),flush=True)
       def replay():
        p=values.detach().clone().requires_grad_();out=render(p,model,'replay',256)
        torch.stack([criterion(row) for row in out]).sum().backward()
       timing=measure(replay,torch.device('cuda'),a.repeats)
       add(dict(method='cuda_replay',batch=batch,duration=duration,status='passed',**timing))
    result['metadata']['status']='complete';save()
if __name__=='__main__':main()
