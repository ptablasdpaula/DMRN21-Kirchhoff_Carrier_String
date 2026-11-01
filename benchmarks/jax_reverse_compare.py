"""JAX full reverse-mode, with/without per-step checkpoint; HPC experiment."""
import argparse, hashlib, json, os, statistics, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('JAX_ENABLE_X64','true');os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import jax
import torch
from experiments.jax_study.jax_modal import renderer
from experiments.jax_study.jax_modal_remat import renderer as remat_renderer
from experiments.jax_loss_benchmark import jax_loss,prepare_target
from benchmarks.exact_methods import BOX, GRID, varied_parameters, measure,metadata,filter_audio
from benchmarks.reconstruction import render,relative_error
from src.optimization import criterion_for
from src.synthesis import Renderer

torch.set_num_threads(1)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
 p.add_argument('--durations',default='.025,.25,4');p.add_argument('--batches',default='1,8')
 p.add_argument('--repeats',type=int,default=3);p.add_argument('--max-predicted-seconds',type=float,default=20.)
 a=p.parse_args();batches=[int(x) for x in a.batches.split(',')];durations=[float(x) for x in a.durations.split(',')]
 assert torch.cuda.is_available() and jax.default_backend()=='gpu'
 d=jax.devices()[0]
 result={'metadata':{**metadata(),'jax':jax.__version__,'method':'JAX full reverse-mode through lax.scan and native JAX filter/CeL',
 'benchmark_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'jax_memory_note':'JAX device cumulative peak across previous cases, not per case; excludes Torch', 'status':'running'},'checks':[],'rows':[]}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 def save(): a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 def add(row): result['rows'].append(row);save();print(json.dumps(row),flush=True)
 previous={};save()
 for duration in durations:
  target=Renderer(GRID,duration).synthesize(BOX.defaults()).detach().cuda()
  model=Renderer(GRID,duration,device='cuda');criterion=criterion_for(target,GRID.sample_rate)
  prepared=jax.jit(prepare_target)(jax.dlpack.from_dlpack(target.contiguous()))
  kernel=jax.dlpack.from_dlpack(model.filter.reshape(-1).contiguous());steps=round(duration*GRID.integration_rate)
  for batch in batches:
   values=varied_parameters(batch).cuda();x=jax.dlpack.from_dlpack(values.contiguous())
   for name,solver in [('plain',renderer),('step_remat',remat_renderer)]:
    key=(batch,name);estimate=previous[key]*duration/previous[(key,'duration')] if key in previous else 0
    if estimate>a.max_predicted_seconds:
     add(dict(method='jax_reverse_'+name,batch=batch,duration=duration,status='skipped',estimated_seconds=estimate,reason='projected time exceeds bound'));continue
    # Hard bound: never attempt long unmeasured B8 due to reverse tape size.
    if batch>1 and duration>0.25:
     add(dict(method='jax_reverse_'+name,batch=batch,duration=duration,status='skipped',reason='reverse tape memory unmeasured for long batches'));continue
    try:
     def objective(v,prep,kernel_):
      raw=solver(v,steps); loss=jax_loss(raw,prep,kernel_)*batch
      return loss,raw
     f=jax.jit(jax.value_and_grad(objective,has_aux=True))
     t=time.perf_counter();exe=f.lower(x,prepared,kernel).compile();compile_s=time.perf_counter()-t
     t=time.perf_counter();(loss,raw),gradient=exe(x,prepared,kernel);jax.block_until_ready((loss,raw,gradient));first=time.perf_counter()-t
     previous[key]=first;previous[(key,'duration')]=duration
     ref=values.detach().clone().requires_grad_();ref_audio=render(ref,model,'replay',256)
     ref_loss=torch.stack([criterion(row) for row in ref_audio]).sum();ref_grad,=torch.autograd.grad(ref_loss,ref)
     candidate=filter_audio(torch.from_dlpack(raw),model);actual_loss=torch.from_dlpack(loss);actual_grad=torch.from_dlpack(gradient)
     errors={'audio_relative_l2':relative_error(candidate,ref_audio.detach()),'loss_relative_l2':relative_error(actual_loss,ref_loss.detach()),'scaled_gradient_relative_l2':relative_error(values*actual_grad,values*ref_grad)}
     torch.testing.assert_close(candidate,ref_audio.detach(),rtol=1e-5,atol=1e-9)
     torch.testing.assert_close(actual_loss,ref_loss.detach(),rtol=1e-3,atol=1e-10)
     torch.testing.assert_close(values*actual_grad,values*ref_grad,rtol=1e-3,atol=1e-8)
     result['checks'].append(dict(batch=batch,duration=duration,method=name,status='passed',**errors));save()
     times=[]
     for _ in range(a.repeats):
      t=time.perf_counter();out=exe(x,prepared,kernel);jax.block_until_ready(out);times.append(time.perf_counter()-t)
     mem=d.memory_stats()
     add(dict(method='jax_reverse_'+name,batch=batch,duration=duration,status='passed',compile_seconds=compile_s,first_run_seconds=first,median_seconds=statistics.median(times),repetitions_seconds=times,jax_device_memory_stats=mem))
    except Exception as e:
     result['checks'].append(dict(batch=batch,duration=duration,method=name,status='failed',error=repr(e)));save();print('FAILED',batch,duration,name,repr(e),flush=True)
   def replay():
    p=values.detach().clone().requires_grad_();out=render(p,model,'replay',256)
    torch.stack([criterion(row) for row in out]).sum().backward()
   timing=measure(replay,torch.device('cuda'),a.repeats)
   add(dict(method='cuda_replay',batch=batch,duration=duration,status='passed',**timing))
 result['metadata']['status']='complete';save()
if __name__=='__main__':main()
