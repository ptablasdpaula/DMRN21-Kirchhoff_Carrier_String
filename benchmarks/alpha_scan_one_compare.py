"""Benchmark alpha-conditioned state scan vs exact adjoints (not parallel adjoints)."""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from benchmarks.exact_methods import BOX,GRID,filter_audio,measure,metadata,theta_of,varied_parameters
from benchmarks.reconstruction import relative_error
from experiments.loss_batch.grouped import grouped_loss_backward
from src.optimization import criterion_for
from src.synthesis import Renderer

def implementation_hash():
    h=hashlib.sha256()
    for path in ('experiments/parallel_alpha_scan_one/alpha_scan.cu','experiments/parallel_alpha_scan_one/reconstruct.py','experiments/scalar_cache/kc_scalar_cache.cu','experiments/scalar_cache/reconstruct.py'):
        h.update((ROOT/path).read_bytes())
    return h.hexdigest()

def baseline_hash():
    h=hashlib.sha256()
    for path in ('experiments/cached_reconstruction/kc_cached_reconstruct.cu','experiments/cached_reconstruction/reconstruct.py'):
        h.update((ROOT/path).read_bytes())
    return h.hexdigest()

def render(values,renderer,method,k,group):
    if method=='replay':from src.adjoint import simulate
    elif method=='cached':from experiments.cached_reconstruction.reconstruct import simulate
    elif method=='scan':from experiments.parallel_alpha_scan_one.reconstruct import simulate
    else:raise ValueError(method)
    args=(theta_of(values),GRID.intervals,GRID.integration_rate,round(renderer.duration*GRID.integration_rate),'multimode',k)
    return filter_audio(simulate(*args,group) if method=='scan' else simulate(*args),renderer)

def run(values,renderer,criterion,method,k,group,loss_group):
    p=values.detach().clone().requires_grad_();audio=render(p,renderer,method,k,group)
    if loss_group:
        loss=grouped_loss_backward(audio,criterion,loss_group)
    else:
        loss=torch.stack([criterion(row) for row in audio]).sum();loss.backward();loss=loss.detach()
    return audio.detach(),loss,p.grad.detach()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--validation',type=Path,required=True);p.add_argument('--durations',default='.25,4');p.add_argument('--batches',default='1,128');p.add_argument('--checkpoint',type=int,default=256);p.add_argument('--groups',default='8,16');p.add_argument('--loss-group',type=int,default=16);p.add_argument('--repeats',type=int,default=3);p.add_argument('--max-estimated-seconds',type=float,default=90);a=p.parse_args()
    assert torch.cuda.is_available();torch.set_num_threads(1);device=torch.device('cuda')
    impl=implementation_hash();evidence=json.loads(a.validation.read_text())
    test_hash=hashlib.sha256((ROOT/'test/alpha_scan_one_reconstruction.py').read_bytes()).hexdigest()
    if evidence.get('status')!='passed' or evidence.get('source_changed') or evidence.get('implementation_sha256')!=impl or evidence.get('source_sha256')!=metadata()['source_sha256'] or evidence.get('test_sha256')!=test_hash:raise RuntimeError('Stale or failed scan validation')
    helper_hash=hashlib.sha256((ROOT/'experiments/loss_batch/grouped.py').read_bytes()).hexdigest();script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();baseline=baseline_hash()
    result={'metadata':{**metadata(),'validation':str(a.validation),'implementation_sha256':impl,'baseline_sha256':baseline,'grouped_loss_sha256':helper_hash,'benchmark_sha256':script_hash,'status':'running'},'checks':[],'rows':[]};a.output.parent.mkdir(parents=True,exist_ok=True)
    def save():a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    def emit(row):result['rows'].append(row);save();print(json.dumps(row),flush=True)
    save(); previous={}
    for duration in map(float,a.durations.split(',')):
      target=Renderer(GRID,duration).synthesize(BOX.defaults()).detach().to(device);model=Renderer(GRID,duration,device=device);criterion=criterion_for(target,GRID.sample_rate)
      for batch in map(int,a.batches.split(',')):
        values=varied_parameters(batch).to(device)
        for group in map(int,a.groups.split(',')):
          key=(batch,group)
          if key in previous:
            predicted=previous[key][0]*duration/previous[key][1]
            if predicted>a.max_estimated_seconds:
                emit(dict(method='scan',batch=batch,duration=duration,checkpoint=a.checkpoint,group=group,status='skipped',reason='predicted runtime exceeds bound',predicted_seconds=predicted));continue
          ref=candidate=None
          try:
            ref=run(values,model,criterion,'replay',a.checkpoint,group,a.loss_group)
            started=time.perf_counter();candidate=run(values,model,criterion,'scan',a.checkpoint,group,a.loss_group);torch.cuda.synchronize();check_time=time.perf_counter()-started
            error=dict(audio_relative_l2=relative_error(candidate[0],ref[0]),loss_relative_l2=relative_error(candidate[1],ref[1]),scaled_gradient_relative_l2=relative_error(values*candidate[2],values*ref[2]))
            torch.testing.assert_close(candidate[0],ref[0],rtol=1e-6,atol=1e-10);torch.testing.assert_close(candidate[1],ref[1],rtol=5e-4,atol=1e-10);torch.testing.assert_close(values*candidate[2],values*ref[2],rtol=5e-4,atol=1e-8)
            result['checks'].append(dict(batch=batch,duration=duration,checkpoint=a.checkpoint,group=group,status='passed',first_check_seconds=check_time,**error))
          except (RuntimeError,AssertionError) as exc:
            result['checks'].append(dict(batch=batch,duration=duration,checkpoint=a.checkpoint,group=group,status='failed',error=str(exc)));save();print('FAILED',batch,duration,group,str(exc),flush=True);continue
          finally:del ref,candidate
          save()
          for method in ('replay','cached','scan'):
            try:
                row=dict(method=method,batch=batch,duration=duration,checkpoint=a.checkpoint,group=group,status='passed',**measure(lambda:run(values,model,criterion,method,a.checkpoint,group,a.loss_group),device,a.repeats))
                if method=='scan':previous[key]=(row['median_seconds'],duration)
                emit(row)
            except RuntimeError as exc:emit(dict(method=method,batch=batch,duration=duration,checkpoint=a.checkpoint,group=group,status='failed',error=str(exc)))
    now=implementation_hash()
    result['metadata']['source_changed_during_run']=(metadata()['source_sha256']!=result['metadata']['source_sha256'] or now!=impl or baseline_hash()!=baseline or hashlib.sha256((ROOT/'experiments/loss_batch/grouped.py').read_bytes()).hexdigest()!=helper_hash or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=script_hash)
    result['metadata']['status']='complete_with_failures' if any(r.get('status')=='failed' for r in result['checks']+result['rows']) else 'complete';save()
    if result['metadata']['source_changed_during_run'] or result['metadata']['status']!='complete':raise RuntimeError('Invalid scan benchmark; inspect JSON')
if __name__=='__main__':main()
