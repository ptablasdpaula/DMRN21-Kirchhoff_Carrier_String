"""Scalar cache sweep, preserving the original FP64 nonlinear recurrence."""
import argparse, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from benchmarks.exact_methods import BOX, GRID, filter_audio, measure, metadata, theta_of, varied_parameters
from benchmarks.reconstruction import relative_error
from benchmarks.loss_batch_grouped import Seeded
from experiments.loss_batch.grouped import grouped_loss_backward
from experiments.scalar_cache.reconstruct import simulate
from src.adjoint import simulate as replay
from src.optimization import criterion_for
from src.synthesis import Renderer


def run(values,renderer,criterion,method,count,group,checkpoint):
    selected=values[:,:count].detach().clone().requires_grad_()
    parameters=torch.cat((selected,values[:,count:].detach()),dim=1)
    steps=round(renderer.duration*GRID.integration_rate)
    if method=='seeded': raw=Seeded.apply(parameters,steps,count)
    elif method=='replay': raw=replay(theta_of(parameters),GRID.intervals,GRID.integration_rate,steps,'multimode',checkpoint)
    else: raw=simulate(theta_of(parameters),GRID.intervals,GRID.integration_rate,steps,'multimode',checkpoint,int(method))
    audio=filter_audio(raw,renderer)
    if group:
        loss=grouped_loss_backward(audio,criterion,group)
    else:
        loss=torch.stack([criterion(row) for row in audio]).sum();loss.backward();loss=loss.detach()
    return audio.detach(),loss,selected.grad.detach()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--durations',default='.25,4');ap.add_argument('--batches',default='1,128')
    ap.add_argument('--directions',default='2,6');ap.add_argument('--groups',default='0,16')
    ap.add_argument('--repeats',type=int,default=3);ap.add_argument('--checkpoint',type=int,default=256)
    a=ap.parse_args();torch.set_num_threads(1);device=torch.device('cuda')
    assert torch.cuda.is_available()
    tracked=[ROOT/'experiments/scalar_cache/reconstruct.py',ROOT/'experiments/scalar_cache/kc_scalar_cache.cu',Path(__file__)]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
    result={'metadata':{**metadata(),'files_sha256':hashes,'status':'running'},'checks':[],'rows':[]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def save():a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    save()
    for duration in map(float,a.durations.split(',')):
      target=Renderer(GRID,duration).synthesize(BOX.defaults()).detach().to(device)
      renderer=Renderer(GRID,duration,device=device);criterion=criterion_for(target,GRID.sample_rate)
      for batch in map(int,a.batches.split(',')):
       values=varied_parameters(batch).to(device)
       for count in map(int,a.directions.split(',')):
        ref=run(values,renderer,criterion,'replay',count,0,a.checkpoint)
        with torch.no_grad():ref_items=torch.stack([criterion(row) for row in ref[0]])
        for group in map(int,a.groups.split(',')):
         for method in ('0','1','2','3','seeded','replay'):
          try:
            out=run(values,renderer,criterion,method,count,group,a.checkpoint)
            scaled=values[:,:count]*out[2];rg=values[:,:count]*ref[2]
            with torch.no_grad():items=torch.stack([criterion(row) for row in out[0]])
            torch.testing.assert_close(out[0],ref[0],rtol=1e-6,atol=1e-10)
            torch.testing.assert_close(out[1],ref[1],rtol=5e-4,atol=1e-10)
            torch.testing.assert_close(items,ref_items,rtol=5e-4,atol=1e-10)
            torch.testing.assert_close(scaled,rg,rtol=5e-4,atol=1e-8)
            per=torch.linalg.vector_norm(scaled-rg,dim=1)/torch.linalg.vector_norm(rg,dim=1).clamp_min(1e-30)
            check=dict(batch=batch,duration=duration,directions=count,group=group,method=method,status='passed',
                waveform_relative_l2=relative_error(out[0],ref[0]),loss_relative_l2=relative_error(out[1],ref[1]),
                per_item_loss_relative_l2=relative_error(items,ref_items),
                scaled_gradient_relative_l2=relative_error(scaled,rg),max_per_item_gradient_relative_l2=per.max().item())
            result['checks'].append(check);del out,scaled,rg,per,items;save()
          except Exception as exc:
            result['checks'].append(dict(batch=batch,duration=duration,directions=count,group=group,method=method,status='failed',error=repr(exc)))
            save();print('CHECK FAILED',result['checks'][-1],flush=True)
        del ref,ref_items
        for group in map(int,a.groups.split(',')):
         for method in ('seeded','replay','0','1','2','3'):
          check=next(x for x in result['checks'] if x['batch']==batch and x['duration']==duration and x['directions']==count and x['group']==group and x['method']==method)
          if check['status']!='passed': continue
          try:
            row=dict(batch=batch,duration=duration,directions=count,group=group,method=method,checkpoint=a.checkpoint,
                     **measure(lambda:run(values,renderer,criterion,method,count,group,a.checkpoint),device,a.repeats))
            result['rows'].append(row);save();print(json.dumps(row),flush=True)
          except Exception as exc:
            result['rows'].append(dict(batch=batch,duration=duration,directions=count,group=group,method=method,error=repr(exc)));save();print('TIMING FAILED',result['rows'][-1],flush=True)
    current={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
    result['metadata']['source_changed_during_run']=(current!=hashes or metadata()['source_sha256']!=result['metadata']['source_sha256'])
    result['metadata']['status']='complete' if all(x['status']=='passed' for x in result['checks']) and not any('error' in x for x in result['rows']) and not result['metadata']['source_changed_during_run'] else 'failed'
    save()
    if result['metadata']['status']!='complete':raise RuntimeError('One or more checks failed or sources changed')

if __name__=='__main__':main()
