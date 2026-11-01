"""Compare selected f0/B gradients with the full nonlinear alpha-scan adjoint."""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from benchmarks.exact_methods import BOX,GRID,coordinates_and_seeds,filter_audio,measure,metadata,varied_parameters
from benchmarks.alpha_scan_compare import render
from benchmarks.reconstruction import relative_error
from experiments.loss_batch.grouped import grouped_loss_backward
from src.accelerated import evaluate
from src.optimization import criterion_for
from src.synthesis import Renderer

class SeededTwo(torch.autograd.Function):
    @staticmethod
    def forward(ctx,values,steps):
        theta,seeds=coordinates_and_seeds(values)
        audio,jac=evaluate(theta,GRID.intervals,GRID.integration_rate,steps,1,'multimode','modal',True,seeds=seeds[:,:,:2])
        ctx.save_for_backward(jac);ctx.count=values.shape[1];return audio
    @staticmethod
    def backward(ctx,grad):
        jac,=ctx.saved_tensors;selected=torch.einsum('bt,btd->bd',grad.contiguous(),jac)
        out=grad.new_zeros((grad.shape[0],ctx.count));out[:,:2]=selected;return out,None

def run(values,model,criterion,method,k,group,loss_group):
    p=values.detach().clone().requires_grad_()
    if method=='seeded':raw=SeededTwo.apply(p,round(model.duration*GRID.integration_rate));audio=filter_audio(raw,model)
    else:
        masked=torch.cat((p[:,:2],p[:,2:].detach()),dim=1)
        audio=render(masked,model,method,k,group)
    loss=grouped_loss_backward(audio,criterion,loss_group)
    return audio.detach(),loss,p.grad.detach()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--validation',type=Path,required=True);p.add_argument('--durations',default='.25,4');p.add_argument('--batches',default='1,128');p.add_argument('--checkpoint',type=int,default=256);p.add_argument('--group',type=int,default=8);p.add_argument('--loss-group',type=int,default=16);p.add_argument('--repeats',type=int,default=3);p.add_argument('--max-estimated-seconds',type=float,default=90);a=p.parse_args()
    assert torch.cuda.is_available();torch.set_num_threads(1);device=torch.device('cuda')
    def implhash():return hashlib.sha256((ROOT/'experiments/parallel_alpha_scan/alpha_scan.cu').read_bytes()+(ROOT/'experiments/parallel_alpha_scan/reconstruct.py').read_bytes()).hexdigest()
    def dependency_hash():
        h=hashlib.sha256()
        for f in (Path(__file__),ROOT/'benchmarks/alpha_scan_compare.py',ROOT/'experiments/loss_batch/grouped.py'):
            h.update(f.read_bytes())
        return h.hexdigest()
    impl=implhash();test_hash=hashlib.sha256((ROOT/'test/alpha_scan_reconstruction.py').read_bytes()).hexdigest();deps=dependency_hash();evidence=json.loads(a.validation.read_text());meta=metadata()
    if evidence.get('status')!='passed' or evidence.get('source_changed') or evidence.get('implementation_sha256')!=impl or evidence.get('source_sha256')!=meta['source_sha256'] or evidence.get('test_sha256')!=test_hash:raise RuntimeError('Stale or failed scan validation')
    result={'metadata':{**meta,'validation':str(a.validation),'implementation_sha256':impl,'dependencies_sha256':deps,'directions':['f0','B'],'status':'running'},'checks':[],'rows':[]};a.output.parent.mkdir(parents=True,exist_ok=True)
    def save():a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    def emit(row):result['rows'].append(row);save();print(json.dumps(row),flush=True)
    save();previous={}
    for duration in map(float,a.durations.split(',')):
      target=Renderer(GRID,duration).synthesize(BOX.defaults()).detach().to(device);model=Renderer(GRID,duration,device=device);criterion=criterion_for(target,GRID.sample_rate)
      for batch in map(int,a.batches.split(',')):
        if batch in previous:
            predicted=previous[batch][0]*duration/previous[batch][1]
            if predicted>a.max_estimated_seconds:
                emit(dict(method='scan',batch=batch,duration=duration,status='skipped',reason='predicted runtime exceeds bound',predicted_seconds=predicted));continue
        values=varied_parameters(batch).to(device);reference=run(values,model,criterion,'replay',a.checkpoint,a.group,a.loss_group)
        for method in ('seeded','cached','scan'):
            candidate=run(values,model,criterion,method,a.checkpoint,a.group,a.loss_group)
            err=dict(audio_relative_l2=relative_error(candidate[0],reference[0]),loss_relative_l2=relative_error(candidate[1],reference[1]),scaled_gradient_relative_l2=relative_error(values[:,:2]*candidate[2][:,:2],values[:,:2]*reference[2][:,:2]))
            torch.testing.assert_close(candidate[0],reference[0],rtol=1e-6,atol=1e-10);torch.testing.assert_close(candidate[1],reference[1],rtol=5e-4,atol=1e-10);torch.testing.assert_close(values*candidate[2],values*reference[2],rtol=5e-4,atol=1e-8)
            assert not bool(torch.count_nonzero(candidate[2][:,2:])), 'Unselected gradients must be zero'
            result['checks'].append(dict(method=method,batch=batch,duration=duration,status='passed',**err));del candidate
        del reference;save()
        for method in ('seeded','replay','cached','scan'):
            row=dict(method=method,batch=batch,duration=duration,checkpoint=a.checkpoint,group=a.group,loss_group=a.loss_group,status='passed',**measure(lambda:run(values,model,criterion,method,a.checkpoint,a.group,a.loss_group),device,a.repeats))
            if method=='scan':previous[batch]=(row['median_seconds'],duration)
            emit(row)
    result['metadata']['source_changed_during_run']=metadata()['source_sha256']!=meta['source_sha256'] or implhash()!=impl or dependency_hash()!=deps
    result['metadata']['status']='complete';save()
    if result['metadata']['source_changed_during_run']:raise RuntimeError('Source changed during benchmark')
if __name__=='__main__':main()
