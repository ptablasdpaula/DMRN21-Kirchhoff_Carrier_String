"""Benchmark analytical default CeL waveform VJP vs grouped autograd."""
import argparse, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from benchmarks.cache_reconstruction_compare import render
from benchmarks.exact_methods import BOX, GRID, filter_audio, measure, metadata, varied_parameters
from benchmarks.loss_batch_grouped import Seeded
from benchmarks.reconstruction import relative_error
from experiments.loss_batch.grouped import grouped_loss_backward
from experiments.loss_batch.manual_ce_loss import manual_loss_backward
from src.optimization import criterion_for
from src.synthesis import Renderer

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--batch',type=int,default=128);p.add_argument('--duration',type=float,default=4);p.add_argument('--groups',default='4,8,16');p.add_argument('--repeats',type=int,default=3);a=p.parse_args()
    torch.set_num_threads(1);device=torch.device('cuda')
    target=Renderer(GRID,a.duration).synthesize(BOX.defaults()).detach().to(device)
    model=Renderer(GRID,a.duration,device=device);criterion=criterion_for(target,GRID.sample_rate)
    values=varied_parameters(a.batch).to(device)
    def calc(method,count,variant,group):
        x=values.detach().clone().requires_grad_()
        if method=='seeded':audio=filter_audio(Seeded.apply(x,round(a.duration*GRID.integration_rate),count),model)
        else:audio=render(x,model,method,256)
        loss=(manual_loss_backward if variant=='manual' else grouped_loss_backward)(audio,criterion,group)
        return audio.detach(),loss,x.grad.detach()
    tests=[]
    for method,count in [('seeded',2),('seeded',6),('cached',6)]:
        ref=calc(method,count,'autograd',8)
        for group in map(int,a.groups.split(',')):
            candidate=calc(method,count,'manual',group)
            torch.testing.assert_close(candidate[0],ref[0],rtol=1e-6,atol=1e-10)
            torch.testing.assert_close(candidate[1],ref[1],rtol=1e-9,atol=1e-12)
            torch.testing.assert_close(values*candidate[2],values*ref[2],rtol=1e-6,atol=1e-9)
            tests.append(dict(method=method,directions=count,group=group,status='passed',loss_relative=relative_error(candidate[1],ref[1]),gradient_relative=relative_error(values*candidate[2],values*ref[2])))
            del candidate
        del ref
    fingerprint=hashlib.sha256((ROOT/'experiments/loss_batch/manual_ce_loss.py').read_bytes()).hexdigest()
    result={'metadata':{**metadata(),'manual_sha256':fingerprint,'status':'running'},'checks':tests,'rows':[]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def save():a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    save()
    for method,count in [('seeded',2),('seeded',6),('cached',6)]:
      for variant in ('autograd','manual'):
       for group in map(int,a.groups.split(',')):
        row=dict(method=method,directions=count,variant=variant,group=group,batch=a.batch,duration=a.duration,**measure(lambda:calc(method,count,variant,group),device,a.repeats))
        result['rows'].append(row);save();print(json.dumps(row),flush=True)
    result['metadata']['status']='complete';save()
if __name__=='__main__':main()
