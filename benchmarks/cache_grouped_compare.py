"""Paired exact cached-adjoint / grouped-loss experiments, all coordinates."""
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from benchmarks.cache_reconstruction_compare import render
from benchmarks.exact_methods import BOX, GRID, measure, metadata, varied_parameters
from benchmarks.reconstruction import relative_error
from experiments.loss_batch.grouped import grouped_loss_backward
from src.optimization import criterion_for
from src.synthesis import Renderer

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--batch',type=int,default=128);p.add_argument('--duration',type=float,default=4);p.add_argument('--group',type=int,default=16);p.add_argument('--repeats',type=int,default=3);a=p.parse_args()
    torch.set_num_threads(1);device=torch.device('cuda')
    target=Renderer(GRID,a.duration).synthesize(BOX.defaults()).detach().to(device)
    model=Renderer(GRID,a.duration,device=device);criterion=criterion_for(target,GRID.sample_rate)
    values=varied_parameters(a.batch).to(device)
    def calc(method, grouped):
        x=values.detach().clone().requires_grad_();audio=render(x,model,method,256)
        if grouped: loss=grouped_loss_backward(audio,criterion,a.group)
        else:
            loss=torch.stack([criterion(row) for row in audio]).sum();loss.backward();loss=loss.detach()
        return audio.detach(),loss,x.grad.detach()
    reference=calc('replay',False)
    checks=[]
    for method,grouped in [('seeded',True),('cached',False),('cached',True)]:
        candidate=calc(method,grouped)
        err={'audio_relative_l2':relative_error(candidate[0],reference[0]),'loss_relative_l2':relative_error(candidate[1],reference[1]),'scaled_gradient_relative_l2':relative_error(values*candidate[2],values*reference[2])}
        torch.testing.assert_close(candidate[0],reference[0],rtol=1e-6,atol=1e-10)
        torch.testing.assert_close(candidate[1],reference[1],rtol=5e-4,atol=1e-10)
        torch.testing.assert_close(values*candidate[2],values*reference[2],rtol=5e-4,atol=1e-8)
        checks.append(dict(method=method,grouped=grouped,status='passed',**err));del candidate
    del reference
    result={'metadata':{**metadata(),'status':'running'},'checks':checks,'rows':[]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def save():a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    save()
    for method,grouped in [('seeded',False),('seeded',True),('replay',False),('reconstruction',False),('cached',False),('cached',True)]:
        row=dict(method=method,grouped=grouped,group=a.group if grouped else 1,batch=a.batch,duration=a.duration,**measure(lambda:calc(method,grouped),device,a.repeats))
        result['rows'].append(row);save();print(json.dumps(row),flush=True)
    result['metadata']['status']='complete';save()
if __name__=='__main__':main()
