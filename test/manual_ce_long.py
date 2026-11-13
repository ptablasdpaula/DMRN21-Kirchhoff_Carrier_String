"""Compare full-duration waveform VJPs for the experimental CeL backward."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from cels import STFTCumulativeEnergyLoss
from experiments.loss_batch.manual_ce_loss import manual_loss_backward

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(34);dev=torch.device(a.device);n=4*44100
    t=torch.arange(n,dtype=torch.float64,device=dev)/44100
    target=.01*torch.exp(-1.7*t)*(torch.sin(2*torch.pi*220*t)+.35*torch.sin(2*torch.pi*448*t))
    criterion=STFTCumulativeEnergyLoss(sample_rate=44100).to(dev).bind(target)
    checks=[]
    for batch,mode in [(1,'exact'),(1,'near'),(8,'amplitude')]:
        x=target.repeat(batch,1)
        if mode=='near':x=x+torch.randn_like(x)*1e-10
        if mode=='amplitude':x=x*torch.linspace(.2,1.8,batch,dtype=x.dtype,device=dev)[:,None]+1e-6*torch.randn_like(x)
        for group in (1,4,8):
            ref=x.detach().clone().requires_grad_();candidate=x.detach().clone().requires_grad_()
            loss=torch.stack([criterion(row) for row in ref]).sum();grad,=torch.autograd.grad(loss,ref)
            got=manual_loss_backward(candidate,criterion,group)
            torch.testing.assert_close(got,loss.detach(),rtol=1e-9,atol=1e-12)
            torch.testing.assert_close(candidate.grad,grad,rtol=1e-6,atol=1e-11)
            error=(torch.linalg.vector_norm(candidate.grad-grad)/torch.linalg.vector_norm(grad).clamp_min(1e-30)).item()
            checks.append(dict(batch=batch,mode=mode,group=group,gradient_relative_l2=error,status='passed'))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(status='passed',device=str(dev),checks=checks),indent=2)+'\n')
    print(json.dumps(checks),flush=True)
if __name__=='__main__':main()
