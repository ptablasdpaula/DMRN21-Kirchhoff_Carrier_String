"""Independent waveform/loss/VJP checks for the experimental CeL derivative."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from cels import STFTCumulativeEnergyLoss
from experiments.loss_batch.manual_ce_loss import manual_loss_backward

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path);a=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(1292);device=torch.device(a.device)
    target=torch.randn(4096,device=device,dtype=torch.float64)*.01
    criterion=STFTCumulativeEnergyLoss(sample_rate=44100).to(device).bind(target)
    tests=[]
    for mode in ('random','near','exact','tiny'):
        x=target.repeat(7,1)
        if mode=='random':x=x+torch.randn_like(x)*.001
        if mode=='near':x=x+torch.randn_like(x)*1e-9
        if mode=='tiny':x=x*1e-8
        for group in (1,2,3,7,8):
            ref=x.detach().clone().requires_grad_();candidate=x.detach().clone().requires_grad_()
            loss=torch.stack([criterion(row) for row in ref]).sum();grad,=torch.autograd.grad(loss,ref)
            result=manual_loss_backward(candidate,criterion,group)
            torch.testing.assert_close(result,loss.detach(),rtol=1e-9,atol=1e-12)
            torch.testing.assert_close(candidate.grad,grad,rtol=1e-6,atol=1e-11)
            err=(torch.linalg.vector_norm(candidate.grad-grad)/torch.linalg.vector_norm(grad).clamp_min(1e-30)).item()
            tests.append(dict(mode=mode,group=group,grad_relative_l2=err,status='passed'))
    # Unsupported per-item targets must never be silently treated as the
    # direction axis. The full experimental pipeline uses one common target.
    batched=STFTCumulativeEnergyLoss(sample_rate=44100).to(device).bind(target.repeat(7,1))
    unsupported=target.repeat(7,1).detach().clone().requires_grad_()
    try:
        manual_loss_backward(unsupported,batched,4)
    except ValueError as error:
        if 'single common bound target' not in str(error):raise
    else:
        raise AssertionError('Per-item bound targets must be rejected')
    tests.append(dict(mode='per_item_targets',status='rejected_as_unsupported'))
    out={'device':str(device),'checks':tests,'status':'passed'}
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
