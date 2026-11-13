"""Check experimental CUDA paths directly against the existing modal recurrence."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from src.accelerated import basis, kernel
from src.kc_string_forward import DUAL_COLUMNS, SPATIAL_ORDER, initialize, dual_multiply, pickup_weights

@torch.no_grad()
def inputs(n, columns, batch, excitation):
    # Off-centre pluck and pickup, so the last two dual columns carry tangents.
    values=torch.tensor([1.01,.98,1.02,1.03,.99,.95,1.01,.97,1.02,.97],dtype=torch.float64,device='cuda')
    states=[];coefficients=[];pickups=[]
    for i in range(batch):
        row=values*(1+i*.0001)
        s,c,_=initialize(row,n,132300,excitation)
        h2=c[0].clone();c[2]=dual_multiply(c[2],h2)
        c[3]=dual_multiply(dual_multiply(c[3],h2),h2)
        c[5]=dual_multiply(c[5],h2);c[7]=dual_multiply(c[7],h2)
        states.append(s[:,:,:columns]);coefficients.append(c[:,:columns])
        # Per-item modal pickup weights with the same column slice as the states.
        pickups.append(pickup_weights(row,n,False)[:,:columns])
    transform,eigen,_,ready=basis(n,'cuda:0',True)
    if ready is not None:torch.cuda.current_stream().wait_event(ready)
    state=torch.stack(states);state=torch.einsum('ij,btjd->btid',transform.T,state).contiguous()
    return state,torch.stack(coefficients).contiguous(),eigen,torch.stack(pickups).contiguous()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    assert torch.cuda.is_available()
    extension=kernel(True);cases=0;worst=0.
    for n in (4,32,128,176):
        for columns in range(1,DUAL_COLUMNS+1):
            for excitation in ('first_mode','multimode','raised_cosine'):
                state,coeff,eigen,weight=inputs(n,columns,2,excitation)
                for steps,stride,chunk in ((0,1,0),(29,1,7),(31,3,0)):
                    reference=extension.simulate(state,coeff,eigen,weight,steps,stride,True,chunk,256,SPATIAL_ORDER,0)
                    variants=(1,2) if columns==1 else (1,)
                    for variant in variants:
                        actual=extension.simulate(state,coeff,eigen,weight,steps,stride,True,chunk,256,SPATIAL_ORDER,variant)
                        for got,expected in zip(actual,reference):
                            torch.testing.assert_close(got,expected,rtol=1e-8,atol=1e-12)
                            if got.numel(): worst=max(worst,(got-expected).abs().max().item())
                        cases+=1
    torch.cuda.synchronize()
    result=dict(device=torch.cuda.get_device_name(),cases=cases,max_absolute_error=worst,status='passed')
    print(json.dumps(result,indent=2),flush=True)
    if args.output:args.output.write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
