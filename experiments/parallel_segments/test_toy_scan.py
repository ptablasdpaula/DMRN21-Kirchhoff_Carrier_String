"""Validate segment reconstruction, affine scan, and parameter VJP vs autograd."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from experiments.parallel_segments.toy_scan import make_problem,forward,segment_scan,dense_storage_bytes

def main(device='cpu'):
    torch.set_num_threads(1)
    for n in (2,4,8,16):
        for batch in (1,3):
            for chunk in (1,4,8):
                steps=24;problem=make_problem(n,batch,steps,device=device)
                p=problem['p0'].detach().clone().requires_grad_();u=problem['u0'].detach().clone().requires_grad_()
                theta=torch.zeros(batch,8,device=device,dtype=torch.float64,requires_grad=True)
                problem['p0'],problem['u0']=p,u
                audio,checkpoints=forward(problem,theta,chunk)
                direct=torch.autograd.grad((audio*problem['upstream']).sum(),(p,u,theta))
                with torch.no_grad():
                    gi,gt,diagnostics=segment_scan(problem,theta.detach(),checkpoints.detach(),chunk,True)
                torch.testing.assert_close(gi[:,:n],direct[0],rtol=2e-9,atol=1e-12)
                torch.testing.assert_close(gi[:,n:],direct[1],rtol=2e-9,atol=1e-12)
                torch.testing.assert_close(gt,direct[2],rtol=2e-9,atol=1e-12)
                assert diagnostics['relative_boundary_error']<1e-10,diagnostics
                print('PASS',device,'n',n,'batch',batch,'chunk',chunk,flush=True)
    size=dense_storage_bytes(176,128,529200,256)
    print('Full-grid 4s B128 dense segment mappings only:',size,'bytes')
if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'cpu')
