"""GPU comparisons of scalar-cache strategies against the replay adjoint."""
import json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from benchmarks.exact_methods import varied_parameters,theta_of
from benchmarks.reconstruction import relative_error
from experiments.scalar_cache.reconstruct import simulate
from src.adjoint import simulate as replay

torch.set_num_threads(1)
assert torch.cuda.is_available()
checks=[]
for intervals,steps,batch,checkpoint in [(4,0,1,7),(4,1,1,7),(32,73,3,16),(176,1025,8,256),(176,529200,1,256),(176,529200,128,256)]:
    values=varied_parameters(batch).cuda(); theta=theta_of(values).detach()
    # The perceptual box keeps the pluck and pickup centred; move both so the
    # position gradients (pickup through the per-mode weight gradient) are live.
    theta[:,8:]=theta[:,8:]*theta.new_tensor([.96,1.05]);theta.requires_grad_()
    signal=torch.linspace(-1,1,steps+2,dtype=torch.float64,device=theta.device).reshape(1,-1)
    reference=replay(theta,intervals,132300,steps,'multimode',checkpoint)
    rg,=torch.autograd.grad((reference*signal).sum(),theta)
    for k in (0,1,2,3):
        out=simulate(theta,intervals,132300,steps,'multimode',checkpoint,k)
        grad,=torch.autograd.grad((out*signal).sum(),theta)
        torch.testing.assert_close(out,reference,rtol=1e-8,atol=1e-11)
        torch.testing.assert_close(grad,rg,rtol=5e-4,atol=1e-8)
        row=dict(intervals=intervals,steps=steps,batch=batch,checkpoint=checkpoint,cache_count=k,
                 waveform_relative_l2=relative_error(out,reference),gradient_relative_l2=relative_error(grad,rg),status='passed')
        checks.append(row);print(json.dumps(row),flush=True)
print('ALL SCALAR CACHE SMOKE CHECKS PASSED',flush=True)
