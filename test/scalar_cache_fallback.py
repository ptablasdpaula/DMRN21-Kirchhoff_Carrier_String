"""Exercise safe replay fallback and refusal of nonfinite/corrupt checkpoints."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import experiments.scalar_cache.reconstruct as scalar
from experiments.scalar_cache.fallback import simulate
from src.adjoint import simulate as replay
from src.kc_string_forward import PLUCK,PICKUP
from benchmarks.exact_methods import varied_parameters,theta_of

torch.set_num_threads(1)
theta=theta_of(varied_parameters(3).cuda()).detach()
# The perceptual box pins both positions at nominal; move them off centre so
# the fallback's pickup chain rule and the pluck tangent are both exercised.
theta[:,PLUCK]=theta.new_tensor([1.,.9,1.1]);theta[:,PICKUP]=theta.new_tensor([1.1,1.,.9])
theta.requires_grad_()
steps=2049;probe=torch.sin(torch.arange(steps+2,dtype=theta.dtype,device=theta.device)*.003)
audio=replay(theta,176,132300,steps,'multimode',256);reference,=torch.autograd.grad((audio*probe).sum(),theta)
for k in (1,2,3):
    direct=simulate(theta,176,132300,steps,'multimode',256,k,False)
    normal,=torch.autograd.grad((direct*probe).sum(),theta)
    forced=simulate(theta,176,132300,steps,'multimode',256,k,True)
    forced_grad,=torch.autograd.grad((forced*probe).sum(),theta)
    torch.testing.assert_close(forced,audio,rtol=1e-8,atol=1e-11)
    torch.testing.assert_close(forced_grad,reference,rtol=1e-8,atol=1e-10)
    torch.testing.assert_close(normal,reference,rtol=5e-4,atol=1e-8)

    original=scalar.kernel
    class CorruptCache:
        def forward(self,*args):return original().forward(*args)
        def backward(self,*args):
            args=list(args);cache=args[5].clone();cache[:,0,0]=float('nan');args[5]=cache
            return original().backward(*args)
    scalar.kernel=lambda:CorruptCache()
    try:
        out=simulate(theta,176,132300,steps,'multimode',256,k,False)
        grad,=torch.autograd.grad((out*probe).sum(),theta)
        torch.testing.assert_close(grad,reference,rtol=1e-8,atol=1e-10)
    finally: scalar.kernel=original
    print('PASS cache',k,'normal, forced and flagged fallback',flush=True)

# Finite checkpoint corruption must be rejected instead of silently trusting it.
original=scalar.kernel
class CorruptCheckpoint:
    def forward(self,*args):
        audio,checks,cache=original().forward(*args)
        checks=checks.clone();checks[-1,0,0,0]+=1.
        return audio,checks,cache
    def backward(self,*args):return original().backward(*args)
scalar.kernel=lambda:CorruptCheckpoint()
try:
    out=simulate(theta,176,132300,steps,'multimode',256,3,True)
    try:
        torch.autograd.grad((out*probe).sum(),theta)
        raise AssertionError('corrupt checkpoint not rejected')
    except RuntimeError as exc:
        assert 'inconsistent forward checkpoints' in str(exc),str(exc)
finally: scalar.kernel=original

# Nonfinite upstream must not silently fall back and generate invalid gradients.
bad=probe.clone();bad[0]=float('nan')
try:
    out=simulate(theta,176,132300,steps,'multimode',256,3,True)
    torch.autograd.grad((out*bad).sum(),theta)
    raise AssertionError('nonfinite upstream not rejected')
except RuntimeError as exc:
    assert 'nonfinite' in str(exc).lower(),str(exc)
print('ALL SCALAR CACHE FALLBACK CHECKS PASSED',flush=True)
