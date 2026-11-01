import sys, hashlib
from pathlib import Path
sys.path.insert(0,'/data/home/acw794/DMRN21-Kirchhoff_Carrier_String')
import torch
from src.adjoint import simulate as replay
from experiments.cached_reconstruction.reconstruct import simulate as cached, kernel
from src.parameters import ParameterBox
from src.grid import GridConfig
from src.kc_string_forward import nominal_parameters
from benchmarks.exact_methods import varied_parameters

torch.set_num_threads(1)
source=Path(__file__).with_name('kc_cached_reconstruct.cu')
wrapper=Path(__file__).with_name('reconstruct.py')
digest=lambda: hashlib.sha256(source.read_bytes()+wrapper.read_bytes()).hexdigest()
start_digest=digest()
print('SOURCE_SHA256',start_digest,flush=True)
box=ParameterBox();grid=GridConfig(box)
def calc(fn,theta,n,rate,steps,excitation,checkpoint):
 x=theta.detach().clone().requires_grad_()
 a=fn(x,n,rate,steps,excitation,checkpoint)
 probe=(torch.sin(torch.arange(steps+2,device='cuda')*.013)/(steps+2)).to(torch.float64)
 g,=torch.autograd.grad((a*probe).sum(),x)
 return a.detach(),g.detach()
for n,rate,steps,b,exc,k in [(4,44100,0,1,'multimode',7),(4,44100,1,1,'first_mode',1),(32,44100,17,1,'raised_cosine',7),(176,132300,2049,8,'multimode',256),(176,132300,529200,1,'multimode',256),(176,132300,529200,128,'multimode',256)]:
 vals=varied_parameters(b).cuda(); t=torch.vmap(box.to_material)(vals)/nominal_parameters('cuda'); t=t[0] if b==1 else t
 a,g=calc(cached,t,n,rate,steps,exc,k); y,h=calc(replay,t,n,rate,steps,exc,k)
 err=((g-h).norm()/h.norm().clamp_min(1e-30)).item(); wa=((a-y).norm()/y.norm().clamp_min(1e-30)).item()
 print('CHECK', n,steps,b,k,wa,err,flush=True)
 torch.testing.assert_close(a,y,rtol=1e-9,atol=1e-12); torch.testing.assert_close(g,h,rtol=5e-4,atol=1e-9)
# Deliberate corruption must not silently pass the low-level guards.
k=kernel(); dtype=torch.float64; device='cuda'; n=3; steps=7; chunk=7
e=torch.ones(n,dtype=dtype,device=device); w=torch.ones(1,n,dtype=dtype,device=device)  # per-item (B, n) pickup weights
state=torch.tensor([[[.001,.002,-.001],[.002,-.001,.003]]],dtype=dtype,device=device)
c=torch.zeros(1,8,dtype=dtype,device=device); c[0,1]=.9;c[0,4]=-1.;c[0,6]=1.
a,checkpoints,cache=k.forward(state,c,e,w,steps,chunk)
upstream=torch.ones(1,steps+2,dtype=dtype,device=device)
def flags(checks=checkpoints,values=cache,coeff=c,up=upstream):
    return int(k.backward(coeff,e,w,up,checks,values,steps,chunk)[3][0])  # (bar, grads, gw, status, drift)
assert flags()==0
corrupted=checkpoints.clone();corrupted[-1,0,0,0]+=.01
assert flags(checks=corrupted)&8, 'corrupted checkpoint went undetected'
corrupted=checkpoints.clone();corrupted[-1,0,0,0]=float('nan')
assert flags(checks=corrupted)&4, 'nonfinite checkpoint went undetected'
corrupted=cache.clone();corrupted[0,3,0]=float('nan')
assert flags(values=corrupted)&2, 'nonfinite cached alpha went undetected'
corrupted=cache.clone();corrupted[0,3,1]=-1.
assert flags(values=corrupted)&2, 'singular cached denominator went undetected'
singular=c.clone();singular[0,4]=0.
a,checkpoints_s,cache_s=k.forward(state,singular,e,w,steps,chunk)
assert flags(checks=checkpoints_s,values=cache_s,coeff=singular)&1, 'singular diagonal went undetected'
up=upstream.clone();up[0,4]=float('inf')
assert flags(up=up)&4, 'nonfinite upstream gradient went undetected'
assert digest()==start_digest, 'candidate source changed during run'
print('PASS guards and stable source',start_digest,flush=True)
