"""Filtered CeL gradient validation in perceptual and direct material coordinates."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from benchmarks.exact_methods import digest as source_digest
from experiments.cached_reconstruction.reconstruct import simulate as reconstruct
from src.adjoint import simulate as replay
from src.grid import GridConfig
from src.kc_string_forward import nominal_parameters
from src.optimization import criterion_for
from src.parameters import MaterialParameterBox,ParameterBox
from src.synthesis import Renderer


def reconstruction_digest():
    folder=ROOT/'experiments/cached_reconstruction'
    return hashlib.sha256((folder/'kc_cached_reconstruct.cu').read_bytes()+(folder/'reconstruct.py').read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    assert torch.cuda.is_available(),'Run on a Slurm GPU worker'
    torch.set_num_threads(1)
    sh=source_digest();eh=reconstruction_digest();th=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rows=[];device='cuda'
    for box in (ParameterBox(),MaterialParameterBox()):
        grid=GridConfig(box);reference=box.defaults();start=box.perturb(reference)
        for duration in (.03,4.):
            cpu=Renderer(grid,duration)
            gpu=Renderer(grid,duration,device=device,backend='modal')
            target=cpu.synthesize(reference);loss_gpu=criterion_for(target.to(device),grid.sample_rate)
            def render(values, method):
                multipliers=box.to_material(values)/nominal_parameters(values.device)
                raw=method(multipliers,grid.intervals,grid.integration_rate,round(duration*grid.integration_rate),'multimode',256)
                return gpu._filter_audio(raw)
            def run(method):
                values=start.to(device).detach().clone().requires_grad_()
                audio=render(values,method);loss=loss_gpu(audio)
                (gradient,)=torch.autograd.grad(loss,values)
                return audio.detach(),loss.detach(),gradient.detach()
            a,l,g=run(reconstruct);b,m,h=run(replay)
            torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-12)
            torch.testing.assert_close(l,m,rtol=1e-6,atol=1e-12)
            scale=start.to(device)
            torch.testing.assert_close(g*scale,h*scale,rtol=5e-4,atol=1e-8)
            # A second independent native-forward Jacobian checks both physical
            # reparameterization and filtering, rather than replay alone.
            q=start.clone().requires_grad_();native=cpu(q);loss_cpu=criterion_for(target,grid.sample_rate)(native)
            (native_g,)=torch.autograd.grad(loss_cpu,q)
            torch.testing.assert_close(a.cpu(),native.detach(),rtol=1e-6,atol=1e-10)
            torch.testing.assert_close(l.cpu(),loss_cpu.detach(),rtol=5e-4,atol=1e-10)
            torch.testing.assert_close((g*scale).cpu(),native_g*start,rtol=5e-4,atol=1e-8)
            # Finite differences on a relative coordinate direction, including
            # f0 and inharmonicity for perceptual coordinates.
            if duration==.03:
                q=start.to(device).detach().clone().requires_grad_()
                mask=torch.tensor([i<2 or i%2==0 for i in range(q.numel())],device=device)
                vec=torch.linspace(.1,.25,q.numel(),dtype=q.dtype,device=device)*q.detach()*mask
                def objective(p):return loss_gpu(render(torch.where(mask,p,q.detach()),reconstruct))
                derivative=torch.autograd.grad(objective(q),q)[0]
                assert torch.count_nonzero(derivative[~mask])==0
                eps=1e-5
                with torch.no_grad(): finite=(objective(q.detach()+eps*vec)-objective(q.detach()-eps*vec))/(2*eps)
                torch.testing.assert_close(derivative@vec,finite,rtol=2e-3,atol=1e-8)
            rows.append(dict(coordinates=box.coordinate_system,intervals=grid.intervals,rate=grid.integration_rate,
                             duration=duration,loss=float(l),gradient_relative_l2=float((g*scale-h*scale).norm()/(h*scale).norm().clamp_min(1e-30)),status='passed'))
            print(json.dumps(rows[-1]),flush=True)
    # Additional high-damping/amplitude, undamped and frequency/inharmonicity
    # corners with independent replay VJP and a directional finite difference.
    box=ParameterBox();grid=GridConfig(box);device='cuda';nominal=nominal_parameters(device)
    cases=[('low_amplitude',.0005,None,None,None,None),('high_amplitude',.003,None,None,None,None),
           ('undamped',None,0.,0.,None,None),('high_damping',None,2.,.001,None,None),
           ('stability_corner',.003,2.,.001,100.,.001)]
    for name,amp,sigma0,sigma1,f0,inharm in cases:
        q=box.defaults().to(device)
        for i,value in ((2,amp),(4,sigma0),(5,sigma1),(0,f0),(1,inharm)):
            if value is not None:q[i]=value
        box.validate(q);q=q.detach().requires_grad_();steps=2049
        time=torch.arange(steps+2,dtype=q.dtype,device=device)
        probe=(torch.sin(.013*time)+.27*torch.cos(.0037*time))/(steps+2)
        def objective(values, method):
            theta=box.to_material(values)/nominal
            return (method(theta,grid.intervals,grid.integration_rate,steps,'multimode',256)*probe).sum()
        loss=objective(q,reconstruct);grad,=torch.autograd.grad(loss,q)
        ref_loss=objective(q,replay);ref_grad,=torch.autograd.grad(ref_loss,q)
        torch.testing.assert_close(loss,ref_loss,rtol=1e-9,atol=1e-12)
        scale=q.detach().clone().abs().clamp_min(1e-4)
        torch.testing.assert_close(grad*scale,ref_grad*scale,rtol=5e-4,atol=1e-8)
        # Interior direction leaves any coordinate sitting at a boundary fixed.
        lower,upper=box.bounds();lower=lower.to(device);upper=upper.to(device)
        mask=(q.detach()>lower+1e-9*(1+lower.abs()))&(q.detach()<upper-1e-9*(1+upper.abs()))
        vec=torch.linspace(.05,.2,q.numel(),dtype=q.dtype,device=device)*q.detach()*mask
        eps=1e-5
        with torch.no_grad(): finite=(objective(q.detach()+eps*vec,reconstruct)-objective(q.detach()-eps*vec,reconstruct))/(2*eps)
        torch.testing.assert_close(grad@vec,finite,rtol=2e-3,atol=1e-8)
        row=dict(name=name,gradient_relative_l2=float(((grad-ref_grad)*scale).norm()/(ref_grad*scale).norm().clamp_min(1e-30)), status='passed')
        rows.append(row);print(json.dumps(row),flush=True)
    changed=(source_digest()!=sh or reconstruction_digest()!=eh or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=th)
    if changed:raise RuntimeError('Source changed during coordinate validation')
    output=dict(status='passed',source_sha256=sh,reconstruction_sha256=eh,test_sha256=th,source_changed_during_run=changed,
                gpu=torch.cuda.get_device_name(),checks=rows)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(output,indent=2)+'\n')
    print('ALL FILTERED COORDINATE CHECKS PASSED',flush=True)
if __name__=='__main__':main()
