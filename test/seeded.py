"""Independent material-Jacobian and coordinate VJP checks for seeded propagation."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.accelerated import evaluate
from src.kc_string_forward import ForwardString,PARAMETER_COUNT
from src.parameters import ParameterBox,MaterialParameterBox
from src.grid import GridConfig
from src.synthesis import Renderer
from src.optimization import fit,criterion_for


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');args=parser.parse_args()
    torch.set_num_threads(1);device=args.device
    # Off-centre pluck and pickup: every seed mixes all ten multiplier directions.
    theta=torch.tensor([1.02,.98,1.01,1.03,1.01,.9,1.1,.95,1.02,.97],dtype=torch.float64,device=device)
    seeds=torch.arange(PARAMETER_COUNT**2,dtype=torch.float64,device=device).reshape(PARAMETER_COUNT,PARAMETER_COUNT).sin()*.2
    for excitation in ('first_mode','multimode','raised_cosine'):
        for backend in ('modal','spatial'):
            model=ForwardString(128,44100,excitation=excitation,backend=backend,chunk_steps=7)
            base,jac=model.output_and_tangents(theta,.001)
            for size in range(1,PARAMETER_COUNT+1):
                audio,j=evaluate(theta,128,44100,44,1,excitation,backend,True,7,seeds=seeds[:,:size])
                torch.testing.assert_close(audio,base,rtol=1e-10,atol=1e-13)
                torch.testing.assert_close(j,jac@seeds[:,:size],rtol=2e-6,atol=1e-10)
            batch=torch.stack((theta,theta*1.001));seed_batch=torch.stack((seeds[:,:3],seeds[:,:3]*1.5))
            ba,bj=evaluate(batch,128,44100,44,1,excitation,backend,True,7,seeds=seed_batch)
            for i in range(2):
                ref,refj=model.output_and_tangents(batch[i],.001)
                torch.testing.assert_close(ba[i],ref,rtol=1e-10,atol=1e-13)
                torch.testing.assert_close(bj[i],refj@seed_batch[i],rtol=2e-6,atol=1e-10)
            print('PASS directional',excitation,backend,flush=True)
    for box in (ParameterBox(),MaterialParameterBox()):
        grid=GridConfig(box)
        for backend in ('modal','spatial'):
            renderer=Renderer(grid,.03,device=device,backend=backend)
            truth=box.defaults();target=renderer.synthesize(truth)
            x=box.perturb(truth).to(device).detach().requires_grad_()
            loss=criterion_for(target,grid.sample_rate)
            actual=renderer(x);ref=torch.autograd.grad(loss(actual),x)[0]
            for mask in ([True]*len(box.names),[True]+[False]*(len(box.names)-1),[i%2==0 for i in range(len(box.names))]):
                audio=renderer.forward_seeded(x,mask)
                grad=torch.autograd.grad(loss(audio),x)[0]
                expected=ref*torch.tensor(mask,device=device,dtype=ref.dtype)
                torch.testing.assert_close(audio,actual,rtol=1e-9,atol=1e-12)
                torch.testing.assert_close(grad,expected,rtol=5e-5,atol=1e-7)
            mask=[i%2==0 for i in range(len(box.names))]
            direction=x.detach()*torch.tensor([.2 if flag else 0. for flag in mask],device=device)
            energy=lambda audio:(audio/.002).square().mean()
            gradient=torch.autograd.grad(energy(renderer.forward_seeded(x,mask)),x)[0]
            epsilon=1e-4
            with torch.no_grad():
                finite=(energy(renderer(x.detach()+epsilon*direction))-
                        energy(renderer(x.detach()-epsilon*direction)))/(2*epsilon)
            torch.testing.assert_close(gradient@direction,finite,rtol=5e-4,atol=1e-7)
            # Ordinary optimizers retain frozen coordinates and identical first step.
            for opt in ('adam','lbfgs'):
                base=fit(renderer,target,x.detach(),mask,optimizer_name=opt,max_steps=1)
                seeded=fit(renderer,target,x.detach(),mask,optimizer_name=opt,max_steps=1,sensitivity='seeded')
                torch.testing.assert_close(seeded['best_physical'],base['best_physical'],rtol=1e-7,atol=1e-11)
                assert abs(seeded['best_loss']-base['best_loss'])<1e-9
            frozen=fit(renderer,target,x.detach(),[False]*len(box.names),sensitivity='seeded')
            assert frozen['stop_reason']=='No parameters selected.'
            torch.testing.assert_close(frozen['best_physical'],x.detach())
            print('PASS coordinates',box.coordinate_system,backend,flush=True)
    print('ALL SEEDED TESTS PASSED',device,flush=True)
if __name__=='__main__':main()
