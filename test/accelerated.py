"""CPU/CUDA waveform, sensitivity, device and batch parity."""
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.kc_string_forward import ForwardString,PARAMETER_COUNT
from src.accelerated import evaluate
from src.parameters import ParameterBox,MaterialParameterBox
from src.grid import GridConfig
from src.synthesis import Renderer
from src.optimization import fit,criterion_for

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');parser.add_argument('--output',type=Path)
    args=parser.parse_args();torch.set_num_threads(1);device=args.device
    def digest():
        h=hashlib.sha256()
        for f in sorted((Path(__file__).resolve().parents[1]/'src').glob('*')):
            if f.suffix in ('.py','.cpp','.cu','.h'):h.update(f.name.encode()+f.read_bytes())
        return h.hexdigest()
    source_hash=digest()
    reports=[]
    for N in (4,32,128,176,514):
        # Both stencils are exercised on one grid: the direct grid stencil (spatial) must
        # match the eigenvalue table (modal) and the native kernel for each order.
        for excitation,order in [(e,o) for e in ('first_mode','multimode','raised_cosine') for o in ((4,2) if N==32 else (4,))]:
            # Off-centre pluck and pickup: all ten tangent columns are compared across backends.
            theta=torch.tensor([1.02,.98,1.01,1.03,1.01,.9,1.1,.95,1.02,.97],dtype=torch.float64)
            # Use a sufficiently small timestep, including for the large-grid test.
            rate=44100*max(1,(N/128)**2)
            ref=ForwardString(N,rate,excitation=excitation,order=order)
            audio,jac=ref.output_and_tangents(theta,17/rate)
            for backend in ('spatial','modal'):
                model=ForwardString(N,rate,excitation=excitation,backend=backend,chunk_steps=7,order=order)
                x=theta.to(device).detach().clone().requires_grad_()
                a,j=model.output_and_tangents(x,17/rate)
                torch.testing.assert_close(a.cpu(),audio,rtol=1e-8,atol=1e-12)
                torch.testing.assert_close(j.cpu(),jac,rtol=1e-6,atol=1e-10)
                torch.testing.assert_close(model(x,17/rate).detach(),a,rtol=1e-8,atol=1e-12)
                probe=torch.linspace(-1,1,a.numel(),device=device,dtype=torch.float64)
                loss=(model(x,17/rate)*probe).sum();loss.backward()
                torch.testing.assert_close(x.grad.cpu(),probe.cpu()@jac,rtol=1e-6,atol=1e-9)
                with torch.no_grad():torch.testing.assert_close(model(x,17/rate),a,rtol=1e-8,atol=1e-12)
                full=ForwardString(N,rate,excitation=excitation,backend=backend,chunk_steps=0,order=order)
                torch.testing.assert_close(full(x,17/rate),a,rtol=0,atol=0)
                for stride in (2,7,1000):
                    sampled=ForwardString(N,rate,stride,excitation,backend,7,order=order)
                    sa,sj=sampled.output_and_tangents(x,17/rate)
                    torch.testing.assert_close(sa,a[::stride],rtol=0,atol=0)
                    torch.testing.assert_close(sj,j[::stride],rtol=0,atol=0)
                assert model(x,0).numel()==2
                batch=torch.stack([x.detach(),x.detach()*1.001])
                ba,bj=evaluate(batch,N,rate,17,1,excitation,backend,True,7,order=order)
                for i in range(2):
                    ia,ij=model.output_and_tangents(batch[i],17/rate)
                    torch.testing.assert_close(ba[i],ia,rtol=1e-12,atol=1e-14)
                    # Vectorised startup reductions can differ in sub-ulp near-zero tangents.
                    torch.testing.assert_close(bj[i],ij,rtol=1e-12,atol=1e-14)
            print('PASS',N,excitation,'order',order,flush=True)
    # Finite differences along a direction moving all ten multipliers from an
    # off-centre pluck and pickup (the pluck only moves a raised cosine).
    for backend in ('spatial','modal'):
        for excitation in ('multimode','raised_cosine'):
            model=ForwardString(128,44100,excitation=excitation,backend=backend)
            x=torch.tensor([1.]*8+[1.02,.97],device=device,dtype=torch.float64,requires_grad=True)
            direction=x.new_tensor([.3,-.4,.2,-.5,.1,.2,.15,-.2,.25,-.3])
            def objective(z):return (model(z,.02)/.002).square().mean()
            gradient=torch.autograd.grad(objective(x),x)[0]
            assert bool(gradient[9]!=0) and bool(gradient[8]!=0)==(excitation=='raised_cosine')
            epsilon=1e-4
            fd=(objective(x.detach()+epsilon*direction)-objective(x.detach()-epsilon*direction))/(2*epsilon)
            torch.testing.assert_close(gradient@direction,fd,rtol=5e-4,atol=1e-7)
            g=torch.autograd.grad(objective(x),x,create_graph=True)[0]
            try: torch.autograd.grad(g.sum(),x)
            except RuntimeError: pass
            else: raise AssertionError('Higher derivatives must be rejected')
    # The whole state (including modes suppressed by midpoint pickup) must match.
    from src.kc_string_forward import initialize, native_block, pickup_weights, second_derivative
    for amplitude in (.0001,.002,.004):
        from src import kc_string as ref
        x=torch.ones(PARAMETER_COUNT,dtype=torch.float64);x[5]=amplitude/ref.amplitude;x[6:8]=0
        state,coeff,loss=initialize(x,128,44100,'multimode')
        final,_,_=native_block(state,coeff,pickup_weights(x,128,True),loss,1000)
        for backend in ('spatial','modal'):
            _,_,actual=evaluate(x.to(device),128,44100,1000,1,'multimode',backend,True,333,return_states=True)
            torch.testing.assert_close(actual.cpu(),final,rtol=1e-6,atol=1e-10)
            def energy(s):
                # The scheme's own symmetric operator (default order) defines its conserved energy.
                p,u=s[:,:,0];h=ref.length/128;k=1/44100
                inverse_h2=torch.tensor([1/h**2],dtype=torch.float64)
                dp=second_derivative(p[:,None],inverse_h2)[:,0];du=second_derivative(u[:,None],inverse_h2)[:,0]
                return (.5*ref.linear_mass_density*h*((u-p)/k).square().sum()
                    -.5*ref.tension*h*torch.dot(u,dp)+.5*ref.young_modulus*ref.moment_of_inertia*h*torch.dot(du,dp)
                    +ref.young_modulus*ref.area*h*h/(8*ref.length)*torch.dot(u,dp).square())
            drift=abs((energy(actual.cpu())/energy(state)-1).item())
            assert drift<1e-10,(backend,amplitude,drift)
    # High nonlinearity, full duration and differentiable resampling in both coordinate systems.
    for box in (ParameterBox(),MaterialParameterBox()):
        grid=GridConfig(box);p=box.defaults();p[2 if box.coordinate_system=='perceptual' else 5]*=1.1
        reference=Renderer(grid,duration=4.)
        x=p.clone().requires_grad_();ra=reference(x);rl=criterion_for(ra.detach()*.99,grid.sample_rate)(ra);rl.backward();rg=x.grad
        for backend in ('spatial','modal'):
            renderer=Renderer(grid,duration=4.,device=device,backend=backend)
            x=p.to(device).detach().clone().requires_grad_();a=renderer(x);loss=criterion_for(ra.detach().to(device)*.99,grid.sample_rate)(a);loss.backward()
            error=(a.cpu()-ra.detach()).norm()/ra.detach().norm()
            torch.testing.assert_close(a.cpu(),ra.detach(),rtol=1e-6,atol=1e-10)
            torch.testing.assert_close(x.grad.cpu(),rg,rtol=5e-4,atol=1e-8)
            reports.append(dict(coordinates=box.coordinate_system,backend=backend,waveform_relative_l2=error.item(),loss=loss.item()))
            # Check frozen coordinates and both optimizers through the full loss.
            short=Renderer(grid,duration=.03,device=device,backend=backend)
            target=short.synthesize(p)
            initial=box.perturb(p)
            for name in ('adam','lbfgs'):
                mask=[True]+[False]*(len(box.names)-1)
                result=fit(short,target,initial,mask=mask,optimizer_name=name,max_steps=2)
                torch.testing.assert_close(result['best_physical'][1:].cpu(),initial[1:],rtol=0,atol=0)
                assert result['best_loss']<=result['history'][0]['loss']
            print('PASS full duration',box.coordinate_system,backend,reports[-1],flush=True)
    # Global-workspace fallback (does not fit in shared memory), odd chunk parity.
    if device.startswith('cuda'):
        x=torch.ones(PARAMETER_COUNT,dtype=torch.float64,device=device)
        for backend in ('spatial','modal'):
            big=ForwardString(2048,44100*256,backend=backend,chunk_steps=3)
            a,j=big.output_and_tangents(x,5/(44100*256))
            ref=ForwardString(2048,44100*256)
            ra,rj=ref.output_and_tangents(x.cpu(),5/(44100*256))
            torch.testing.assert_close(a.cpu(),ra,rtol=1e-8,atol=1e-12)
            torch.testing.assert_close(j.cpu(),rj,rtol=1e-6,atol=1e-10)
        # First-use cache creation on a different stream must be ordered explicitly.
        from src.accelerated import basis
        creator,consumer=torch.cuda.Stream(),torch.cuda.Stream()
        with torch.cuda.stream(creator):
            torch.cuda._sleep(10_000_000)
            basis(42,str(x.device),True)
        consumer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(consumer):
            cached=ForwardString(42,44100,backend='modal')(x,.001)
        torch.cuda.synchronize()
        expected=ForwardString(42,44100)(x.cpu(),.001)
        torch.testing.assert_close(cached.cpu(),expected,rtol=1e-8,atol=1e-12)
        model=ForwardString(backend='spatial');streams=[torch.cuda.Stream(),torch.cuda.Stream()];outputs=[]
        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):outputs.append(model(x,.001))
        torch.cuda.synchronize();torch.testing.assert_close(*outputs,rtol=0,atol=0)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(dict(device=device,torch=torch.__version__,source_sha256=source_hash,source_changed_during_run=digest()!=source_hash,checks=reports),indent=2))
    print('ALL PASSED',device,flush=True)
if __name__=='__main__':main()
