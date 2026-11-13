"""Independent linear closed form, finite-amplitude derivatives and discrete energy."""
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src import kc_string as ref
from src.kc_string_forward import (DUAL_COLUMNS,PARAMETER_COUNT,STENCILS,discrete_eigenvalues,initialize,native_block,
                                   pickup_weights,second_derivative,_native_kernel)
from src.grid import GridConfig
from src.synthesis import Renderer
from src.parameters import reference_parameters


def linear_check(order):
    N=32; k=1/44100; steps=80; sigma0=.7; sigma1=.0003
    base=torch.tensor([ref.wave_speed**2,ref.stiffness**2,ref.length,ref.amplitude],dtype=torch.float64)
    relative=torch.ones(4,dtype=torch.float64)
    # First-mode eigenvalue of the chosen stencil in the discrete sine basis.
    lam1=-float(discrete_eigenvalues(N,order)[0])
    # Independent scalar modal recurrence; the reference startup includes h.
    def closed_form(relative):
        c2,K2,L,a=base*relative; h=L/N
        q2=lam1/h**2
        w2=c2*q2+K2*q2*q2
        A=1+sigma0*k
        D=-2+w2*k*k+2*sigma1*k*q2
        C=1-sigma0*k-2*sigma1*k*q2
        root=torch.sqrt((D*D-4*A*C).to(torch.complex128))
        r1=(-D+root)/(2*A);r2=(-D-root)/(2*A)
        y1=a*(1-.5*k*k*h*w2)
        alpha=(y1-a*r2)/(r1-r2)
        t=torch.arange(steps+2,dtype=torch.float64)
        return (alpha*r1**t+(a-alpha)*r2**t).real
    # Prepare ordinary tensor coefficients, then seed their exact derivatives.
    # The native step is unchanged; q=0 isolates the linear update.
    def packed(relative):
        c2,K2,L,a=base*relative;h=L/N
        shape=torch.sin(math.pi*torch.arange(1,N,dtype=torch.float64)/N)
        q2=lam1/h**2
        states=torch.stack((a*shape,a*shape*(1-.5*k*k*h*(c2*q2+K2*q2*q2))))
        zero=c2*0
        coeff=torch.stack((1/h**2,zero+1,c2*k*k+2*sigma1*k,K2*k*k,
                           zero+sigma0*k-1,zero+2*sigma1*k,zero+1/(1+sigma0*k),zero))
        return torch.cat((states.flatten(),coeff))
    packed_values=packed(relative)
    jac=torch.func.jacfwd(packed)(relative)
    dual=torch.zeros((packed_values.numel(),DUAL_COLUMNS),dtype=torch.float64)
    dual[:,0]=packed_values;dual[:,1:5]=jac
    state=dual[:2*(N-1)].reshape(2,N-1,DUAL_COLUMNS).contiguous()
    coeff=dual[2*(N-1):].contiguous()
    # Nominal pickup: the exact one-hot midpoint node, where the first mode peaks.
    weights=pickup_weights(torch.ones(PARAMETER_COUNT,dtype=torch.float64),N,True)
    audio,tangents=_native_kernel().simulate(state,coeff,weights,steps,1,order)
    expected=closed_form(relative);expected_jac=torch.func.jacfwd(closed_form)(relative)
    torch.testing.assert_close(audio,expected,rtol=1e-9,atol=1e-12)
    torch.testing.assert_close(tangents[:,:4],expected_jac,rtol=2e-6,atol=2e-11)
    print(f'Order {order}: independent linear closed-form value and c²,K²,L,a derivatives passed.')


def energy_check(order):
    N=128;rate=44100;k=1/rate;h=ref.length/N
    inverse_h2=torch.tensor([1/h**2],dtype=torch.float64)
    # The scheme's own symmetric operator defines its conserved discrete energy.
    def dxx(values): return second_derivative(values[:,None],inverse_h2,order)[:,0]
    def energy(states):
        previous,current=states[:,:,0]
        dprev=dxx(previous)
        dcurrent=dxx(current)
        kinetic=.5*ref.linear_mass_density*h*((current-previous)/k).square().sum()
        linear=-.5*ref.tension*h*torch.dot(current,dprev)+.5*ref.young_modulus*ref.moment_of_inertia*h*torch.dot(dcurrent,dprev)
        nonlinear=ref.young_modulus*ref.area*h*h/(8*ref.length)*torch.dot(current,dprev).square()
        return kinetic+linear+nonlinear
    if order==2:
        probe=torch.randn(N-1,dtype=torch.float64)
        torch.testing.assert_close(dxx(probe),ref.second_spatial_derivative(probe,h),rtol=1e-13,atol=1e-13)
    for amplitude in (.0001,.002,.004):
        theta=torch.ones(PARAMETER_COUNT,dtype=torch.float64);theta[5]=amplitude/ref.amplitude;theta[6:8]=0
        states,coeff,loss=initialize(theta,N,rate,'multimode',order)
        weights=pickup_weights(theta,N,True)
        initial=energy(states);maximum=0.
        for _ in range(200):
            states,loss,_=native_block(states,coeff,weights,loss,5,order)
            maximum=max(maximum,abs((energy(states)/initial-1).item()))
        assert maximum<1e-10,maximum
        print(f'Order {order}: undamped discrete energy, amplitude {amplitude:g} m: maximum relative drift {maximum:.3g}')


def nonlinear_gradient_check():
    grid=GridConfig();renderer=Renderer(grid,.02,'multimode')
    base=reference_parameters();base[2]=.003
    x=torch.ones(6,dtype=torch.float64,requires_grad=True)
    def objective(x):
        audio=renderer(base*x)/.002
        return audio.square().mean()+.1*audio.mean()
    y=objective(x);g=torch.autograd.grad(y,x)[0]
    eps=1e-4
    fd=[]
    for i in range(6):
        offset=torch.zeros(6,dtype=torch.float64);offset[i]=eps
        fd.append((objective(x.detach()+offset)-objective(x.detach()-offset))/(2*eps))
    torch.testing.assert_close(g,torch.stack(fd),rtol=5e-4,atol=1e-7)
    direction=torch.tensor([.3,-.4,.2,-.5,.1,.2],dtype=torch.float64)
    errors=[]
    for step in (.004,.002,.001):
        errors.append(abs((objective(x.detach()+step*direction)-y.detach()-step*torch.dot(g,direction)).item()))
    assert errors[0]/errors[1]>3 and errors[1]/errors[2]>3,errors
    print('Finite-amplitude central differences and quadratic Taylor remainder passed:',errors)

if __name__=='__main__':
    torch.set_num_threads(1)
    for order in sorted(STENCILS):
        linear_check(order);energy_check(order)
    nonlinear_gradient_check()
