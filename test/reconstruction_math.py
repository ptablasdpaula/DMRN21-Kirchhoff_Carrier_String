"""Independently check the algebra used for reverse state reconstruction."""
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
torch.manual_seed(709)

def forward(p,u,A,P,Q,m):
    q=Q*u;l=A*u+P*p
    alpha=(q@p+m*(q@l))/(1+m*(q@q))
    return m*(l-q*alpha)

def inverse(u,v,A,P,Q,m):
    q=Q*u
    x=(v/m-A*u)/P;y=q/P
    alpha=(q@x+q@v)/(1-q@y)
    return x+y*alpha

for n in (3,7,175):
    p=torch.randn(n)*.01;u=torch.randn(n)*.01
    A=1.7+.01*torch.randn(n);P=-.9+.01*torch.randn(n)
    Q=.1*torch.randn(n);m=torch.tensor(.8)
    v=forward(p,u,A,P,Q,m)
    torch.testing.assert_close(inverse(u,v,A,P,Q,m),p,rtol=1e-11,atol=1e-13)

# Verify a multi-step physical recurrence with exact checkpoint anchors.
# Damped nonlinear trajectories make long inversion ill-conditioned, so each
# segment is checked independently against its original starting state.
n=31;A=torch.linspace(1.60,1.97,n);P=-torch.ones(n)*.999
Q=-torch.linspace(.1,.7,n);m=torch.tensor(1.)
p=torch.randn(n)*.002;u=torch.randn(n)*.002
trajectory=[p,u]
for _ in range(1031):
    p,u=u,forward(p,u,A,P,Q,m);trajectory.append(u)
for stride in (1,7,32,128,256):
    for begin in range(0,1031,stride):
        end=min(begin+stride,1031)
        u=trajectory[end].clone();v=trajectory[end+1].clone()
        for t in range(end-1,begin-1,-1):
            p=inverse(u,v,A,P,Q,m);v,u=u,p
        torch.testing.assert_close(u,trajectory[begin],rtol=1e-8,atol=1e-11)
        torch.testing.assert_close(v,trajectory[begin+1],rtol=1e-8,atol=1e-11)

# At one step, reconstruction as a composite map must differentiate to the
# identity in p and cancel dependence on u and all model coefficients.
n=7
p=(.01*torch.randn(n)).requires_grad_();u=(.01*torch.randn(n)).requires_grad_()
A=(1.7+.01*torch.randn(n)).requires_grad_();P=(-.9+.01*torch.randn(n)).requires_grad_()
Q=(.1*torch.randn(n)).requires_grad_();m=torch.tensor(.8,requires_grad=True)
weights=torch.randn(n)
v=forward(p,u,A,P,Q,m)
variables=(p,u,A,P,Q,m)
grads=torch.autograd.grad(weights@inverse(u,v,A,P,Q,m),variables)
torch.testing.assert_close(grads[0],weights,rtol=1e-10,atol=1e-10)
for grad in grads[1:]:torch.testing.assert_close(grad,torch.zeros_like(grad),rtol=0,atol=1e-10)
print('PASS independent inverse algebra, checkpoint reconstruction and differential identity')
