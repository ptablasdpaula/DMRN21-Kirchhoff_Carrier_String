"""Independent one-step reverse-mode formula check (CPU, no CUDA needed)."""
import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(4)
for n in (3,7,175):
    p=torch.randn(n,requires_grad=True);u=torch.randn(n,requires_grad=True)
    A=torch.randn(n,requires_grad=True);P=torch.randn(n,requires_grad=True)
    Q=torch.randn(n,requires_grad=True);m=torch.tensor(.1,requires_grad=True)
    vbar=torch.randn(n);pbar_next=torch.randn(n)
    l=A*u+P*p;q=Q*u
    a=q@p;b=q@q;c=q@l
    alpha=(a+m*c)/(1+m*b);v=m*(l-q*alpha)
    expected=torch.autograd.grad(v@vbar+u@pbar_next,(p,u,A,P,Q,m))
    with torch.no_grad():
        den=1+m*b;balpha=-m*(vbar@q);ba=balpha/den;bc=ba*m;bb=-balpha*alpha*m/den
        bl=m*vbar+bc*q;bq=-m*alpha*vbar+ba*p+2*bb*q+bc*l
        actual=(ba*q+bl*P,pbar_next+bl*A+bq*Q,bl*u,bl*p,bq*u,
                vbar@l-alpha*(vbar@q)+ba*c-balpha*alpha*b/den)
    for lhs,rhs in zip(actual,expected):torch.testing.assert_close(lhs,rhs,rtol=1e-11,atol=1e-11)
print('Analytic reverse step matches independent PyTorch autograd.')

# Check the reverse recurrence and per-mode parameter accumulation over a
# complete trajectory, including losses at both startup samples.
n=11;steps=23
torch.manual_seed(9)
p0=torch.randn(n,requires_grad=True);u0=torch.randn(n,requires_grad=True)
A=(torch.ones(n)*1.7+.02*torch.randn(n)).requires_grad_()
P=(-torch.ones(n)*.8+.02*torch.randn(n)).requires_grad_()
Q=(.01*torch.randn(n)).requires_grad_();m=torch.tensor(.5,requires_grad=True)
w=torch.randn(n);probes=torch.randn(steps+2)
p,u=p0,u0;states=[p,u];audio=[w@p,w@u]
for t in range(steps):
    l=A*u+P*p;q=Q*u;alpha=(q@p+m*(q@l))/(1+m*(q@q))
    p,u=u,m*(l-q*alpha);states.append(u);audio.append(w@u)
expected=torch.autograd.grad(torch.stack(audio)@probes,(p0,u0,A,P,Q,m))
with torch.no_grad():
    bp=torch.zeros(n);bu=torch.zeros(n);ga=torch.zeros(n);gp=torch.zeros(n);gq=torch.zeros(n);gm=torch.zeros(())
    for t in reversed(range(steps)):
        p,u=states[t],states[t+1];l=A*u+P*p;q=Q*u;bv=bu+w*probes[t+2]
        a=q@p;b=q@q;c=q@l;den=1+m*b;alpha=(a+m*c)/den
        balpha=-m*(bv@q);ba=balpha/den;bc=ba*m;bb=-balpha*alpha*m/den
        bl=m*bv+bc*q;bq=-m*alpha*bv+ba*p+2*bb*q+bc*l
        ga+=bl*u;gp+=bl*p;gq+=bq*u;gm+=bv@l-alpha*(bv@q)+ba*c-balpha*alpha*b/den
        bp,bu=ba*q+bl*P,bp+bl*A+bq*Q
    bp+=w*probes[0];bu+=w*probes[1]
    for actual,reference in zip((bp,bu,ga,gp,gq,gm),expected):
        torch.testing.assert_close(actual,reference,rtol=1e-10,atol=1e-10)
print('Analytic trajectory VJP matches PyTorch autograd.')
