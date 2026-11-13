"""Independent CPU autograd check for all cached-scalar reconstruction widths."""
import torch

torch.manual_seed(602)
for n in (3,7,32,175):
  for trial in range(24):
    p=torch.randn(n,dtype=torch.float64,requires_grad=True)
    u=torch.randn(n,dtype=torch.float64,requires_grad=True)
    A=(torch.randn(n,dtype=torch.float64)*.3).requires_grad_()
    P=(torch.rand(n,dtype=torch.float64)*.4+.6).requires_grad_()
    Q=(torch.randn(n,dtype=torch.float64)*.2).requires_grad_()
    m=torch.tensor(.8,dtype=torch.float64,requires_grad=True)
    l=A*u+P*p;q=Q*u;q2=q@q;cross=q@l;alpha=(q@p+m*cross)/(1+m*q2);v=m*(l-q*alpha)
    bu=torch.randn_like(u);bv=torch.randn_like(v)
    reference=torch.autograd.grad((u*bu+v*bv).sum(),(p,u,A,P,Q,m))
    with torch.no_grad():
      rp=(v/m-A*u+q*alpha)/P;rl=v/m+q*alpha
      torch.testing.assert_close(rp,p,rtol=1e-11,atol=1e-11)
      for count in (1,2,3):
        rq2=q@q if count==1 else q2
        rcross=q@rl if count<=2 else cross
        den=1+m*rq2;balpha=-m*(bv@q);ba=balpha/den;bb=-balpha*alpha*m/den;bc=ba*m
        bl=m*bv+bc*q;bq=-m*alpha*bv+ba*rp+2*bb*q+bc*rl
        gm=(bv@v)/m+ba*rcross-balpha*alpha*rq2/den
        actual=(ba*q+bl*P,bu+bl*A+bq*Q,bl*u,bl*rp,bq*u,gm)
        for observed,expected in zip(actual,reference):
          torch.testing.assert_close(observed,expected,rtol=1e-11,atol=1e-11)
print('PASS: 96 independent FP64 one-step cases, all scalar cache widths',flush=True)
