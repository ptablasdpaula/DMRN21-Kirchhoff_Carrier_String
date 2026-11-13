"""Bounded-window exact time-segment adjoint via structured affine scans.

PyTorch research prototype, not a production kernel.  A state vector is
represented as [mode, (previous,current)]. Each step's state adjoint is a
2-by-2 block diagonal matrix plus one rank-one update. Products are represented
without any rank truncation and are composed with an inclusive suffix scan.
Only a finite window of segments is held at once; windows are processed in
reverse order. Parameter VJPs inside each window then run concurrently over its
segments using their *correct* suffix-computed terminal adjoints.
"""
import torch
from experiments.parallel_segments.toy_scan import coefficients


def checked_relative_per_item(actual,wanted,name,rtol=1e-8,atol=1e-13):
    """Check every [segment,item] separately, without loud items masking quiet ones."""
    if actual.shape!=wanted.shape or actual.ndim<3:
        raise ValueError('Expected matching [segment,item,...] arrays')
    axes=tuple(range(2,actual.ndim));width=actual[0,0].numel()
    error=(actual-wanted).square().sum(dim=axes).sqrt()
    norm=wanted.square().sum(dim=axes).sqrt()
    bound=atol*(width**.5)+rtol*norm
    ratio=error/(norm+atol*(width**.5))
    bad=(~torch.isfinite(error))|(~torch.isfinite(norm))|(~torch.isfinite(ratio))|(error>bound)
    if bool(bad.any()):
        indices=torch.nonzero(bad,as_tuple=False).cpu().tolist()
        raise RuntimeError(f'{name} failed for [segment,item] indices {indices[:8]} (max relative {ratio.max().item()})')
    return float(ratio.max()) if ratio.numel() else 0.


def matvec(d,u,v,x):
    out=torch.einsum('...nij,...nj->...ni',d,x)
    if u.shape[-1]:
        weights=torch.einsum('...nir,...ni->...r',v,x)
        out=out+torch.einsum('...nir,...r->...ni',u,weights)
    return out


def compose(left,right):
    """Return left(right(x)) without diagonal inverses or rank truncation."""
    dl,ul,vl,bl=left;dr,ur,vr,br=right
    diag=torch.einsum('...nij,...njk->...nik',dl,dr)
    du=torch.einsum('...nij,...njr->...nir',dl,ur)
    dtv=torch.einsum('...nji,...njr->...nir',dr,vl)
    if ur.shape[-1] and vl.shape[-1]:
        cross=torch.einsum('...nir,...nis->...rs',ur,vl)
        dtv=dtv+torch.einsum('...nir,...rs->...nis',vr,cross)
    us=torch.cat((du,ul),dim=-1)
    vs=torch.cat((vr,dtv),dim=-1)
    bias=bl+matvec(dl,ul,vl,br)
    return diag,us,vs,bias


def suffix_scan(d,u,v,b):
    """Inclusive ordered suffix scan over segment axis, padding exact zeros."""
    n=d.shape[0];offset=1
    while offset<n:
        left=(d[:-offset],u[:-offset],v[:-offset],b[:-offset])
        right=(d[offset:],u[offset:],v[offset:],b[offset:])
        nd,nu,nv,nb=compose(left,right)
        # Earlier rank becomes 2*r; tails retain their original factors plus
        # zero columns. Padding changes neither map nor its derivative.
        z=torch.zeros_like(u[-offset:])
        d=torch.cat((nd,d[-offset:]),dim=0)
        u=torch.cat((nu,torch.cat((u[-offset:],z),dim=-1)),dim=0)
        v=torch.cat((nv,torch.cat((v[-offset:],z),dim=-1)),dim=0)
        b=torch.cat((nb,b[-offset:]),dim=0)
        offset*=2
    return d,u,v,b


def recover_window(problem,theta,checkpoints,first,last,chunk):
    """Independently recover the per-segment forward trajectory (end anchored)."""
    A,P,Q,m=coefficients(problem,theta)
    A,P,Q,m=(z.unsqueeze(0) for z in (A,P,Q,m))
    if not bool(torch.isfinite(P).all()) or bool((P.abs()<1e-12).any()):
        raise ValueError('Singular mode diagonal')
    if not bool(torch.isfinite(m).all()) or bool((m.abs()<=torch.finfo(m.dtype).tiny).any()):
        raise ValueError('Singular mass coefficient')
    u=checkpoints[first+1:last+1,:,0,:].clone()
    v=checkpoints[first+1:last+1,:,1,:].clone()
    states=[None]*chunk
    indices=torch.arange(first,last,device=theta.device)
    total_steps=problem['upstream'].shape[-1]
    for k in range(chunk-1,-1,-1):
        active=(indices*chunk+k<total_steps)[:,None,None]
        q=Q*u;x=(v/m[...,None]-A*u)/P;y=q/P
        invden=1-(q*y).sum(-1)
        if not bool(torch.isfinite(invden).all()) or bool((invden.abs()<1e-10).any()):
            raise ValueError('Singular rank-one inverse')
        alpha=((q*x).sum(-1)+(q*v).sum(-1))/invden
        p=x+y*alpha[...,None]
        states[k]=(p,u,v,alpha,active)
        v,u=torch.where(active,u,v),torch.where(active,p,u)
    actual=torch.stack((u,v),dim=-2);wanted=checkpoints[first:last]
    error=checked_relative_per_item(actual,wanted,'Checkpoint reconstruction')
    return (A,P,Q,m),states,error


def local_terms(p,u,v,alpha,A,P,Q,m):
    """Exact J^T = block diagonal + rank one for [p,u] -> [u,v]."""
    q=Q*u;l=v/m[...,None]+q*alpha[...,None]
    q2=(q*q).sum(-1);cross=(q*l).sum(-1);den=1+m*q2
    alphap=(1+m[...,None]*P)*q/den[...,None]
    alphau=(Q*p+m[...,None]*(Q*l+A*q-2*alpha[...,None]*Q*q))/den[...,None]
    dp=(m[...,None]*P).expand_as(u);du=m[...,None]*(A-alpha[...,None]*Q)
    zeros=torch.zeros_like(dp);ones=torch.ones_like(dp)
    D=torch.stack((torch.stack((zeros,dp),dim=-1),torch.stack((ones,du),dim=-1)),dim=-2)
    a=-m[...,None,None]*torch.stack((alphap,alphau),dim=-1)
    b=torch.stack((zeros,q),dim=-1)
    return D,a,b,q,l,q2,cross,den,alphap,alphau,dp,du


def window_adjoint(problem,theta,checkpoints,chunk,first,last,terminal):
    """Full state + parameter VJP for segments [first,last), scanned jointly."""
    (A,P,Q,m),states,boundary_error=recover_window(problem,theta,checkpoints,first,last,chunk)
    S=last-first;B,n=problem['p0'].shape;shape=(S,B,n)
    eye=torch.eye(2,dtype=A.dtype,device=A.device).expand(*shape,2,2).clone()
    factors=torch.empty(*shape,2,0,dtype=A.dtype,device=A.device)
    D,U,V,bias=eye,factors,factors,torch.zeros(*shape,2,dtype=A.dtype,device=A.device)
    full=problem['upstream'][:,first*chunk:min(last*chunk,problem['upstream'].shape[-1])]
    full=torch.nn.functional.pad(full,(0,S*chunk-full.shape[-1]))
    upstream=full.reshape(B,S,chunk).permute(1,0,2)
    for k in range(chunk-1,-1,-1):
        p,u,v,alpha,active=states[k]
        diag,a,b,*_=local_terms(p,u,v,alpha,A,P,Q,m)
        diag=torch.where(active[...,None,None],diag,eye)
        a=torch.where(active[...,None],a,0.)
        # Injection of the current output y=w·v into terminal adjoint.
        inject=torch.stack((torch.zeros_like(u),upstream[:,:,k,None]*problem['w']),dim=-1)
        source=torch.einsum('...nij,...nj->...ni',diag,inject)+a*(b*inject).sum((-1,-2))[...,None,None]
        # compose one rank-one step on the left of the existing map.
        D,U,V,bias=compose((diag,a[...,None],b[...,None],source),(D,U,V,bias))
    sd,su,sv,sb=suffix_scan(D,U,V,bias)
    lastbar=terminal.unsqueeze(0).expand(S,-1,-1,-1)
    starts=matvec(sd,su,sv,lastbar)+sb
    ends=torch.cat((starts[1:],terminal.unsqueeze(0)),dim=0)
    # Each segment's exact VJP is independent once its ending adjoint is known.
    bp,bu=ends.unbind(-1)
    dA,dP,dQ=(problem[z].unsqueeze(0) for z in ('dA','dP','dQ'));dm=problem['dm'].unsqueeze(0)
    grad=torch.zeros(S,B,dm.shape[-1],device=A.device,dtype=A.dtype)
    # d loss/d w per mode: each output sample y_t = w . v_t weighted by its cotangent.
    pickup=torch.zeros(S,B,n,device=A.device,dtype=A.dtype)
    for k in range(chunk-1,-1,-1):
        p,u,v,alpha,active=states[k]
        _,_,_,q,l,q2,cross,den,alphap,alphau,dp,du=local_terms(p,u,v,alpha,A,P,Q,m)
        pickup=pickup+torch.where(active,upstream[:,:,k,None]*v,0.)
        r=bu+upstream[:,:,k,None]*problem['w'];qr=(q*r).sum(-1)
        dl=dA*u[...,None]+dP*p[...,None];dq=dQ*u[...,None]
        da=(dq*p[...,None]).sum(-2);db=2*(dq*q[...,None]).sum(-2)
        dc=(dq*l[...,None]+q[...,None]*dl).sum(-2)
        dalpha=(da+dm*cross[...,None]+m[...,None]*dc-alpha[...,None]*(dm*q2[...,None]+m[...,None]*db))/den[...,None]
        kval=dm[...,None,:]*(l-q*alpha[...,None])[...,None]+m[...,None,None]*(dl-dq*alpha[...,None,None]-q[...,None]*dalpha[...,None,:])
        grad=grad+torch.where(active,torch.einsum('...np,...n->...p',kval,r),0.)
        next_bp=dp*r-m[...,None]*alphap*qr[...,None]
        next_bu=bp+du*r-m[...,None]*alphau*qr[...,None]
        bp,bu=torch.where(active,next_bp,bp),torch.where(active,next_bu,bu)
    # Cross-check map-composed initial adjoints with the local recurrence.
    got=torch.stack((bp,bu),dim=-1)
    err=checked_relative_per_item(got,starts,'Segment boundary adjoint')
    bad=~torch.isfinite(grad).all(dim=-1)
    if bool(bad.any()):
        indices=torch.nonzero(bad,as_tuple=False).cpu().tolist()
        raise RuntimeError(f'Nonfinite parameter VJP at [segment,item] indices {indices[:8]}')
    return starts[0],grad.sum(0),pickup.sum(0),dict(boundary_error=boundary_error,adjoint_error=err,rank=su.shape[-1])


def structured_vjp(problem,theta,checkpoints,chunk,window=4):
    """Return initial-state, parameter and readout-weight VJPs; windows processed sequentially.

    The readout VJP (B, n) is d loss/d w over the stepped samples only; the
    caller chains it through the pickup weights' tangent columns and adds the
    two startup samples' contribution from the initial state itself.
    """
    S=checkpoints.shape[0]-1;B,n=problem['p0'].shape
    if chunk<1 or window<1 or problem['upstream'].shape[0]!=B or (problem['upstream'].shape[1]+chunk-1)//chunk!=S:
        raise ValueError('invalid window/checkpoint shape')
    terminal=torch.zeros(B,n,2,device=theta.device,dtype=theta.dtype)
    gradient=torch.zeros(B,problem['dm'].shape[-1],device=theta.device,dtype=theta.dtype)
    pickup=torch.zeros(B,n,device=theta.device,dtype=theta.dtype)
    details=[]
    for end in range(S,0,-window):
        first=max(0,end-window)
        terminal,g,gw,stats=window_adjoint(problem,theta,checkpoints,chunk,first,end,terminal)
        gradient=gradient+g;pickup=pickup+gw;details.append(stats)
    return terminal.transpose(-1,-2).reshape(B,2*n),gradient,pickup,details
