"""Toy exact discrete-adjoint segment scan for the nonlinear rank-one string.

All segments reconstruct backwards in parallel from checkpoint pairs. Each
segment produces its affine gradient map; a suffix scan connects boundaries.
This is a dense research prototype: it is NOT used by the production solver.
"""
import math
import torch


def make_problem(n=4,batch=1,steps=16,seed=41,device='cpu'):
    gen=torch.Generator(device='cpu').manual_seed(seed)
    def random(*shape): return torch.randn(*shape,generator=gen,dtype=torch.float64).to(device)
    b=batch
    result=dict(p0=.01*random(b,n),u0=.01*random(b,n),
       A0=1.60+.01*random(b,n), P0=-.70+.01*random(b,n), Q0=.01*random(b,n),
       m0=torch.full((b,),1.,device=device,dtype=torch.float64),
       dA=.005*random(b,n,8),dP=.005*random(b,n,8),dQ=.001*random(b,n,8),dm=.005*random(b,8),
       w=random(n),upstream=random(b,steps)/steps)
    return result


def coefficients(problem,theta):
    return (problem['A0']+torch.einsum('bni,bi->bn',problem['dA'],theta),
            problem['P0']+torch.einsum('bni,bi->bn',problem['dP'],theta),
            problem['Q0']+torch.einsum('bni,bi->bn',problem['dQ'],theta),
            problem['m0']+torch.einsum('bi,bi->b',problem['dm'],theta))


def advance(p,u,A,P,Q,m):
    q=Q*u;l=A*u+P*p
    alpha=((q*p).sum(-1)+m*(q*l).sum(-1))/(1+m*(q*q).sum(-1))
    return m[...,None]*(l-q*alpha[...,None])


def forward(problem,theta,chunk,requires_autograd=False):
    steps=problem['upstream'].shape[-1]
    if chunk<1 or steps%chunk:raise ValueError('steps must be divisible by chunk')
    A,P,Q,m=coefficients(problem,theta)
    p,u=problem['p0'],problem['u0'];checkpoints=[torch.stack((p,u),dim=-2)];outputs=[]
    for t in range(steps):
        v=advance(p,u,A,P,Q,m);outputs.append((v*problem['w']).sum(-1));p,u=u,v
        if (t+1)%chunk==0:checkpoints.append(torch.stack((p,u),dim=-2))
    audio=torch.stack(outputs,dim=1)
    return audio,torch.stack(checkpoints)


def affine_suffix(M,b):
    """Inclusive suffix composition of x -> M_s x + b_s over segment axis."""
    M,b=M.clone(),b.clone();n=M.shape[0];offset=1
    while offset<n:
        outM,outb=M.clone(),b.clone()
        outM[:-offset]=M[:-offset]@M[offset:]
        outb[:-offset]=b[:-offset]+(M[:-offset]@b[offset:].unsqueeze(-1)).squeeze(-1)
        M,b=outM,outb;offset*=2
    return M,b


def segment_scan(problem,theta,checkpoints,chunk,return_debug=False):
    """Reconstruct independent segments concurrently and scan exact adjoints.

    Returns gradients for initial p/u and eight coefficient directions per
    item. One dense 2N-square state transition is stored per segment.
    """
    A,P,Q,m=coefficients(problem,theta)
    S=checkpoints.shape[0]-1;B,n=problem['p0'].shape;dim=2*n
    if checkpoints.shape!=(S+1,B,2,n) or problem['upstream'].shape!=(B,S*chunk):raise ValueError('Invalid checkpoint shapes')
    A,P,Q,m=(x.unsqueeze(0) for x in (A,P,Q,m))
    if not bool(torch.isfinite(P).all()) or bool((P.abs()<1e-12).any()):raise ValueError('Singular diagonal')
    dA,dP,dQ=(problem[k].unsqueeze(0) for k in ('dA','dP','dQ'));dm=problem['dm'].unsqueeze(0)
    M=torch.eye(dim,dtype=A.dtype,device=A.device).expand(S,B,dim,dim).clone()
    b=torch.zeros(S,B,dim,dtype=A.dtype,device=A.device)
    G=torch.zeros(S,B,8,dim,dtype=A.dtype,device=A.device)
    h=torch.zeros(S,B,8,dtype=A.dtype,device=A.device)
    u=checkpoints[1:,:,0,:].clone();v=checkpoints[1:,:,1,:].clone()
    upstream=problem['upstream'].reshape(B,S,chunk).permute(1,0,2)
    for k in range(chunk-1,-1,-1):
        q=Q*u;x=(v/m[...,None]-A*u)/P;y=q/P
        invden=1-(q*y).sum(-1)
        if not bool(torch.isfinite(invden).all()) or bool((invden.abs()<1e-10).any()):raise ValueError('Singular rank-one inverse')
        alpha=((q*x).sum(-1)+(q*v).sum(-1))/invden
        p=x+y*alpha[...,None];l=v/m[...,None]+q*alpha[...,None]
        bq=(q*q).sum(-1);cross=(q*l).sum(-1);den=1+m*bq
        # Gradients of the nonlinear scalar alpha with respect to p and u.
        alphap=(1+m[...,None]*P)*q/den[...,None]
        alphau=(Q*p+m[...,None]*(Q*l+A*q-2*alpha[...,None]*Q*q))/den[...,None]
        dp=m[...,None]*P;du=m[...,None]*(A-alpha[...,None]*Q)
        # Forward parameter Jacobian of v for eight fixed directions.
        dl=dA*u[...,None]+dP*p[...,None];dq=dQ*u[...,None]
        da=(dq*p[...,None]).sum(-2);db=2*(dq*q[...,None]).sum(-2)
        dc=(dq*l[...,None]+q[...,None]*dl).sum(-2)
        dalpha=(da+dm*cross[...,None]+m[...,None]*dc-alpha[...,None]*(dm*bq[...,None]+m[...,None]*db))/den[...,None]
        K=dm[...,None,:]*(l-q*alpha[...,None])[...,None]+m[...,None,None]*(dl-dq*alpha[...,None,None]-q[...,None]*dalpha[...,None,:])
        top,bottom=M[...,:n,:],M[...,n:,:]
        btop,bbottom=b[...,:n],b[...,n:]
        r=bbottom+upstream[:,:,k,None]*problem['w']
        # Parameter gradient is affine in the segment's ending adjoint.
        G=G+torch.einsum('sbnp,sbnk->sbpk',K,bottom)
        h=h+torch.einsum('sbnp,sbn->sbp',K,r)
        qM=torch.einsum('sbn,sbnk->sbk',q,bottom);qr=(q*r).sum(-1)
        M=torch.cat((dp[...,None]*bottom-m[...,None,None]*alphap[...,None]*qM[...,None,:],
                     top+du[...,None]*bottom-m[...,None,None]*alphau[...,None]*qM[...,None,:]),dim=-2)
        b=torch.cat((dp*r-m[...,None]*alphap*qr[...,None],
                     btop+du*r-m[...,None]*alphau*qr[...,None]),dim=-1)
        v,u=u,p
    reference=checkpoints[:-1]
    reconstructed=torch.stack((u,v),dim=-2)
    boundary_error=(reconstructed-reference).norm()/reference.norm().clamp_min(1e-30)
    _,suffix_b=affine_suffix(M,b)
    end_adj=torch.cat((suffix_b[1:],torch.zeros_like(suffix_b[:1])),dim=0)
    grad_theta=(h+torch.einsum('sbpk,sbk->sbp',G,end_adj)).sum(0)
    grad_initial=suffix_b[0]
    if return_debug:
        return grad_initial,grad_theta,{'relative_boundary_error':boundary_error.item(),
          'segment_count':S,'state_dim':dim,'segment_map_bytes':M.numel()*M.element_size(),
          'parameter_map_bytes':G.numel()*G.element_size()}
    return grad_initial,grad_theta


def dense_storage_bytes(intervals,batch,steps,chunk,parameters=8):
    """Dense segment state/parameter maps only; excludes scratch and scan copies."""
    n=intervals-1;d=2*n;segments=math.ceil(steps/chunk)
    return segments*batch*8*(d*d+parameters*d+d+parameters)
