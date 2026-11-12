#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <climits>

// The grid counterpart of kc_adjoint.cu: the same scheme, the same checkpoints
// and the same hand-derived adjoint, stepped on the grid values instead of the
// sine modes. One block per independent string, one thread per interior grid
// point. Where the modal kernel multiplies a mode by its eigenvalue, this one
// applies the five-point stencil to its neighbours through shared memory; the
// scalar rank-one coupling is reduced exactly as in the modal kernel.
template<int R> __device__ void sums(double (&x)[R], double* part) {
    int lane=threadIdx.x%32, warp=threadIdx.x/32;
    for(int r=0;r<R;r++) {
        double v=x[r];
        for(int j=16;j;j/=2) v+=__shfl_down_sync(0xffffffff,v,j);
        if(lane==0) part[r*8+warp]=v;
    }
    __syncthreads();
    if(warp==0) for(int r=0;r<R;r++) {
        double v=lane<blockDim.x/32?part[r*8+lane]:0.;
        for(int j=16;j;j/=2) v+=__shfl_down_sync(0xffffffff,v,j);
        if(lane==0) part[r*8]=v;
    }
    __syncthreads();
    for(int r=0;r<R;r++) x[r]=part[r*8];
    __syncthreads();
}
// Second difference (dimensionless, h^2 D_xx) of the block's vector whose entry at
// this thread is v. Simply supported ends: zero end points and odd reflection about
// them, as experiments/figure_comparison/fdtd.py::_second. Consecutive calls must
// alternate between two buffers: a thread may still be reading the buffer of the
// previous call when another one writes the next.
__device__ inline double second(double* buf,const double* s,double v,int i,int n,bool active) {
    if(active) {
        buf[i+2]=v;
        if(i==0){buf[1]=0.;buf[0]=-v;}
        if(i==n-1){buf[n+2]=0.;buf[n+3]=-v;}
    }
    __syncthreads();
    return active?s[0]*buf[i]+s[1]*buf[i+1]+s[2]*buf[i+2]+s[3]*buf[i+3]+s[4]*buf[i+4]:0.;
}
// One step, with D the stencil above and coefficients c (those of kc_adjoint.cu):
//   l = 2 c1 u + c4 p - c5 D p + D (c2 u - c3 D u),  q = c7 D u,
//   alpha = (q.p + m q.l) / (1 + m q.q),  u+ = m (l - q alpha),  m = c6.
__global__ void forward_chunk(const double* cp,double* next,const double* coeff,const double* st,const double* w,
                              double* out,int n,int start,int end,int count) {
    __shared__ double red[40];__shared__ double bufa[260];__shared__ double bufb[260];
    int i=threadIdx.x,b=blockIdx.x;bool active=i<n;
    const double* c=coeff+b*8;double s[5]={st[0],st[1],st[2],st[3],st[4]};
    double c1=2*c[1],c2=c[2],c3=c[3],c4=c[4],c5=c[5],c7=c[7],m=c[6];
    double p=active?cp[(b*2)*n+i]:0.,u=active?cp[(b*2+1)*n+i]:0.;
    double wt=active?w[b*n+i]:0.;
    double d2p=second(bufb,s,p,i,n,active);
    if(start==0) {double z[2]={wt*p,wt*u};sums(z,red);if(i==0){out[b*count]=z[0];out[b*count+1]=z[1];}}
    for(int t=start;t<end;t++) {
        double d2u=second(bufa,s,u,i,n,active);
        double v=second(bufb,s,c2*u-c3*d2u,i,n,active);
        double l=c1*u+c4*p-c5*d2p+v,q=c7*d2u;double z[5]={q*p,q*q,q*l,wt*l,wt*q};sums(z,red);
        double alpha=(z[0]+m*z[2])/(1+m*z[1]);
        p=u;d2p=d2u;u=m*(l-q*alpha);
        if(i==0) out[b*count+t+2]=m*(z[3]-z[4]*alpha);
    }
    if(active){next[(b*2)*n+i]=p;next[(b*2+1)*n+i]=u;}
}
// Re-runs one checkpointed segment keeping u, D u and l of every step, then steps
// the adjoint back through it. D is symmetric, so the adjoint applies the same
// stencil to the adjoint variables. Per grid point it accumulates the six inner
// products that give the gradient with respect to c1, c2, c3, c4, c5 and c7.
__global__ void backward_chunk(const double* cp,const double* coeff,const double* st,const double* w,
                                const double* gout,double* tape,double* bar,double* cg,double* gw,
                                int n,int start,int end,int count,int cap) {
    __shared__ double red[40];__shared__ double bufa[260];__shared__ double bufb[260];
    int i=threadIdx.x,bid=blockIdx.x;bool active=i<n;
    const double* c=coeff+bid*8;double s[5]={st[0],st[1],st[2],st[3],st[4]};
    double c1=2*c[1],c2=c[2],c3=c[3],c4=c[4],c5=c[5],c7=c[7],m=c[6],wt=active?w[bid*n+i]:0.;
    size_t rows=(size_t)cap+2;
    double* hist=tape+(size_t)bid*3*rows*n;double* d2h=hist+rows*n;double* lh=d2h+rows*n;
    double p=active?cp[(bid*2)*n+i]:0.,u=active?cp[(bid*2+1)*n+i]:0.;
    double d2p=second(bufb,s,p,i,n,active);
    if(active){hist[i]=p;hist[n+i]=u;d2h[i]=d2p;}
    for(int j=0;j<end-start;j++) {
        double d2u=second(bufa,s,u,i,n,active);
        double v=second(bufb,s,c2*u-c3*d2u,i,n,active);
        double l=c1*u+c4*p-c5*d2p+v,q=c7*d2u;double z[3]={q*p,q*q,q*l};sums(z,red);
        double alpha=(z[0]+m*z[2])/(1+m*z[1]);
        if(active){d2h[(size_t)(j+1)*n+i]=d2u;lh[(size_t)j*n+i]=l;}
        p=u;d2p=d2u;u=m*(l-q*alpha);
        if(active)hist[(size_t)(j+2)*n+i]=u;
    }
    double bp=active?bar[(bid*2)*n+i]:0.,bu=active?bar[(bid*2+1)*n+i]:0.;
    double g1=0,g2=0,g3=0,g4=0,g5=0,g7=0,gm=0,gwl=0;
    for(int t=end-1;t>=start;t--) {
        size_t j=t-start;double following=0,d2u=0,l=0;p=0;u=0;d2p=0;
        if(active){p=hist[j*n+i];u=hist[(j+1)*n+i];following=hist[(j+2)*n+i];
                   d2p=d2h[j*n+i];d2u=d2h[(j+1)*n+i];l=lh[j*n+i];}
        double q=c7*d2u,go=gout[bid*count+t+2];
        double bv=bu+wt*go;gwl+=go*following;
        double z[5]={q*p,q*q,q*l,bv*q,bv*l};sums(z,red);
        double den=1+m*z[1],alpha=(z[0]+m*z[2])/den;
        double balpha=-m*z[3],ba=balpha/den,bc=ba*m,bb=-balpha*alpha*m/den;
        // Mass derivative: a scalar, accumulated redundantly per thread.
        gm+=z[4]-alpha*z[3]+ba*z[2]-balpha*alpha*z[1]/den;
        double bl=m*bv+bc*q;
        double bq=-m*alpha*bv+ba*p+2*bb*q+bc*l;
        double s1=second(bufa,s,bl,i,n,active);
        double s2=second(bufb,s,c2*bl+c7*bq-c3*s1,i,n,active);
        g1+=bl*u;g2+=bl*d2u;g3+=s1*d2u;g4+=bl*p;g5+=bl*d2p;g7+=bq*d2u;
        double oldbp=ba*q+c4*bl-c5*s1,oldbu=bp+c1*bl+s2;
        bp=oldbp;bu=oldbu;
    }
    if(start==0){bp+=wt*gout[bid*count];bu+=wt*gout[bid*count+1];gwl+=gout[bid*count]*(active?hist[i]:0.)+gout[bid*count+1]*(active?hist[n+i]:0.);}
    if(active){bar[(bid*2)*n+i]=bp;bar[(bid*2+1)*n+i]=bu;gw[bid*n+i]+=gwl;
      double* g=cg+(size_t)bid*(6*n+1);
      g[i]+=g1;g[n+i]+=g2;g[2*n+i]+=g3;g[3*n+i]+=g4;g[4*n+i]+=g5;g[5*n+i]+=g7;}
    if(i==0) cg[(size_t)bid*(6*n+1)+6*n]+=gm;
}
std::vector<torch::Tensor> forward(torch::Tensor s,torch::Tensor c,torch::Tensor st,torch::Tensor w,int64_t steps,int64_t chunk) {
    TORCH_CHECK(s.is_cuda() && s.scalar_type()==torch::kFloat64 && s.is_contiguous(),"expected contiguous CUDA float64");
    TORCH_CHECK(s.dim()==3&&s.size(1)==2&&s.size(2)<=256&&s.size(2)>=3&&steps>=0&&steps<=INT_MAX-2&&chunk>0&&chunk<=INT_MAX-2,"invalid adjoint dimensions");
    TORCH_CHECK(c.dim()==2 && c.size(0)==s.size(0) && c.size(1)==8 && st.dim()==1 && st.numel()==5 && w.dim()==2 && w.size(0)==s.size(0) && w.size(1)==s.size(2),"invalid coefficient, stencil or pickup shape");
    for(const auto& t:{c,st,w}) TORCH_CHECK(t.device()==s.device() && t.scalar_type()==torch::kFloat64 && t.is_contiguous(),"all inputs must be contiguous float64 tensors on one device");
    c10::cuda::CUDAGuard guard(s.device());
    auto B=s.size(0),n=s.size(2),count=steps+2,blocks=(steps+chunk-1)/chunk;
    auto audio=torch::empty({B,count},s.options());auto checks=torch::empty({blocks+1,B,2,n},s.options());checks[0].copy_(s);
    auto stream=at::cuda::getCurrentCUDAStream();
    if(steps==0) {forward_chunk<<<B,256,0,stream>>>(checks[0].data_ptr<double>(),checks[0].data_ptr<double>(),c.data_ptr<double>(),st.data_ptr<double>(),w.data_ptr<double>(),audio.data_ptr<double>(),n,0,0,count);}
    for(int64_t j=0;j<blocks;j++) forward_chunk<<<B,256,0,stream>>>(checks[j].data_ptr<double>(),checks[j+1].data_ptr<double>(),c.data_ptr<double>(),st.data_ptr<double>(),w.data_ptr<double>(),audio.data_ptr<double>(),n,j*chunk,std::min((j+1)*chunk,steps),count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {audio,checks};
}
std::vector<torch::Tensor> backward(torch::Tensor c,torch::Tensor st,torch::Tensor w,torch::Tensor gout,torch::Tensor checks,int64_t steps,int64_t chunk) {
    TORCH_CHECK(checks.is_cuda() && checks.scalar_type()==torch::kFloat64 && checks.is_contiguous() && checks.dim()==4 && checks.size(2)==2 && checks.size(3)>=3 && checks.size(3)<=256 && steps>=0 && steps<=INT_MAX-2 && chunk>0 && chunk<=INT_MAX-2,"invalid checkpoint dimensions");
    auto B=checks.size(1),n=checks.size(3),count=steps+2,blocks=(steps+chunk-1)/chunk;
    TORCH_CHECK(checks.size(0)==blocks+1 && c.dim()==2 && c.size(0)==B && c.size(1)==8 && st.dim()==1 && st.numel()==5 && w.dim()==2 && w.size(0)==B && w.size(1)==n && gout.dim()==2 && gout.size(0)==B && gout.size(1)==count,"invalid adjoint input shape");
    for(const auto& t:{c,st,w,gout}) TORCH_CHECK(t.device()==checks.device() && t.scalar_type()==torch::kFloat64 && t.is_contiguous(),"all inputs must be contiguous float64 tensors on one device");
    c10::cuda::CUDAGuard guard(checks.device());
    // Per item: u, D u and l of every step of one segment.
    auto tape=torch::empty({B,3*(chunk+2),n},checks.options());auto bar=torch::zeros({B,2,n},checks.options());
    // Six per-point inner products and one scalar mass gradient per item.
    auto grads=torch::zeros({B,6*n+1},checks.options());
    auto gw=torch::zeros({B,n},checks.options());
    auto stream=at::cuda::getCurrentCUDAStream();
    if(steps==0) backward_chunk<<<B,256,0,stream>>>(checks[0].data_ptr<double>(),c.data_ptr<double>(),st.data_ptr<double>(),w.data_ptr<double>(),gout.data_ptr<double>(),tape.data_ptr<double>(),bar.data_ptr<double>(),grads.data_ptr<double>(),gw.data_ptr<double>(),n,0,0,count,chunk);
    for(int64_t j=blocks;j-->0;) backward_chunk<<<B,256,0,stream>>>(checks[j].data_ptr<double>(),c.data_ptr<double>(),st.data_ptr<double>(),w.data_ptr<double>(),gout.data_ptr<double>(),tape.data_ptr<double>(),bar.data_ptr<double>(),grads.data_ptr<double>(),gw.data_ptr<double>(),n,j*chunk,std::min((j+1)*chunk,steps),count,chunk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {bar,grads,gw};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("forward",&forward);m.def("backward",&backward);}
