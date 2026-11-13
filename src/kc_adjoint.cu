#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <climits>

// One block per independent string. Every active mode owns its state and
// adjoints; reductions communicate only the scalar rank-one coupling.
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
__global__ void forward_chunk(const double* cp,double* next,const double* coeff,const double* eig,const double* w,
                              double* out,int n,int start,int end,int count) {
    __shared__ double red[40];int i=threadIdx.x,b=blockIdx.x;bool active=i<n;
    const double* c=coeff+b*8; double e=active?eig[i]:0;
    double A=2*c[1]+e*c[2]-e*e*c[3],P=c[4]-e*c[5],Q=e*c[7],m=c[6];
    double p=active?cp[(b*2)*n+i]:0.,u=active?cp[(b*2+1)*n+i]:0.;
    double wt=active?w[b*n+i]:0.;
    if(start==0) {double z[2]={wt*p,wt*u};sums(z,red);if(i==0){out[b*count]=z[0];out[b*count+1]=z[1];}}
    for(int t=start;t<end;t++) {
        double l=A*u+P*p,q=Q*u;double z[5]={q*p,q*q,q*l,wt*l,wt*q};sums(z,red);
        double alpha=(z[0]+m*z[2])/(1+m*z[1]);
        p=u;u=m*(l-q*alpha);
        if(i==0) out[b*count+t+2]=m*(z[3]-z[4]*alpha);
    }
    if(active){next[(b*2)*n+i]=p;next[(b*2+1)*n+i]=u;}
}
__global__ void backward_chunk(const double* cp,const double* coeff,const double* eig,const double* w,
                                const double* gout,double* tape,double* bar,double* cg,double* gw,
                                int n,int start,int end,int count,int cap) {
    __shared__ double red[40]; int i=threadIdx.x,bid=blockIdx.x;bool active=i<n;
    const double* c=coeff+bid*8;double e=active?eig[i]:0.;
    double A=2*c[1]+e*c[2]-e*e*c[3],P=c[4]-e*c[5],Q=e*c[7],m=c[6],wt=active?w[bid*n+i]:0.;
    double gwl=0.;  // d loss / d w_i from the taped states
    double* hist=tape+(size_t)bid*(cap+2)*n;
    double p=active?cp[(bid*2)*n+i]:0.,u=active?cp[(bid*2+1)*n+i]:0.;
    if(active){hist[i]=p;hist[n+i]=u;}
    for(int j=0;j<end-start;j++) {
        double l=A*u+P*p,q=Q*u;double z[3]={q*p,q*q,q*l};sums(z,red);
        double alpha=(z[0]+m*z[2])/(1+m*z[1]);p=u;u=m*(l-q*alpha);
        if(active)hist[(size_t)(j+2)*n+i]=u;
    }
    double bp=active?bar[(bid*2)*n+i]:0.,bu=active?bar[(bid*2+1)*n+i]:0.;
    double ga=0,gp=0,gq=0,gm=0;
    for(int t=end-1;t>=start;t--) {
        int j=t-start;p=active?hist[(size_t)j*n+i]:0.;u=active?hist[(size_t)(j+1)*n+i]:0.;
        double l=A*u+P*p,q=Q*u;double bv=bu+wt*gout[bid*count+t+2];
        gwl+=gout[bid*count+t+2]*(active?hist[(size_t)(j+2)*n+i]:0.);
        double z[5]={q*p,q*q,q*l,bv*q,bv*l};sums(z,red);
        double den=1+m*z[1],alpha=(z[0]+m*z[2])/den;
        double balpha=-m*z[3],ba=balpha/den,bc=ba*m,bb=-balpha*alpha*m/den;
        // Sum local mass derivative as a scalar, accumulated redundantly per thread.
        gm+=z[4]-alpha*z[3]+ba*z[2]-balpha*alpha*z[1]/den;
        double bl=m*bv+bc*q;
        double bq=-m*alpha*bv+ba*p+2*bb*q+bc*l;
        ga+=bl*u;gp+=bl*p;gq+=bq*u;
        double oldbp=ba*q+bl*P,oldbu=bp+bl*A+bq*Q;
        bp=oldbp;bu=oldbu;
    }
    if(start==0){bp+=wt*gout[bid*count];bu+=wt*gout[bid*count+1];gwl+=gout[bid*count]*(active?hist[i]:0.)+gout[bid*count+1]*(active?hist[n+i]:0.);}
    if(active){bar[(bid*2)*n+i]=bp;bar[(bid*2+1)*n+i]=bu;gw[bid*n+i]+=gwl;
      cg[bid*(3*n+1)+i]+=ga;cg[bid*(3*n+1)+n+i]+=gp;cg[bid*(3*n+1)+2*n+i]+=gq;}
    if(i==0) cg[bid*(3*n+1)+3*n]+=gm;
}
std::vector<torch::Tensor> forward(torch::Tensor s,torch::Tensor c,torch::Tensor e,torch::Tensor w,int64_t steps,int64_t chunk) {
    TORCH_CHECK(s.is_cuda() && s.scalar_type()==torch::kFloat64 && s.is_contiguous(),"expected contiguous CUDA float64");
    TORCH_CHECK(s.dim()==3&&s.size(1)==2&&s.size(2)<=256&&s.size(2)>=3&&steps>=0&&steps<=INT_MAX-2&&chunk>0&&chunk<=INT_MAX-2,"invalid adjoint dimensions");
    TORCH_CHECK(c.dim()==2 && c.size(0)==s.size(0) && c.size(1)==8 && e.dim()==1 && e.numel()==s.size(2) && w.dim()==2 && w.size(0)==s.size(0) && w.size(1)==s.size(2),"invalid coefficient or pickup shape");
    for(const auto& t:{c,e,w}) TORCH_CHECK(t.device()==s.device() && t.scalar_type()==torch::kFloat64 && t.is_contiguous(),"all inputs must be contiguous float64 tensors on one device");
    c10::cuda::CUDAGuard guard(s.device());
    auto B=s.size(0),n=s.size(2),count=steps+2,blocks=(steps+chunk-1)/chunk;
    auto audio=torch::empty({B,count},s.options());auto checks=torch::empty({blocks+1,B,2,n},s.options());checks[0].copy_(s);
    auto stream=at::cuda::getCurrentCUDAStream();
    if(steps==0) {forward_chunk<<<B,256,0,stream>>>(checks[0].data_ptr<double>(),checks[0].data_ptr<double>(),c.data_ptr<double>(),e.data_ptr<double>(),w.data_ptr<double>(),audio.data_ptr<double>(),n,0,0,count);}
    for(int64_t j=0;j<blocks;j++) forward_chunk<<<B,256,0,stream>>>(checks[j].data_ptr<double>(),checks[j+1].data_ptr<double>(),c.data_ptr<double>(),e.data_ptr<double>(),w.data_ptr<double>(),audio.data_ptr<double>(),n,j*chunk,std::min((j+1)*chunk,steps),count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {audio,checks};
}
std::vector<torch::Tensor> backward(torch::Tensor c,torch::Tensor e,torch::Tensor w,torch::Tensor gout,torch::Tensor checks,int64_t steps,int64_t chunk) {
    TORCH_CHECK(checks.is_cuda() && checks.scalar_type()==torch::kFloat64 && checks.is_contiguous() && checks.dim()==4 && checks.size(2)==2 && checks.size(3)>=3 && checks.size(3)<=256 && steps>=0 && steps<=INT_MAX-2 && chunk>0 && chunk<=INT_MAX-2,"invalid checkpoint dimensions");
    auto B=checks.size(1),n=checks.size(3),count=steps+2,blocks=(steps+chunk-1)/chunk;
    TORCH_CHECK(checks.size(0)==blocks+1 && c.dim()==2 && c.size(0)==B && c.size(1)==8 && e.dim()==1 && e.numel()==n && w.dim()==2 && w.size(0)==B && w.size(1)==n && gout.dim()==2 && gout.size(0)==B && gout.size(1)==count,"invalid adjoint input shape");
    for(const auto& t:{c,e,w,gout}) TORCH_CHECK(t.device()==checks.device() && t.scalar_type()==torch::kFloat64 && t.is_contiguous(),"all inputs must be contiguous float64 tensors on one device");
    c10::cuda::CUDAGuard guard(checks.device());
    auto tape=torch::empty({B,chunk+2,n},checks.options());auto bar=torch::zeros({B,2,n},checks.options());
    // Fourth array slot stores a scalar mass gradient per item.
    auto grads=torch::zeros({B,3*n+1},checks.options());
    auto gw=torch::zeros({B,n},checks.options());
    auto stream=at::cuda::getCurrentCUDAStream();
    if(steps==0) backward_chunk<<<B,256,0,stream>>>(checks[0].data_ptr<double>(),c.data_ptr<double>(),e.data_ptr<double>(),w.data_ptr<double>(),gout.data_ptr<double>(),tape.data_ptr<double>(),bar.data_ptr<double>(),grads.data_ptr<double>(),gw.data_ptr<double>(),n,0,0,count,chunk);
    for(int64_t j=blocks;j-->0;) backward_chunk<<<B,256,0,stream>>>(checks[j].data_ptr<double>(),c.data_ptr<double>(),e.data_ptr<double>(),w.data_ptr<double>(),gout.data_ptr<double>(),tape.data_ptr<double>(),bar.data_ptr<double>(),grads.data_ptr<double>(),gw.data_ptr<double>(),n,j*chunk,std::min((j+1)*chunk,steps),count,chunk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {bar,grads,gw};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("forward",&forward);m.def("backward",&backward);}
