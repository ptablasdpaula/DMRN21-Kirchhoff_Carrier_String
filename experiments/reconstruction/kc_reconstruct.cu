#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <climits>
#include <cfloat>

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
// Reconstruct the previous state from the end checkpoint while applying
// the adjoint. The nonlinear rank-one inverse and reverse quantities share
// one reduction per step. Each segment is independently anchored to its end
// pair; its reconstructed start is checked against the saved forward pair.
// status bits: 1 invalid diagonal/mass, 2 invalid rank-one denominator,
// 4 nonfinite values, 8 boundary drift above tolerance.
__global__ void reconstruct_chunk(const double* cp_start,const double* cp_end,
                                const double* coeff,const double* eig,const double* w,
                                const double* gout,double* bar,double* cg,double* gw,int* status,double* drift,
                                int n,int start,int end,int count,double rtol,double atol) {
    __shared__ double red[48]; int i=threadIdx.x,bid=blockIdx.x;bool active=i<n;
    const double* c=coeff+bid*8;double e=active?eig[i]:0.;
    double A=2*c[1]+e*c[2]-e*e*c[3],P=c[4]-e*c[5],Q=e*c[7],m=c[6],wt=active?w[bid*n+i]:0.;
    double gwl=0.;  // d loss / d w_i from the reconstructed states
    double u=active?cp_end[(bid*2)*n+i]:0.,v=active?cp_end[(bid*2+1)*n+i]:0.;
    double bp=active?bar[(bid*2)*n+i]:0.,bu=active?bar[(bid*2+1)*n+i]:0.;
    double ga=0,gp=0,gq=0,gm=0;
    double pscale=fabs(c[4])+fabs(e*c[5]);
    bool needs_inverse=end>start;
    bool badp=active&&needs_inverse&&(!isfinite(P)||fabs(P)<=fmax(DBL_MIN,1e-12*pscale));
    bool badm=needs_inverse&&(!isfinite(m)||fabs(m)<=DBL_MIN);
    if(badp||badm) atomicOr(status+bid,1);
    double invp=active&&needs_inverse&&!badp?1./P:0.,invm=needs_inverse&&!badm?1./m:0.;
    for(int t=end-1;t>=start;t--) {
        double q=Q*u,bv=bu+wt*gout[bid*count+t+2];
        gwl+=gout[bid*count+t+2]*v;
        double x=(v*invm-A*u)*invp,y=q*invp;
        if(active&&(!isfinite(u)||!isfinite(v)||!isfinite(q)||!isfinite(bv)||!isfinite(x)||!isfinite(y))) {
            atomicOr(status+bid,4);q=0.;bv=0.;x=0.;y=0.;
        }
        double z[6]={q*x,q*y,q*v,q*q,bv*q,bv*v};sums(z,red);
        double invden=1.-z[1];
        if(!isfinite(invden)||fabs(invden)<=1e-10*(1.+fabs(z[1]))) {
            if(i==0) atomicOr(status+bid,2);invden=1.;
        }
        double alpha=(z[0]+z[2])/invden;
        double p=x+y*alpha,l=v*invm+q*alpha;
        double den=1.+m*z[3],cross=z[2]*invm+alpha*z[3];
        if(!isfinite(alpha)||!isfinite(den)||fabs(den)<=DBL_MIN) {
            if(i==0) atomicOr(status+bid,4);alpha=0.;den=1.;cross=0.;p=0.;l=0.;
        }
        double balpha=-m*z[4],ba=balpha/den,bc=ba*m,bb=-balpha*alpha*m/den;
        gm+=z[5]*invm+ba*cross-balpha*alpha*z[3]/den;
        double bl=m*bv+bc*q,bq=-m*alpha*bv+ba*p+2*bb*q+bc*l;
        ga+=bl*u;gp+=bl*p;gq+=bq*u;
        double oldbp=ba*q+bl*P,oldbu=bp+bl*A+bq*Q;
        if(active&&(!isfinite(p)||!isfinite(oldbp)||!isfinite(oldbu)||!isfinite(ga)||!isfinite(gp)||!isfinite(gq)||!isfinite(gm))) {
            atomicOr(status+bid,4);p=0.;oldbp=0.;oldbu=0.;ga=0.;gp=0.;gq=0.;gm=0.;
        }
        bp=oldbp;bu=oldbu;v=u;u=p;
    }
    // Sum normalized state discrepancy across BOTH levels at this boundary.
    double ru=active?cp_start[(bid*2)*n+i]:0.,rv=active?cp_start[(bid*2+1)*n+i]:0.;
    double du=u-ru,dv=v-rv;
    double check[2]={du*du+dv*dv,ru*ru+rv*rv};sums(check,red);
    double err=sqrt(check[0]),norm=sqrt(check[1]);
    if(i==0) {
        double ratio=err/(norm+atol*sqrt(2.*n));
        if(!isfinite(err)||!isfinite(norm)||!isfinite(ratio)) atomicOr(status+bid,4);
        else {
            drift[bid]=fmax(drift[bid],ratio);
            if(err>atol*sqrt(2.*n)+rtol*norm) atomicOr(status+bid,8);
        }
    }
    if(start==0){bp+=wt*gout[bid*count];bu+=wt*gout[bid*count+1];gwl+=gout[bid*count]*ru+gout[bid*count+1]*rv;}
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
    auto bar=torch::zeros({B,2,n},checks.options());
    auto grads=torch::zeros({B,3*n+1},checks.options());
    auto gw=torch::zeros({B,n},checks.options());
    auto status=torch::zeros({B},checks.options().dtype(torch::kInt32));
    auto drift=torch::zeros({B},checks.options());
    auto stream=at::cuda::getCurrentCUDAStream();
    constexpr double rtol=1e-8,atol=1e-13;
    if(steps==0) reconstruct_chunk<<<B,256,0,stream>>>(checks[0].data_ptr<double>(),checks[0].data_ptr<double>(),c.data_ptr<double>(),e.data_ptr<double>(),w.data_ptr<double>(),gout.data_ptr<double>(),bar.data_ptr<double>(),grads.data_ptr<double>(),gw.data_ptr<double>(),status.data_ptr<int>(),drift.data_ptr<double>(),n,0,0,count,rtol,atol);
    for(int64_t j=blocks;j-->0;) reconstruct_chunk<<<B,256,0,stream>>>(checks[j].data_ptr<double>(),checks[j+1].data_ptr<double>(),c.data_ptr<double>(),e.data_ptr<double>(),w.data_ptr<double>(),gout.data_ptr<double>(),bar.data_ptr<double>(),grads.data_ptr<double>(),gw.data_ptr<double>(),status.data_ptr<int>(),drift.data_ptr<double>(),n,j*chunk,std::min((j+1)*chunk,steps),count,rtol,atol);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {bar,grads,gw,status,drift};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("forward",&forward);m.def("backward",&backward);}
