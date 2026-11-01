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
// Each block reconstructs a different segment from its saved end checkpoint.
// Segments are independent here. The adjoint pass below is still sequential.
__global__ void reconstruct_group(const double* checks,const double* c_all,const double* eig,
    double* tape,int* status,double* drift,int B,int n,int first,int steps,int chunk,int capacity) {
    __shared__ double red[24];int i=threadIdx.x,b=blockIdx.x,segment=first+blockIdx.y;
    bool active=i<n;int start=segment*chunk,end=min(start+chunk,steps),length=end-start;
    const double* cp0=checks+(size_t)segment*B*2*n;
    const double* cp1=cp0+(size_t)B*2*n;
    const double* c=c_all+b*8;double e=active?eig[i]:0.;
    double A=2*c[1]+e*c[2]-e*e*c[3],P=c[4]-e*c[5],Q=e*c[7],m=c[6];
    double u=active?cp1[(b*2)*n+i]:0.,v=active?cp1[(b*2+1)*n+i]:0.;
    double* hist=tape+((size_t)blockIdx.y*B+b)*(capacity+2)*n;
    if(active){hist[(size_t)length*n+i]=u;hist[(size_t)(length+1)*n+i]=v;}
    bool badp=active&&(!isfinite(P)||fabs(P)<=fmax(DBL_MIN,1e-12*(fabs(c[4])+fabs(e*c[5]))));
    bool badm=!isfinite(m)||fabs(m)<=DBL_MIN;
    if(badp||badm) atomicOr(status+segment*B+b,1);
    double ip=active&&!badp?1./P:0.,im=!badm?1./m:0.;
    for(int t=end-1;t>=start;t--) {
        double q=Q*u,x=(v*im-A*u)*ip,y=q*ip;
        if(active&&(!isfinite(u)||!isfinite(v)||!isfinite(q)||!isfinite(x)||!isfinite(y))) {
            atomicOr(status+segment*B+b,4);q=x=y=0.;
        }
        double z[3]={q*x,q*y,q*v};sums(z,red);
        double den=1.-z[1];
        if(!isfinite(den)||fabs(den)<=1e-10*(1.+fabs(z[1]))) {if(i==0) atomicOr(status+segment*B+b,2);den=1.;}
        double alpha=(z[0]+z[2])/den,p=x+y*alpha;
        if(active&&(!isfinite(alpha)||!isfinite(p))){atomicOr(status+segment*B+b,4);p=0.;}
        if(active) hist[(size_t)(t-start)*n+i]=p;
        v=u;u=p;
    }
    double ru=active?cp0[(b*2)*n+i]:0.,rv=active?cp0[(b*2+1)*n+i]:0.;
    double du=u-ru,dv=v-rv;double err[2]={du*du+dv*dv,ru*ru+rv*rv};sums(err,red);
    if(i==0){double de=sqrt(err[0]),norm=sqrt(err[1]);
        double ratio=de/(norm+1e-13*sqrt(2.*n));drift[segment*B+b]=ratio;
        if(!isfinite(ratio)) atomicOr(status+segment*B+b,4);
        else if(de>1e-13*sqrt(2.*n)+1e-8*norm) atomicOr(status+segment*B+b,8);
    }
}
__global__ void backward_tape(const double* tape,const double* coeff,const double* eig,const double* w,
    const double* gout,double* bar,double* cg,double* gw,int n,int start,int end,int count,int capacity) {
    __shared__ double red[40];int i=threadIdx.x,b=blockIdx.x;bool active=i<n;
    const double* c=coeff+b*8;double e=active?eig[i]:0.;
    double A=2*c[1]+e*c[2]-e*e*c[3],P=c[4]-e*c[5],Q=e*c[7],m=c[6],wt=active?w[b*n+i]:0.;
    double gwl=0.;  // d loss / d w_i from the taped states
    const double* hist=tape+(size_t)b*(capacity+2)*n;
    double bp=active?bar[(b*2)*n+i]:0.,bu=active?bar[(b*2+1)*n+i]:0.;
    double ga=0,gp=0,gq=0,gm=0;
    for(int t=end-1;t>=start;t--) {
        int j=t-start;double p=active?hist[(size_t)j*n+i]:0.,u=active?hist[(size_t)(j+1)*n+i]:0.;
        double l=A*u+P*p,q=Q*u,bv=bu+wt*gout[b*count+t+2];
        gwl+=gout[b*count+t+2]*(active?hist[(size_t)(j+2)*n+i]:0.);
        double z[5]={q*p,q*q,q*l,bv*q,bv*l};sums(z,red);
        double den=1+m*z[1],alpha=(z[0]+m*z[2])/den;
        double balpha=-m*z[3],ba=balpha/den,bc=ba*m,bb=-balpha*alpha*m/den;
        gm+=z[4]-alpha*z[3]+ba*z[2]-balpha*alpha*z[1]/den;
        double bl=m*bv+bc*q,bq=-m*alpha*bv+ba*p+2*bb*q+bc*l;
        ga+=bl*u;gp+=bl*p;gq+=bq*u;
        double oldbp=ba*q+bl*P,oldbu=bp+bl*A+bq*Q;bp=oldbp;bu=oldbu;
    }
    if(start==0){bp+=wt*gout[b*count];bu+=wt*gout[b*count+1];gwl+=gout[b*count]*(active?hist[i]:0.)+gout[b*count+1]*(active?hist[n+i]:0.);}
    if(active){bar[(b*2)*n+i]=bp;bar[(b*2+1)*n+i]=bu;gw[b*n+i]+=gwl;
      cg[b*(3*n+1)+i]+=ga;cg[b*(3*n+1)+n+i]+=gp;cg[b*(3*n+1)+2*n+i]+=gq;}
    if(i==0)cg[b*(3*n+1)+3*n]+=gm;
}
std::vector<torch::Tensor> backward(torch::Tensor c,torch::Tensor e,torch::Tensor w,torch::Tensor gout,
    torch::Tensor checks,int64_t steps,int64_t chunk,int64_t group) {
    TORCH_CHECK(checks.is_cuda() && checks.scalar_type()==torch::kFloat64 && checks.is_contiguous() && checks.dim()==4 && checks.size(2)==2 && checks.size(3)>=3 && checks.size(3)<=256 && steps>=0 && steps<=INT_MAX-2 && chunk>0 && chunk<=INT_MAX-2 && group>0 && group<=65535,"invalid checkpoint dimensions");
    auto B=checks.size(1),n=checks.size(3),count=steps+2,blocks=(steps+chunk-1)/chunk;
    TORCH_CHECK(checks.size(0)==blocks+1 && c.dim()==2 && c.size(0)==B && c.size(1)==8 && e.dim()==1 && e.numel()==n && w.dim()==2 && w.size(0)==B && w.size(1)==n && gout.dim()==2 && gout.size(0)==B && gout.size(1)==count,"invalid inputs");
    for(const auto& t:{c,e,w,gout}) TORCH_CHECK(t.device()==checks.device() && t.scalar_type()==torch::kFloat64 && t.is_contiguous(),"inputs must be contiguous float64 tensors on one device");
    c10::cuda::CUDAGuard guard(checks.device());
    auto bar=torch::zeros({B,2,n},checks.options());auto grads=torch::zeros({B,3*n+1},checks.options());
    auto gw=torch::zeros({B,n},checks.options());
    auto status=torch::zeros({std::max<int64_t>(blocks,1),B},checks.options().dtype(torch::kInt32));
    auto drift=torch::zeros({std::max<int64_t>(blocks,1),B},checks.options());
    auto tape=torch::empty({std::min(group,std::max<int64_t>(blocks,1)),B,chunk+2,n},checks.options());
    auto stream=at::cuda::getCurrentCUDAStream();
    // Without steps nothing is reconstructed, but the startup samples still read
    // the initial state from the tape for d loss / d w.
    if(steps==0) tape[0].narrow(1,0,2).copy_(checks[0]);
    if(steps==0) backward_tape<<<B,256,0,stream>>>(tape.data_ptr<double>(),c.data_ptr<double>(),e.data_ptr<double>(),w.data_ptr<double>(),gout.data_ptr<double>(),bar.data_ptr<double>(),grads.data_ptr<double>(),gw.data_ptr<double>(),n,0,0,count,chunk);
    for(int64_t end=blocks;end>0;) {
        int64_t first=std::max<int64_t>(0,end-group),number=end-first;
        reconstruct_group<<<dim3(B,number),256,0,stream>>>(checks.data_ptr<double>(),c.data_ptr<double>(),e.data_ptr<double>(),tape.data_ptr<double>(),status.data_ptr<int>(),drift.data_ptr<double>(),B,n,first,steps,chunk,chunk);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        for(int64_t j=end;j-->first;) {
            auto offset=(j-first)*B*(chunk+2)*n;
            backward_tape<<<B,256,0,stream>>>(tape.data_ptr<double>()+offset,c.data_ptr<double>(),e.data_ptr<double>(),w.data_ptr<double>(),gout.data_ptr<double>(),bar.data_ptr<double>(),grads.data_ptr<double>(),gw.data_ptr<double>(),n,j*chunk,std::min((j+1)*chunk,steps),count,chunk);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();end=first;
    }
    return {bar,grads,gw,status,drift};
}
// Test-only inspection: reconstruct one bounded group and expose its tape.
// Production backward uses the identical reconstruct_group kernel.
std::vector<torch::Tensor> debug_segments(torch::Tensor c,torch::Tensor e,torch::Tensor checks,
    int64_t steps,int64_t chunk,int64_t first,int64_t number) {
    TORCH_CHECK(checks.is_cuda() && checks.scalar_type()==torch::kFloat64 && checks.is_contiguous() && checks.dim()==4 && checks.size(2)==2 && checks.size(3)>=3 && checks.size(3)<=256 && steps>0 && steps<=INT_MAX-2 && chunk>0 && chunk<=INT_MAX-2,"invalid debug checkpoint dimensions");
    auto B=checks.size(1),n=checks.size(3),blocks=(steps+chunk-1)/chunk;
    TORCH_CHECK(first>=0 && number>0 && number<=65535 && first+number<=blocks && checks.size(0)==blocks+1 && c.dim()==2 && c.size(0)==B && c.size(1)==8 && e.dim()==1 && e.numel()==n,"invalid debug input shape");
    for(const auto& t:{c,e}) TORCH_CHECK(t.device()==checks.device() && t.scalar_type()==torch::kFloat64 && t.is_contiguous(),"debug inputs must be contiguous float64 tensors on one device");
    c10::cuda::CUDAGuard guard(checks.device());
    auto tape=torch::empty({number,B,chunk+2,n},checks.options());
    auto status=torch::zeros({blocks,B},checks.options().dtype(torch::kInt32));
    auto drift=torch::zeros({blocks,B},checks.options());
    auto stream=at::cuda::getCurrentCUDAStream();
    reconstruct_group<<<dim3(B,number),256,0,stream>>>(checks.data_ptr<double>(),c.data_ptr<double>(),e.data_ptr<double>(),tape.data_ptr<double>(),status.data_ptr<int>(),drift.data_ptr<double>(),B,n,first,steps,chunk,chunk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {tape,status,drift};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("backward",&backward);m.def("debug_segments",&debug_segments);}
