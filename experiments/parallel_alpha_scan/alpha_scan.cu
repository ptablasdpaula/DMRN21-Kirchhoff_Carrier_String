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
// Prefix products of the recorded-alpha conditional modal propagators. Each
// block handles one (segment, item, mode), with time within the segment
// distributed across threads. The true nonlinear derivative is applied by the
// ordinary discrete adjoint below; alpha is conditioned on ONLY to recover
// primal states, never detached in the actual gradient calculation.
__global__ void alpha_scan_group(const double* checks,const double* coeff,const double* eig,
    const double* cache,double* tape,int* status,int B,int n,int first,int steps,int chunk) {
    __shared__ double aa[256],ab[256],ba[256],bb[256];
    int j=threadIdx.x,b=blockIdx.x,segment=first+blockIdx.y,i=blockIdx.z;
    int start=segment*chunk,end=min(start+chunk,steps),length=end-start;
    const double* c=coeff+b*8;double e=eig[i];
    double A=2*c[1]+e*c[2]-e*e*c[3],P=c[4]-e*c[5],Q=e*c[7],m=c[6];
    double alpha=j<length?cache[((size_t)b*steps+start+j)*3]:0.;
    bool bad=!isfinite(alpha)||!isfinite(A)||!isfinite(P)||!isfinite(Q)||!isfinite(m);
    if(bad) {atomicOr(status+segment*B+b,4);alpha=0.;}
    aa[j]=j<length?m*(A-Q*alpha):1.;ab[j]=j<length?m*P:0.;ba[j]=j<length?1.:0.;bb[j]=j<length?0.:1.;
    __syncthreads();
    for(int stride=1;stride<blockDim.x;stride*=2){
        double x=aa[j],y=ab[j],z=ba[j],w=bb[j],px=1.,py=0.,pz=0.,pw=1.;
        if(j>=stride){px=aa[j-stride];py=ab[j-stride];pz=ba[j-stride];pw=bb[j-stride];}
        __syncthreads();
        aa[j]=x*px+y*pz;ab[j]=x*py+y*pw;ba[j]=z*px+w*pz;bb[j]=z*py+w*pw;
        __syncthreads();
    }
    const double* cp=checks+(size_t)segment*B*2*n;
    double p=cp[(b*2)*n+i],u=cp[(b*2+1)*n+i];
    double* hist=tape+((size_t)blockIdx.y*B+b)*(chunk+2)*n;
    if(j==0){hist[i]=p;hist[n+i]=u;}
    if(j<length){double v=aa[j]*u+ab[j]*p;
        if(!isfinite(v)){atomicOr(status+segment*B+b,4);v=0.;}
        hist[(size_t)(j+2)*n+i]=v;
    }
}

// Compare both reconstructed end states against forward checkpoints. This
// catches nonfinite values and excessive boundary drift but not every possible
// interior error, so VJP checks remain mandatory.
__global__ void check_scan_group(const double* checks,const double* tape,int* status,double* drift,
    int B,int n,int first,int steps,int chunk) {
    __shared__ double red[24];int i=threadIdx.x,b=blockIdx.x,local=blockIdx.y,segment=first+local;
    bool active=i<n;int start=segment*chunk,end=min(start+chunk,steps),length=end-start;
    const double* cp=checks+(size_t)(segment+1)*B*2*n;
    const double* hist=tape+((size_t)local*B+b)*(chunk+2)*n;
    double p=active?hist[(size_t)length*n+i]:0.,u=active?hist[(size_t)(length+1)*n+i]:0.;
    double rp=active?cp[(b*2)*n+i]:0.,ru=active?cp[(b*2+1)*n+i]:0.;
    double dp=p-rp,du=u-ru,z[2]={dp*dp+du*du,rp*rp+ru*ru};sums(z,red);
    if(i==0){double err=sqrt(z[0]),norm=sqrt(z[1]);double ratio=err/(norm+1e-13*sqrt(2.*n));
        drift[segment*B+b]=ratio;
        if(!isfinite(ratio))atomicOr(status+segment*B+b,4);
        else if(err>1e-13*sqrt(2.*n)+1e-8*norm)atomicOr(status+segment*B+b,8);
    }
}
__global__ void backward_tape(const double* tape,const double* coeff,const double* eig,const double* w,
    const double* gout,double* bar,double* cg,double* gw,int* status,int B,int segment,int n,int start,int end,int count,int capacity) {
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
        if(active&&(!isfinite(p)||!isfinite(u)||!isfinite(bv)||!isfinite(den)||fabs(den)<=DBL_MIN||!isfinite(bp)||!isfinite(bu)||!isfinite(ga)||!isfinite(gp)||!isfinite(gq)||!isfinite(gm)))atomicOr(status+segment*B+b,4);
    }
    if(start==0){bp+=wt*gout[b*count];bu+=wt*gout[b*count+1];gwl+=gout[b*count]*(active?hist[i]:0.)+gout[b*count+1]*(active?hist[n+i]:0.);}
    if(active&&(!isfinite(bp)||!isfinite(bu)||!isfinite(ga)||!isfinite(gp)||!isfinite(gq)||!isfinite(gm)))atomicOr(status+segment*B+b,4);
    if(active){bar[(b*2)*n+i]=bp;bar[(b*2+1)*n+i]=bu;gw[b*n+i]+=gwl;
      cg[b*(3*n+1)+i]+=ga;cg[b*(3*n+1)+n+i]+=gp;cg[b*(3*n+1)+2*n+i]+=gq;}
    if(i==0)cg[b*(3*n+1)+3*n]+=gm;
}
std::vector<torch::Tensor> backward(torch::Tensor c,torch::Tensor e,torch::Tensor w,torch::Tensor gout,
    torch::Tensor checks,torch::Tensor cache,int64_t steps,int64_t chunk,int64_t group) {
    TORCH_CHECK(checks.is_cuda() && checks.scalar_type()==torch::kFloat64 && checks.is_contiguous() && checks.dim()==4 && checks.size(2)==2 && checks.size(3)>=3 && checks.size(3)<=256 && steps>=0 && steps<=INT_MAX-2 && chunk>0 && chunk<=256 && group>0 && group<=65535,"invalid checkpoint dimensions");
    auto B=checks.size(1),n=checks.size(3),count=steps+2,blocks=(steps+chunk-1)/chunk;
    TORCH_CHECK(checks.size(0)==blocks+1 && c.dim()==2 && c.size(0)==B && c.size(1)==8 && e.dim()==1 && e.numel()==n && w.dim()==2 && w.size(0)==B && w.size(1)==n && gout.dim()==2 && gout.size(0)==B && gout.size(1)==count,"invalid inputs");
    TORCH_CHECK(cache.dim()==3 && cache.size(0)==B && cache.size(1)==steps && cache.size(2)==3,"invalid scalar cache");
    for(const auto& t:{c,e,w,gout,cache}) TORCH_CHECK(t.device()==checks.device() && t.scalar_type()==torch::kFloat64 && t.is_contiguous(),"inputs must be contiguous float64 tensors on one device");
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
    if(steps==0) backward_tape<<<B,256,0,stream>>>(tape.data_ptr<double>(),c.data_ptr<double>(),e.data_ptr<double>(),w.data_ptr<double>(),gout.data_ptr<double>(),bar.data_ptr<double>(),grads.data_ptr<double>(),gw.data_ptr<double>(),status.data_ptr<int>(),B,0,n,0,0,count,chunk);
    for(int64_t end=blocks;end>0;) {
        int64_t first=std::max<int64_t>(0,end-group),number=end-first;
        int threads=1;while(threads<chunk)threads*=2;
        alpha_scan_group<<<dim3(B,number,n),threads,0,stream>>>(checks.data_ptr<double>(),c.data_ptr<double>(),e.data_ptr<double>(),cache.data_ptr<double>(),tape.data_ptr<double>(),status.data_ptr<int>(),B,n,first,steps,chunk);
        check_scan_group<<<dim3(B,number),256,0,stream>>>(checks.data_ptr<double>(),tape.data_ptr<double>(),status.data_ptr<int>(),drift.data_ptr<double>(),B,n,first,steps,chunk);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        for(int64_t j=end;j-->first;) {
            auto offset=(j-first)*B*(chunk+2)*n;
            backward_tape<<<B,256,0,stream>>>(tape.data_ptr<double>()+offset,c.data_ptr<double>(),e.data_ptr<double>(),w.data_ptr<double>(),gout.data_ptr<double>(),bar.data_ptr<double>(),grads.data_ptr<double>(),gw.data_ptr<double>(),status.data_ptr<int>(),B,j,n,j*chunk,std::min((j+1)*chunk,steps),count,chunk);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();end=first;
    }
    return {bar,grads,gw,status,drift};
}
// Test-only inspection: reconstruct one bounded group and expose its tape.
// Production backward uses the identical reconstruct_group kernel.
std::vector<torch::Tensor> debug_segments(torch::Tensor c,torch::Tensor e,torch::Tensor checks,
    torch::Tensor cache,int64_t steps,int64_t chunk,int64_t first,int64_t number) {
    TORCH_CHECK(checks.is_cuda() && checks.scalar_type()==torch::kFloat64 && checks.is_contiguous() && checks.dim()==4 && checks.size(2)==2 && checks.size(3)>=3 && checks.size(3)<=256 && steps>0 && steps<=INT_MAX-2 && chunk>0 && chunk<=256,"invalid debug checkpoint dimensions");
    auto B=checks.size(1),n=checks.size(3),blocks=(steps+chunk-1)/chunk;
    TORCH_CHECK(first>=0 && number>0 && number<=65535 && first+number<=blocks && checks.size(0)==blocks+1 && c.dim()==2 && c.size(0)==B && c.size(1)==8 && e.dim()==1 && e.numel()==n,"invalid debug input shape");
    TORCH_CHECK(cache.dim()==3 && cache.size(0)==B && cache.size(1)==steps && cache.size(2)==3,"invalid scalar cache");
    for(const auto& t:{c,e,cache}) TORCH_CHECK(t.device()==checks.device() && t.scalar_type()==torch::kFloat64 && t.is_contiguous(),"debug inputs must be contiguous float64 tensors on one device");
    c10::cuda::CUDAGuard guard(checks.device());
    auto tape=torch::empty({number,B,chunk+2,n},checks.options());
    auto status=torch::zeros({blocks,B},checks.options().dtype(torch::kInt32));
    auto drift=torch::zeros({blocks,B},checks.options());
    auto stream=at::cuda::getCurrentCUDAStream();
    int threads=1;while(threads<chunk)threads*=2;
    alpha_scan_group<<<dim3(B,number,n),threads,0,stream>>>(checks.data_ptr<double>(),c.data_ptr<double>(),e.data_ptr<double>(),cache.data_ptr<double>(),tape.data_ptr<double>(),status.data_ptr<int>(),B,n,first,steps,chunk);
    check_scan_group<<<dim3(B,number),256,0,stream>>>(checks.data_ptr<double>(),tape.data_ptr<double>(),status.data_ptr<int>(),drift.data_ptr<double>(),B,n,first,steps,chunk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {tape,status,drift};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("backward",&backward);m.def("debug_segments",&debug_segments);}
