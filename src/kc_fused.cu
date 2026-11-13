#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "kc_dual.h"

// Warp reductions plus a small shared-memory array; no cross-block barriers.
template<int D,int R> __device__ void reduce(KCValue<D> (&v)[R],KCValue<D>* partial) {
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,warps=blockDim.x/32;
    for(int r=0;r<R;++r) for(int j=0;j<D;++j) {
        double x=v[r].v[j];
        for(int off=16;off;off/=2) x+=__shfl_down_sync(0xffffffff,x,off);
        if(lane==0) partial[r*warps+warp].v[j]=x;
    }
    __syncthreads();
    if(warp==0) for(int r=0;r<R;++r) for(int j=0;j<D;++j) {
        double x=lane<warps?partial[r*warps+lane].v[j]:0.;
        for(int off=16;off;off/=2) x+=__shfl_down_sync(0xffffffff,x,off);
        if(lane==0) partial[r*warps].v[j]=x;
    }
    __syncthreads();
    for(int r=0;r<R;++r) v[r]=partial[r*warps];
    // Prevent a fast warp from overwriting partials while another reads them.
    __syncthreads();
}
template<int D,bool Modal,bool Shared> __global__ void advance(
    KCValue<D>* work,const KCValue<D>* coeff,const double* eigen,const KCValue<D>* weight,
    double* audio,double* jac,int n,int64_t start,int64_t end,int64_t count,int64_t stride,int order) {
    using V=KCValue<D>;
    extern __shared__ double storage[];
    auto shared=reinterpret_cast<V*>(storage);
    V* base=work+blockIdx.x*4*n;
    V* p=Shared?shared:base;
    V* u=p+n;V* l=u+n;V* q=l+n;
    V* partial=shared+(Shared?4*n:0);
    const V* c=coeff+blockIdx.x*8;const V* w=weight+blockIdx.x*n;
    if constexpr(Shared) for(int i=threadIdx.x;i<2*n;i+=blockDim.x) p[i]=base[i];
    __syncthreads();
    auto record=[&](V value,int64_t t) {
        if(threadIdx.x==0 && t%stride==0) {
            const auto row=blockIdx.x*count+t/stride;
            audio[row]=value.v[0];
            for(int j=1;j<D;++j) jac[row*(D-1)+j-1]=value.v[j];
        }
    };
    if(start==0) {
        V initial[2]{};
        for(int i=threadIdx.x;i<n;i+=blockDim.x) {
            initial[0]=initial[0]+w[i]*p[i];initial[1]=initial[1]+w[i]*u[i];
        }
        reduce(initial,partial);record(initial[0],0);record(initial[1],1);
    }
    for(int64_t t=start;t<end;++t) {
        // The readout w·p_new = m(w·l - w·q alpha) is formed from the pre-update
        // sums in both coordinate systems; with a one-hot spatial pickup it is
        // the node's own update, term for term.
        V sums[5]{};
        for(int i=threadIdx.x;i<n;i+=blockDim.x) {
            kc_terms<D,Modal>(p,u,c,eigen,i,n,order,l[i],q[i]);
            sums[0]=sums[0]+q[i]*p[i];sums[1]=sums[1]+q[i]*q[i];sums[2]=sums[2]+q[i]*l[i];
            sums[3]=sums[3]+w[i]*l[i];sums[4]=sums[4]+w[i]*q[i];
        }
        reduce(sums,partial);
        V denom=c[6]*sums[1];denom.v[0]+=1.;
        const V alpha=(sums[0]+c[6]*sums[2])/denom;
        for(int i=threadIdx.x;i<n;i+=blockDim.x) p[i]=c[6]*(l[i]-q[i]*alpha);
        record(c[6]*(sums[3]-sums[4]*alpha),t+2);
        __syncthreads();
        V* tmp=p;p=u;u=tmp;
    }
    // Canonical order for the next chunk. Global mode needs a staging buffer
    // when the pointer parity is odd, rather than an overlapping in-place copy.
    if constexpr(Shared) {
        for(int i=threadIdx.x;i<n;i+=blockDim.x) {base[i]=p[i];base[n+i]=u[i];}
    } else if((end-start)%2) {
        for(int i=threadIdx.x;i<n;i+=blockDim.x) {V a=p[i],b=u[i];base[i]=a;base[n+i]=b;}
    }
}

// Default-grid modal path: one mode per thread, persistent states in registers.
// All modes participate; no mode truncation or continuous-frequency substitution.
template<int D> __global__ void modal_register(
    KCValue<D>* work,const KCValue<D>* coeff,const double* eigen,const KCValue<D>* weight,
    double* audio,double* jac,int n,int64_t start,int64_t end,int64_t count,int64_t stride) {
    using V=KCValue<D>;extern __shared__ double storage[];
    auto partial=reinterpret_cast<V*>(storage);
    const int i=threadIdx.x;const bool active=i<n;
    V* base=work+blockIdx.x*4*n;const V* c=coeff+blockIdx.x*8;
    V p{},u{},A{},P{},Q{},mass=c[6],w{};
    if(active) {
        p=base[i];u=base[n+i];w=weight[blockIdx.x*n+i];double ev=eigen[i];
        A=2.0*c[1]+ev*c[2]-(ev*ev)*c[3];P=c[4]-ev*c[5];Q=ev*c[7];
    }
    auto record=[&](V v,int64_t t) {
        if(i==0&&t%stride==0) {
            auto row=blockIdx.x*count+t/stride;audio[row]=v.v[0];
            for(int j=1;j<D;++j) jac[row*(D-1)+j-1]=v.v[j];
        }
    };
    if(start==0) {
        V initial[2]={w*p,w*u};reduce(initial,partial);
        record(initial[0],0);record(initial[1],1);
    }
    for(int64_t t=start;t<end;++t) {
        const V l=A*u+P*p,q=Q*u;
        V sums[5]={q*p,q*q,q*l,w*l,w*q};reduce(sums,partial);
        V denom=mass*sums[1];denom.v[0]+=1;
        const V alpha=(sums[0]+mass*sums[2])/denom;
        p=u;u=mass*(l-q*alpha);
        record(mass*(sums[3]-sums[4]*alpha),t+2);
    }
    if(active) {base[i]=p;base[n+i]=u;}
}


// One warp per dual column: spatial/modal coordinates AND sensitivities run
// concurrently. SoA shared state allows the derivative warps to read the primal.
template<int D> __global__ void modal_channels(
    KCValue<D>* work,const KCValue<D>* coeff,const double* eigen,const KCValue<D>* weight,
    double* audio,double* jac,int n,int64_t start,int64_t end,int64_t count,int64_t stride) {
    extern __shared__ double data[];
    const int lane=threadIdx.x,col=threadIdx.y;
    double* p=data;double* u=p+n*D;double* l=u+n*D;double* q=l+n*D;
    double* A=q+n*D;double* P=A+n*D;double* Q=P+n*D;double* sums=Q+n*D;
    auto base=work+blockIdx.x*4*n;auto c=coeff+blockIdx.x*8;auto W=weight+blockIdx.x*n;
    const double mass=c[6].v[0],dm=c[6].v[col];
    // Dual pickup: column 0 reads w0*x0, a tangent column reads w0*x_col + w_col*x0.
    auto pick=[&](int i,const double* x) {return col==0?W[i].v[0]*x[i]:W[i].v[0]*x[col*n+i]+W[i].v[col]*x[i];};
    for(int i=lane;i<n;i+=32) {
        const int ij=col*n+i;const double e=eigen[i];
        p[ij]=base[i].v[col];u[ij]=base[n+i].v[col];
        A[ij]=2*c[1].v[col]+e*c[2].v[col]-e*e*c[3].v[col];
        P[ij]=c[4].v[col]-e*c[5].v[col];Q[ij]=e*c[7].v[col];
    }
    __syncthreads();
    auto warp_sum=[](double x) {
        for(int offset=16;offset;offset/=2) x+=__shfl_down_sync(0xffffffff,x,offset);
        return x;
    };
    auto record=[&](double value,int64_t t) {
        if(lane==0&&t%stride==0) {
            auto row=blockIdx.x*count+t/stride;
            if(col==0) audio[row]=value;else jac[row*(D-1)+col-1]=value;
        }
    };
    if(start==0) {
        double a=0,b=0;for(int i=lane;i<n;i+=32) {a+=pick(i,p);b+=pick(i,u);}
        record(warp_sum(a),0);record(warp_sum(b),1);
    }
    for(int64_t t=start;t<end;++t) {
        for(int i=lane;i<n;i+=32) {
            int ij=col*n+i;
            if(col==0) {l[ij]=A[i]*u[i]+P[i]*p[i];q[ij]=Q[i]*u[i];}
            else {
                l[ij]=A[ij]*u[i]+A[i]*u[ij]+P[ij]*p[i]+P[i]*p[ij];
                q[ij]=Q[ij]*u[i]+Q[i]*u[ij];
            }
        }
        __syncthreads();
        double a=0,b=0,cross=0,wl=0,wq=0;
        for(int i=lane;i<n;i+=32) {
            int ij=col*n+i;
            if(col==0) {a+=q[i]*p[i];b+=q[i]*q[i];cross+=q[i]*l[i];}
            else {a+=q[ij]*p[i]+q[i]*p[ij];b+=2*q[i]*q[ij];cross+=q[ij]*l[i]+q[i]*l[ij];}
            wl+=pick(i,l);wq+=pick(i,q);
        }
        a=warp_sum(a);b=warp_sum(b);cross=warp_sum(cross);wl=warp_sum(wl);wq=warp_sum(wq);
        if(lane==0) {sums[col*5]=a;sums[col*5+1]=b;sums[col*5+2]=cross;sums[col*5+3]=wl;sums[col*5+4]=wq;}
        __syncthreads();
        double den=1+mass*sums[1],alpha=(sums[0]+mass*sums[2])/den;
        double da=(sums[col*5]+dm*sums[2]+mass*sums[col*5+2]-alpha*(dm*sums[1]+mass*sums[col*5+1]))/den;
        for(int i=lane;i<n;i+=32) {
            int ij=col*n+i;double primal=l[i]-q[i]*alpha;
            p[ij]=col==0?mass*primal:dm*primal+mass*(l[ij]-q[ij]*alpha-q[i]*da);
        }
        double primal=sums[3]-sums[4]*alpha;
        record(col==0?mass*primal:dm*primal+mass*(sums[col*5+3]-sums[col*5+4]*alpha-sums[4]*da),t+2);
        __syncthreads();double* tmp=p;p=u;u=tmp;
    }
    for(int i=lane;i<n;i+=32) {base[i].v[col]=p[col*n+i];base[n+i].v[col]=u[col*n+i];}
}


// Experimental channel layout: recompute per-mode linear terms rather than
// sharing l and q. Three state buffers allow all warps to read old p/u while
// writing next, eliminating the barrier after l/q production.
template<int D> __global__ void modal_recompute(
    KCValue<D>* work,const KCValue<D>* coeff,const double* eigen,const KCValue<D>* weight,
    double* audio,double* jac,int n,int64_t start,int64_t end,int64_t count,int64_t stride) {
    extern __shared__ double data[];
    const int lane=threadIdx.x,col=threadIdx.y;
    double* p=data;double* u=p+n*D;double* next=u+n*D;
    double* A=next+n*D;double* P=A+n*D;double* Q=P+n*D;double* sums=Q+n*D;
    auto base=work+blockIdx.x*4*n;auto c=coeff+blockIdx.x*8;auto W=weight+blockIdx.x*n;
    const double mass=c[6].v[0],dm=c[6].v[col];
    for(int i=lane;i<n;i+=32) {
        int ij=col*n+i;double e=eigen[i];
        p[ij]=base[i].v[col];u[ij]=base[n+i].v[col];
        A[ij]=2*c[1].v[col]+e*c[2].v[col]-e*e*c[3].v[col];
        P[ij]=c[4].v[col]-e*c[5].v[col];Q[ij]=e*c[7].v[col];
    }
    __syncthreads();
    auto warp_sum=[](double x) {
        for(int offset=16;offset;offset/=2) x+=__shfl_down_sync(0xffffffff,x,offset);
        return x;
    };
    auto record=[&](double value,int64_t t) {
        if(lane==0&&t%stride==0) {
            auto row=blockIdx.x*count+t/stride;
            if(col==0) audio[row]=value;else jac[row*(D-1)+col-1]=value;
        }
    };
    if(start==0) {
        double a=0,b=0;
        for(int i=lane;i<n;i+=32) {
            a+=col==0?W[i].v[0]*p[i]:W[i].v[0]*p[col*n+i]+W[i].v[col]*p[i];
            b+=col==0?W[i].v[0]*u[i]:W[i].v[0]*u[col*n+i]+W[i].v[col]*u[i];
        }
        record(warp_sum(a),0);record(warp_sum(b),1);
    }
    for(int64_t t=start;t<end;++t) {
        double a=0,b=0,cross=0,wl=0,wq=0;
        for(int i=lane;i<n;i+=32) {
            int ij=col*n+i;
            double l0=A[i]*u[i]+P[i]*p[i],q0=Q[i]*u[i];
            double l=col==0?l0:A[ij]*u[i]+A[i]*u[ij]+P[ij]*p[i]+P[i]*p[ij];
            double q=col==0?q0:Q[ij]*u[i]+Q[i]*u[ij];
            if(col==0) {a+=q0*p[i];b+=q0*q0;cross+=q0*l0;}
            else {a+=q*p[i]+q0*p[ij];b+=2*q0*q;cross+=q*l0+q0*l;}
            // Dual pickup: the tangent column also carries w_col times the primal terms.
            wl+=W[i].v[0]*l+(col==0?0.:W[i].v[col]*l0);wq+=W[i].v[0]*q+(col==0?0.:W[i].v[col]*q0);
        }
        a=warp_sum(a);b=warp_sum(b);cross=warp_sum(cross);wl=warp_sum(wl);wq=warp_sum(wq);
        if(lane==0) {sums[col*5]=a;sums[col*5+1]=b;sums[col*5+2]=cross;sums[col*5+3]=wl;sums[col*5+4]=wq;}
        __syncthreads();
        double den=1+mass*sums[1],alpha=(sums[0]+mass*sums[2])/den;
        double da=(sums[col*5]+dm*sums[2]+mass*sums[col*5+2]-alpha*(dm*sums[1]+mass*sums[col*5+1]))/den;
        for(int i=lane;i<n;i+=32) {
            int ij=col*n+i;
            double l0=A[i]*u[i]+P[i]*p[i],q0=Q[i]*u[i];
            double l=col==0?l0:A[ij]*u[i]+A[i]*u[ij]+P[ij]*p[i]+P[i]*p[ij];
            double q=col==0?q0:Q[ij]*u[i]+Q[i]*u[ij];
            double primal=l0-q0*alpha;
            next[ij]=col==0?mass*primal:dm*primal+mass*(l-q*alpha-q0*da);
        }
        double primal=sums[3]-sums[4]*alpha;
        record(col==0?mass*primal:dm*primal+mass*(sums[col*5+3]-sums[col*5+4]*alpha-sums[4]*da),t+2);
        __syncthreads();double* old=p;p=u;u=next;next=old;
    }
    for(int i=lane;i<n;i+=32) {base[i].v[col]=p[col*n+i];base[n+i].v[col]=u[col*n+i];}
}

template<int D,bool Modal> std::vector<torch::Tensor> launch(torch::Tensor s,torch::Tensor c,
    torch::Tensor e,torch::Tensor w,int64_t steps,int64_t stride,int64_t chunk,int threads,int variant,int order) {
    int B=s.size(0),n=s.size(2);int64_t count=(steps+1)/stride+1;
    auto audio=torch::empty({B,count},s.options()),jac=torch::empty({B,count,D-1},s.options());
    auto workspace=torch::empty({B,4,n,D},s.options());
    workspace.slice(1,0,2).copy_(s);
    auto stream=at::cuda::getCurrentCUDAStream();
    int device=s.get_device(),limit=0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&limit,cudaDevAttrMaxSharedMemoryPerBlockOptin,device));
    size_t partial=5*(threads/32)*sizeof(KCValue<D>);
    size_t shared=4*n*sizeof(KCValue<D>)+partial;
    bool fits=shared<=static_cast<size_t>(limit);
    size_t channel_shared=(7*n+5)*D*sizeof(double);
    bool channels=Modal && channel_shared<=static_cast<size_t>(limit);
    size_t recompute_shared=(6*n+5)*D*sizeof(double);
    if(variant==1) {
        TORCH_CHECK(recompute_shared<=static_cast<size_t>(limit),"Experimental recompute kernel exceeds shared memory");
        C10_CUDA_CHECK(cudaFuncSetAttribute(modal_recompute<D>,cudaFuncAttributeMaxDynamicSharedMemorySize,limit));
    }
    if(variant==2) TORCH_CHECK(n<=threads,"Experimental primal register kernel requires n <= threads");
    if(channels) C10_CUDA_CHECK(cudaFuncSetAttribute(modal_channels<D>,cudaFuncAttributeMaxDynamicSharedMemorySize,limit));
    if(fits) C10_CUDA_CHECK(cudaFuncSetAttribute(advance<D,Modal,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,limit));
    if(chunk==0) chunk=steps>0?steps:1;
    for(int64_t start=0;start<steps || (start==0&&steps==0);start+=chunk) {
        int64_t end=std::min(start+chunk,steps);
        auto a=reinterpret_cast<KCValue<D>*>(workspace.data_ptr<double>());
        auto b=reinterpret_cast<const KCValue<D>*>(c.data_ptr<double>());
        auto wd=reinterpret_cast<const KCValue<D>*>(w.data_ptr<double>());
        if(variant==1) modal_recompute<D><<<B,dim3(32,D),recompute_shared,stream>>>(a,b,e.data_ptr<double>(),wd,audio.data_ptr<double>(),jac.data_ptr<double>(),n,start,end,count,stride);
        else if(variant==2) modal_register<D><<<B,threads,partial,stream>>>(a,b,e.data_ptr<double>(),wd,audio.data_ptr<double>(),jac.data_ptr<double>(),n,start,end,count,stride);
        else if(channels) modal_channels<D><<<B,dim3(32,D),channel_shared,stream>>>(a,b,e.data_ptr<double>(),wd,audio.data_ptr<double>(),jac.data_ptr<double>(),n,start,end,count,stride);
        else if(Modal && n<=threads) modal_register<D><<<B,threads,partial,stream>>>(a,b,e.data_ptr<double>(),wd,audio.data_ptr<double>(),jac.data_ptr<double>(),n,start,end,count,stride);
        else if(fits) advance<D,Modal,true><<<B,threads,shared,stream>>>(a,b,e.data_ptr<double>(),wd,audio.data_ptr<double>(),jac.data_ptr<double>(),n,start,end,count,stride,order);
        else advance<D,Modal,false><<<B,threads,partial,stream>>>(a,b,e.data_ptr<double>(),wd,audio.data_ptr<double>(),jac.data_ptr<double>(),n,start,end,count,stride,order);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {audio,jac,workspace.slice(1,0,2)};
}
std::vector<torch::Tensor> kc_cuda(torch::Tensor s,torch::Tensor c,torch::Tensor e,torch::Tensor w,
    int64_t steps,int64_t stride,bool modal,int64_t chunk,int64_t threads,int64_t order,int64_t variant) {
    c10::cuda::CUDAGuard guard(s.device());
    const int spatial_order=static_cast<int>(order);
    #define CUDA_DISPATCH(N) case N: return modal?launch<N,true>(s,c,e,w,steps,stride,chunk,threads,variant,spatial_order):launch<N,false>(s,c,e,w,steps,stride,chunk,threads,variant,spatial_order)
    switch(s.size(3)) {CUDA_DISPATCH(1); CUDA_DISPATCH(2); CUDA_DISPATCH(3); CUDA_DISPATCH(4); CUDA_DISPATCH(5); CUDA_DISPATCH(6); CUDA_DISPATCH(7); CUDA_DISPATCH(8); CUDA_DISPATCH(9); CUDA_DISPATCH(10); CUDA_DISPATCH(11);}
    #undef CUDA_DISPATCH
    TORCH_CHECK(false,"Unsupported dual column count");
}

std::vector<int64_t> kc_resources(bool modal,int64_t n,int64_t threads) {
    cudaFuncAttributes a;int limit=0,device=0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&limit,cudaDevAttrMaxSharedMemoryPerBlockOptin,device));
    int64_t shared=(4*n+5*(threads/32))*sizeof(KCValue<11>);
    int kind;
    if(modal&&(7*n+5)*11*8<=limit) {C10_CUDA_CHECK(cudaFuncGetAttributes(&a,modal_channels<11>));kind=3;shared=(7*n+5)*11*8;}
    else if(modal&&n<=threads) {C10_CUDA_CHECK(cudaFuncGetAttributes(&a,modal_register<11>));kind=0;shared=5*(threads/32)*sizeof(KCValue<11>);}
    else if(shared<=limit) {
        if(modal) C10_CUDA_CHECK(cudaFuncGetAttributes(&a,advance<11,true,true>));
        else C10_CUDA_CHECK(cudaFuncGetAttributes(&a,advance<11,false,true>));kind=1;
    } else {
        if(modal) C10_CUDA_CHECK(cudaFuncGetAttributes(&a,advance<11,true,false>));
        else C10_CUDA_CHECK(cudaFuncGetAttributes(&a,advance<11,false,false>));kind=2;shared=5*(threads/32)*sizeof(KCValue<11>);
    }
    return {kind,a.numRegs,static_cast<int64_t>(a.localSizeBytes),shared,a.maxThreadsPerBlock};
}

std::vector<int64_t> kc_variant_resources(int64_t variant,int64_t d,int64_t n,int64_t threads) {
    cudaFuncAttributes a;
    TORCH_CHECK((variant==1 && d>=1 && d<=11)||(variant==2 && d==1),"Invalid experimental variant/columns");
    if(variant==2) C10_CUDA_CHECK(cudaFuncGetAttributes(&a,modal_register<1>));
    else {
        #define ATTR(N) case N: C10_CUDA_CHECK(cudaFuncGetAttributes(&a,modal_recompute<N>)); break
        switch(d) {ATTR(1); ATTR(2); ATTR(3); ATTR(4); ATTR(5); ATTR(6); ATTR(7); ATTR(8); ATTR(9); ATTR(10); ATTR(11);}
        #undef ATTR
    }
    int64_t shared=variant==1?(6*n+5)*d*sizeof(double):5*(threads/32)*sizeof(KCValue<1>);
    return {a.numRegs,static_cast<int64_t>(a.localSizeBytes),shared,a.maxThreadsPerBlock};
}
