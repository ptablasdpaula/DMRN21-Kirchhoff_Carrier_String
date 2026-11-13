#include <torch/extension.h>
#include <vector>
#include "kc_dual.h"
using torch::Tensor;
#ifdef WITH_CUDA
std::vector<int64_t> kc_resources(bool,int64_t,int64_t);
std::vector<Tensor> kc_cuda(Tensor,Tensor,Tensor,Tensor,int64_t,int64_t,bool,int64_t,int64_t,int64_t,int64_t);
std::vector<int64_t> kc_variant_resources(int64_t,int64_t,int64_t,int64_t);
#endif

template<int D,bool Modal> std::vector<Tensor> cpu(Tensor states,Tensor coeff,Tensor eigen,Tensor weights,
                                                int64_t steps,int64_t stride,int order) {
    const int B=states.size(0),n=states.size(2);
    const int64_t count=(steps+1)/stride+1;
    auto audio=torch::empty({B,count},states.options());
    auto final_states=torch::empty_like(states);
    auto final_data=reinterpret_cast<KCValue<D>*>(final_states.data_ptr<double>());
    auto tangents=torch::empty({B,count,D-1},states.options());
    using V=KCValue<D>;
    const V* input=reinterpret_cast<const V*>(states.data_ptr<double>());
    const V* coefficients=reinterpret_cast<const V*>(coeff.data_ptr<double>());
    const auto ev=eigen.data_ptr<double>();
    const V* pickup=reinterpret_cast<const V*>(weights.data_ptr<double>());
    auto y=audio.data_ptr<double>(),jac=tangents.data_ptr<double>();
    for(int b=0;b<B;++b) {
        std::vector<V> p(input+b*2*n,input+b*2*n+n),u(input+b*2*n+n,input+(b+1)*2*n),l(n),q(n);
        const V* c=coefficients+b*8;const V* w=pickup+b*n;
        std::vector<V> A(n),P(n),Q(n);
        if constexpr(Modal) for(int i=0;i<n;++i) {
            A[i]=2.0*c[1]+ev[i]*c[2]-(ev[i]*ev[i])*c[3];
            P[i]=c[4]-ev[i]*c[5];Q[i]=ev[i]*c[7];
        }
        auto record=[&](V v,int64_t t) {if(t%stride==0) {
            auto row=b*count+t/stride; y[row]=v.v[0];
            for(int j=1;j<D;++j) jac[row*(D-1)+j-1]=v.v[j];
        }};
        V first{},second{};
        for(int i=0;i<n;++i) {first=first+w[i]*p[i];second=second+w[i]*u[i];}
        record(first,0);record(second,1);
        for(int64_t t=0;t<steps;++t) {
            V a{},bb{},cc{};
            for(int i=0;i<n;++i) {
                if constexpr(Modal) {l[i]=A[i]*u[i]+P[i]*p[i];q[i]=Q[i]*u[i];}
                else kc_terms<D,false>(p.data(),u.data(),c,ev,i,n,order,l[i],q[i]);
                a=a+q[i]*p[i]; bb=bb+q[i]*q[i]; cc=cc+q[i]*l[i];
            }
            V denom=c[6]*bb;denom.v[0]+=1;
            const V alpha=(a+c[6]*cc)/denom;
            V sample{};
            for(int i=0;i<n;++i) {p[i]=c[6]*(l[i]-q[i]*alpha);sample=sample+w[i]*p[i];}
            record(sample,t+2);p.swap(u);
        }
        std::copy(p.begin(),p.end(),final_data+b*2*n);
        std::copy(u.begin(),u.end(),final_data+b*2*n+n);
    }
    return {audio,tangents,final_states};
}
std::vector<Tensor> simulate(Tensor s,Tensor c,Tensor e,Tensor w,int64_t steps,int64_t stride,
                            bool modal,int64_t chunk,int64_t threads,int64_t order,int64_t variant) {
    TORCH_CHECK(s.dim()==4 && s.size(0)>0 && s.size(1)==2 && s.size(2)>=3,"states: (batch,2,N-1,columns)");
    TORCH_CHECK(order==2||order==4,"Spatial order must be 2 or 4");
    const auto d=s.size(3);
    TORCH_CHECK(d>=1&&d<=11,"Expected one to eleven dual columns");
    TORCH_CHECK(c.dim()==3 && c.size(0)==s.size(0)&&c.size(1)==8&&c.size(2)==d,"Invalid coefficients");
    TORCH_CHECK(e.dim()==1&&e.size(0)==s.size(2),"Invalid eigenvalue table");
    TORCH_CHECK(w.dim()==3&&w.size(0)==s.size(0)&&w.size(1)==s.size(2)&&w.size(2)==d,"Pickup weights: (batch,N-1,columns)");
    for(const auto& t:{s,c,e,w}) {
        TORCH_CHECK(t.device()==s.device()&&t.scalar_type()==torch::kFloat64&&t.is_contiguous(),"Expected contiguous float64 tensors on one device");
        TORCH_CHECK(!t.requires_grad(),"Explicit tangents only");
    }
    TORCH_CHECK(steps>=0&&stride>0&&chunk>=0&&(threads==128||threads==256)&&variant>=0&&variant<=2,"Invalid execution settings");
    TORCH_CHECK(variant==0 || (s.is_cuda() && modal), "Experimental variants require CUDA modal execution");
    TORCH_CHECK(variant!=2 || d==1, "Register variant is for primal-only execution");
    if(s.is_cuda()) {
#ifdef WITH_CUDA
        return kc_cuda(s,c,e,w,steps,stride,modal,chunk,threads,order,variant);
#else
        TORCH_CHECK(false,"Extension was built without CUDA");
#endif
    }
    TORCH_CHECK(s.device().is_cpu(),"Only CPU/CUDA supported");
    const int spatial_order=static_cast<int>(order);
    #define CPU_DISPATCH(N) case N: return modal?cpu<N,true>(s,c,e,w,steps,stride,spatial_order):cpu<N,false>(s,c,e,w,steps,stride,spatial_order)
    switch(d) {CPU_DISPATCH(1); CPU_DISPATCH(2); CPU_DISPATCH(3); CPU_DISPATCH(4); CPU_DISPATCH(5); CPU_DISPATCH(6); CPU_DISPATCH(7); CPU_DISPATCH(8); CPU_DISPATCH(9); CPU_DISPATCH(10); CPU_DISPATCH(11);}
    #undef CPU_DISPATCH
    TORCH_CHECK(false,"Unsupported dual column count");
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("simulate",&simulate,pybind11::arg("states"),pybind11::arg("coefficients"),pybind11::arg("eigen"),pybind11::arg("weights"),pybind11::arg("steps"),pybind11::arg("stride"),pybind11::arg("modal"),pybind11::arg("chunk"),pybind11::arg("threads"),pybind11::arg("order"),pybind11::arg("variant")=0,pybind11::call_guard<pybind11::gil_scoped_release>());
#ifdef WITH_CUDA
    m.def("resources",&kc_resources);
    m.def("variant_resources",&kc_variant_resources);
#endif
}
