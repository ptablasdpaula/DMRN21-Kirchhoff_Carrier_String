#pragma once
#include <cmath>
#ifdef __CUDACC__
#define KC_HD __host__ __device__ __forceinline__
#else
#define KC_HD inline
#endif
// D=1 is primal-only; D=11 carries the ten parameter derivatives (eight material
// parameters, then the pluck and pickup positions). Pickup weights are KCValue<D>
// too, so the readout's product rule covers the pickup-position tangent.
template<int D> struct KCValue { double v[D]{}; };
template<int D> KC_HD KCValue<D> operator+(KCValue<D> a, const KCValue<D>& b) {
    for(int j=0;j<D;++j) a.v[j]+=b.v[j]; return a;
}
template<int D> KC_HD KCValue<D> operator-(KCValue<D> a, const KCValue<D>& b) {
    for(int j=0;j<D;++j) a.v[j]-=b.v[j]; return a;
}
template<int D> KC_HD KCValue<D> operator*(const KCValue<D>& a, const KCValue<D>& b) {
    KCValue<D> r; r.v[0]=a.v[0]*b.v[0];
    for(int j=1;j<D;++j) r.v[j]=a.v[j]*b.v[0]+a.v[0]*b.v[j]; return r;
}
template<int D> KC_HD KCValue<D> operator*(double a, KCValue<D> b) {
    for(int j=0;j<D;++j) b.v[j]*=a; return b;
}
template<int D> KC_HD KCValue<D> operator/(const KCValue<D>& a, const KCValue<D>& b) {
    KCValue<D> r; r.v[0]=a.v[0]/b.v[0];
    for(int j=1;j<D;++j) r.v[j]=(a.v[j]-r.v[0]*b.v[j])/b.v[0]; return r;
}
template<int D> KC_HD KCValue<D> kc_at(const KCValue<D>* u,int i,int n) {
    return i>=0 && i<n ? u[i] : KCValue<D>{};
}
// Odd extension about the simply supported ends: interior nodes 0..n-1, the
// ends sit at i=-1 and i=n, and beyond them u_{-1-m} = -u_{m-1}, u_{n+m} = -u_{n-m}.
template<int D> KC_HD KCValue<D> kc_at_odd(const KCValue<D>* u,int i,int n) {
    if(i>=0 && i<n) return u[i];
    if(i==-1 || i==n) return KCValue<D>{};
    return -1.0*(i<-1 ? u[-i-2] : u[2*n-i]);
}
// order 2: three-point second difference and its square. order 4: five-point
// (-1,16,-30,16,-1)/12 on the odd extension and its square, the nine-point
// (1,-32,316,-992,1414,-992,316,-32,1)/144; both equal D_xx applied twice.
template<int D, bool Modal> KC_HD void kc_terms(
    const KCValue<D>* p,const KCValue<D>* u,const KCValue<D>* c,
    const double* eigen,int i,int n,int order,KCValue<D>& linear,KCValue<D>& q) {
    KCValue<D> second,fourth,previous_second;
    if constexpr(Modal) {
        const auto lambda=eigen[i];
        second=lambda*u[i]; fourth=lambda*second; previous_second=lambda*p[i];
    } else if(order==4) {
        second=(-1.0/12)*(kc_at_odd(u,i-2,n)+kc_at_odd(u,i+2,n))+(4.0/3)*(kc_at_odd(u,i-1,n)+kc_at_odd(u,i+1,n))+(-2.5)*u[i];
        fourth=(1.0/144)*(kc_at_odd(u,i-4,n)+kc_at_odd(u,i+4,n))+(-32.0/144)*(kc_at_odd(u,i-3,n)+kc_at_odd(u,i+3,n))
              +(316.0/144)*(kc_at_odd(u,i-2,n)+kc_at_odd(u,i+2,n))+(-992.0/144)*(kc_at_odd(u,i-1,n)+kc_at_odd(u,i+1,n))
              +(1414.0/144)*u[i];
        previous_second=(-1.0/12)*(kc_at_odd(p,i-2,n)+kc_at_odd(p,i+2,n))+(4.0/3)*(kc_at_odd(p,i-1,n)+kc_at_odd(p,i+1,n))+(-2.5)*p[i];
    } else {
        second=(kc_at(u,i-1,n)-2.0*u[i]+kc_at(u,i+1,n));
        // Dxx squared: endpoint diagonal is five, not six (simply supported).
        fourth=(kc_at(u,i-2,n)-4.0*kc_at(u,i-1,n)+(i==0||i==n-1?5.0:6.0)*u[i]
                -4.0*kc_at(u,i+1,n)+kc_at(u,i+2,n));
        previous_second=(kc_at(p,i-1,n)-2.0*p[i]+kc_at(p,i+1,n));
    }
    linear=(2.0*c[1])*u[i]+c[2]*second-c[3]*fourth+c[4]*p[i]-c[5]*previous_second;
    q=c[7]*second;
}
