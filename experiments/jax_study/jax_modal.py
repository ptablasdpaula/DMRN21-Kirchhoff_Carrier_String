"""JAX implementation of the full nonlinear modal recurrence for benchmarking.

The physical setup is carried out in JAX. Float64 and all modal components are
retained. This is a research implementation, not a production solver.
"""
import math
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax import lax


def renderer(values, steps, intervals=176, rate=132300, order=None):
    from src.kc_string_forward import SPATIAL_ORDER, STENCILS, discrete_eigenvalues
    order = SPATIAL_ORDER if order is None else order
    n = intervals - 1
    ii = jnp.arange(1, intervals, dtype=jnp.float64)
    eig = jnp.asarray(discrete_eigenvalues(intervals, order).numpy())
    transform = jnp.sqrt(2 / intervals) * jnp.sin(jnp.pi * ii[:,None] * ii[None,:] / intervals)
    weights = transform[n // 2]
    modes = jnp.arange(1, min(8, n) + 1, dtype=jnp.float64)
    shape = jnp.sum(jnp.sin(jnp.pi * ii[:, None] * modes / intervals) / modes**2, axis=1)
    shape = shape / jnp.max(jnp.abs(shape))
    f0, inharm, amplitude, radius, sigma0, sigma1 = jnp.moveaxis(values, -1, 0)
    density = 8050.; length = 1.; dt = 1 / rate
    area = jnp.pi * radius**2
    inertia = jnp.pi * radius**4 / 4
    tension = 4 * density * area * length**2 * f0**2
    modulus = inharm * tension * length**2 / (jnp.pi**2 * inertia)
    spacing = length / intervals
    h2 = 1 / spacing**2
    mass = density * area
    previous = amplitude[:,None] * shape[None,:]
    def second(x):
        if order == 2:
            return (jnp.pad(x[:, :-1], ((0,0),(1,0))) - 2*x + jnp.pad(x[:, 1:], ((0,0),(0,1)))) * h2
        # Five-point stencil closed by odd reflection about the fixed ends.
        zero = jnp.zeros_like(x[:, :1])
        padded = jnp.concatenate((-x[:, :1], zero, x, zero, -x[:, -1:]), axis=1)
        weights = STENCILS[4]
        total = weights[0] * padded[:, :n]
        for m in range(1, len(weights)):
            total = total + weights[m] * padded[:, m:m + n]
        return total * h2
    d2 = second(previous)
    d4 = second(d2)
    nonlinear = -jnp.sum(previous*d2, axis=1, keepdims=True)*d2
    c2 = tension/mass
    k2 = modulus*inertia/mass
    nlscale = 0.5*modulus*spacing/(length*density)
    current = previous + 0.5*dt**2*(spacing*c2[:,None]*d2-spacing*k2[:,None]*d4+spacing*nlscale[:,None]*nonlinear)
    p = previous @ transform
    u = current @ transform
    c1=mass
    c2=(tension*dt**2+2*mass*sigma1*dt)*h2
    c3=modulus*inertia*dt**2*h2*h2
    c4=mass*sigma0*dt-mass
    c5=2*mass*sigma1*dt*h2
    m=1/(mass+mass*sigma0*dt)
    c7=0.5*dt*jnp.sqrt(spacing*modulus*area/length)*h2
    A=2*c1[:,None]+eig[None,:]*c2[:,None]-eig[None,:]**2*c3[:,None]
    P=c4[:,None]-eig[None,:]*c5[:,None]
    Q=eig[None,:]*c7[:,None]
    def step(state, unused):
        p,u=state
        linear=A*u+P*p
        q=Q*u
        alpha=(jnp.sum(q*p,axis=1)+m*jnp.sum(q*linear,axis=1))/(1+m*jnp.sum(q*q,axis=1))
        v=m[:,None]*(linear-q*alpha[:,None])
        return (u,v),jnp.sum(v*weights,axis=1)
    (_, _), samples=lax.scan(step,(p,u),None,length=steps)
    initial=jnp.stack((p@weights,u@weights),axis=1)
    return jnp.concatenate((initial,jnp.swapaxes(samples,0,1)),axis=1)


def forward_jacobian(values, steps, count):
    def function(x): return renderer(x, steps)
    seed=jnp.eye(values.shape[1],dtype=values.dtype)[:count]
    seeds=jnp.broadcast_to(seed[:,None,:],(count,values.shape[0],values.shape[1]))
    # Each JVP differentiates every item in a single coordinate direction;
    # items are independent, so this avoids a mostly-zero B*P Jacobian.
    ys, dys = jax.vmap(lambda direction: jax.jvp(function,(values,),(direction,)))(seeds)
    return ys[0],jnp.transpose(dys,(1,2,0))
