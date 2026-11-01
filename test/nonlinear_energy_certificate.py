#!/usr/bin/env python3
"""Check the non-SAV update's damped nonlinear energy identity.

Uses only the standard library and an independently written spatial-grid
update. This is a numerical check of the derivation, not a proof by testing.
"""
import math


def dot(x, y):
    return math.fsum(a * b for a, b in zip(x, y))


def add(x, y, ax=1.0, ay=1.0):
    return [ax * a + ay * b for a, b in zip(x, y)]


def lap_positive(x, h):
    z = [0.0, *x, 0.0]
    return [(2*z[i+1] - z[i] - z[i+2]) / h**2 for i in range(len(x))]


def run(sigma0, sigma1, amplitude):
    n, length, rate = 64, 0.65, 132300.0
    h, k = length/n, 1/rate
    rho, radius, young, tension = 8050.0, 0.0003556, 174e9, 75.0
    area = math.pi * radius**2
    inertia = math.pi * radius**4/4
    mass = rho * area
    bending = young * inertia
    nonlinear_energy_coefficient = young * area * h**2/(8*length)
    max_lam = 4/h**2 * math.sin(math.pi*(n-1)/(2*n))**2
    cfl = (k*k*(tension*max_lam + bending*max_lam**2)/mass
           + 4*k*sigma1*max_lam)
    assert cfl < 4

    def energy(a, b):
        d = add(a, b, 1, -1)
        velocity = [v/k for v in d]
        db = lap_positive(b, h)
        ddb = lap_positive(db, h)
        linear = .5*h*(tension*dot(a, db) + bending*dot(a, ddb))
        nonlinear = nonlinear_energy_coefficient*dot(a, db)**2
        return (.5*mass*h*dot(velocity, velocity) + linear + nonlinear
                - .5*mass*h*k*sigma1*dot(velocity, lap_positive(velocity, h)))

    previous = [amplitude * math.sin(math.pi*i/n) for i in range(1, n)]
    current = [v*(1-1e-5) for v in previous]
    initial = energy(current, previous)
    residual_max, relative_growth_max = 0.0, 0.0
    inv_mass = 1/(mass*(1+sigma0*k))
    for _ in range(256):
        dc, dp = lap_positive(current, h), lap_positive(previous, h)
        ddc = lap_positive(dc, h)
        linear = [2*mass*u - k*k*(tension*du+bending*ddu)
                  - 2*mass*sigma1*k*du + (mass*sigma0*k-mass)*p
                  + 2*mass*sigma1*k*dp_i
                  for u, du, ddu, p, dp_i in zip(current, dc, ddc, previous, dp)]
        q_scale = -k * math.sqrt(h*young*area/length)/2
        q = [q_scale * v for v in dc]
        rhs = add(linear, q, 1, -dot(q, previous))
        temp = [inv_mass * v for v in rhs]
        correction = inv_mass * dot(q, temp)/(1 + inv_mass*dot(q, q))
        following = add(temp, q, 1, -correction)
        old, new = energy(current, previous), energy(following, current)
        centred_velocity = [(a-b)/(2*k) for a, b in zip(following, previous)]
        dissipation = 2*mass*h*k*(sigma0*dot(centred_velocity, centred_velocity)
                                  + sigma1*dot(centred_velocity, lap_positive(centred_velocity, h)))
        residual_max = max(residual_max, abs(new-old+dissipation))
        relative_growth_max = max(relative_growth_max, (new-old)/initial)
        previous, current = current, following
    assert residual_max <= 1e-10 * initial, (residual_max, initial)
    assert relative_growth_max <= 1e-10, relative_growth_max
    print(f'sigma0={sigma0:g}, sigma1={sigma1:g}, amplitude={amplitude:g}: '
          f'ratio={cfl:.6g}, energy0={initial:.9g} J, '
          f'max identity residual={residual_max:.3g} J, '
          f'max relative energy growth={relative_growth_max:.3g}')


if __name__ == '__main__':
    for s0, s1 in ((0.0, 0.0), (1.0, 0.0), (0.0, 1e-4), (1.0, 1e-4)):
        for amp in (.002, .02):
            run(s0, s1, amp)
