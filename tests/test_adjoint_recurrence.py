"""The kernel's backward recurrence is reverse-mode autodiff of its forward step, state by state.

``adjoint.cu`` carries one adjoint state per mode backwards through the time
steps. This test transcribes its forward and backward steps into plain PyTorch
(CPU, no kernel) and checks, at every time step, that the recurrence's adjoint
state equals autograd's derivative of the loss with respect to that state, and
that its accumulated coefficient gradients equal autograd's. Hand-written
adjoint and automatic differentiation are then one computation: the discrete
adjoint of the scheme.
"""

import torch


def forward(p, u, A, P, Q, m, weights, steps):
    """The step of adjoint.cu's forward_chunk; returns every state and the pickup samples."""
    states = [p, u]
    for _ in range(steps):
        lin = A * u + P * p
        q = Q * u
        alpha = ((q * p).sum() + m * (q * lin).sum()) / (1 + m * (q * q).sum())
        p, u = u, m * (lin - q * alpha)
        states.append(u)
    return states, torch.stack([(weights * state).sum() for state in states])


def adjoint(states, A, P, Q, m, weights, gout):
    """The backward recurrence of adjoint.cu's backward_chunk, line for line.

    Returns the adjoint state of every forward state (the kernel's ``bv``:
    what arrives from later steps plus the direct pickup term) and the
    gradients it accumulates for A, P, Q and m.
    """
    steps = len(states) - 2
    bp = torch.zeros_like(states[0])
    bu = torch.zeros_like(states[0])
    ga, gp, gq = (torch.zeros_like(states[0]) for _ in range(3))
    gm = torch.zeros((), dtype=states[0].dtype)
    lambdas = [None] * len(states)
    for t in range(steps - 1, -1, -1):
        p, u = states[t], states[t + 1]
        lin = A * u + P * p
        q = Q * u
        bv = bu + weights * gout[t + 2]
        lambdas[t + 2] = bv
        z = [(q * p).sum(), (q * q).sum(), (q * lin).sum(), (bv * q).sum(), (bv * lin).sum()]
        den = 1 + m * z[1]
        alpha = (z[0] + m * z[2]) / den
        balpha = -m * z[3]
        ba = balpha / den
        bc = ba * m
        bb = -balpha * alpha * m / den
        gm = gm + z[4] - alpha * z[3] + ba * z[2] - balpha * alpha * z[1] / den
        bl = m * bv + bc * q
        bq = -m * alpha * bv + ba * p + 2 * bb * q + bc * lin
        ga, gp, gq = ga + bl * u, gp + bl * p, gq + bq * u
        bp, bu = ba * q + bl * P, bp + bl * A + bq * Q
    lambdas[1] = bu + weights * gout[1]
    lambdas[0] = bp + weights * gout[0]
    return lambdas, (ga, gp, gq, gm)


def test_adjoint_states_equal_autograd():
    torch.manual_seed(1112)
    for modes, steps in ((7, 1), (15, 40), (175, 300)):
        # Coefficients of the magnitude the scheme produces: an oscillator per mode, weakly coupled.
        A = (2 - torch.rand(modes, dtype=torch.float64)).requires_grad_()
        P = (-1 + 1e-3 * torch.rand(modes, dtype=torch.float64)).requires_grad_()
        Q = (0.3 * torch.rand(modes, dtype=torch.float64)).requires_grad_()
        m = torch.tensor(0.999, dtype=torch.float64, requires_grad=True)
        weights = torch.randn(modes, dtype=torch.float64)
        p0 = torch.randn(modes, dtype=torch.float64).requires_grad_()
        u0 = (p0.detach() + 1e-2 * torch.randn(modes, dtype=torch.float64)).requires_grad_()
        gout = torch.randn(steps + 2, dtype=torch.float64)

        states, audio = forward(p0, u0, A, P, Q, m, weights, steps)
        for state in states:
            state.retain_grad()
        (audio * gout).sum().backward()

        with torch.no_grad():
            lambdas, (ga, gp, gq, gm) = adjoint([s.detach() for s in states], A, P, Q, m, weights, gout)
        scale = max(float(s.grad.abs().max()) for s in states)
        for step, (lam, state) in enumerate(zip(lambdas, states, strict=False)):
            torch.testing.assert_close(
                lam,
                state.grad,
                rtol=1e-9,
                atol=1e-11 * scale,
                msg=f"adjoint state {step} of {steps + 2}",
            )
        for name, ours, theirs in (("A", ga, A.grad), ("P", gp, P.grad), ("Q", gq, Q.grad), ("m", gm, m.grad)):
            torch.testing.assert_close(
                ours,
                theirs,
                rtol=1e-9,
                atol=1e-11 * float(theirs.abs().max()),
                msg=f"gradient of {name}",
            )
