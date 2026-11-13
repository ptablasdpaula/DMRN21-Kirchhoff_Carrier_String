"""Independent exact one-step check for the scalar-cache inverse and adjoint."""
import torch


def test_cached_inverse_and_adjoint():
    torch.manual_seed(12)
    for n in (3, 7, 175):
        for _ in range(20):
            p = torch.randn(n, dtype=torch.float64, requires_grad=True)
            u = torch.randn(n, dtype=torch.float64, requires_grad=True)
            A = (torch.randn(n, dtype=torch.float64) * .3).requires_grad_()
            P = (torch.randn(n, dtype=torch.float64) * .2 + .8).requires_grad_()
            Q = (torch.randn(n, dtype=torch.float64) * .2).requires_grad_()
            m = torch.tensor(.8, dtype=torch.float64, requires_grad=True)
            linear = A * u + P * p
            q = Q * u
            q2, cross = q @ q, q @ linear
            alpha = (q @ p + m * cross) / (1 + m * q2)
            v = m * (linear - q * alpha)
            barp = torch.randn(n, dtype=torch.float64)
            barv = torch.randn(n, dtype=torch.float64)
            reference = torch.autograd.grad((u * barp + v * barv).sum(), (p, u, A, P, Q, m))
            with torch.no_grad():
                reconstructed_p = (v / m - A * u + q * alpha) / P
                reconstructed_l = v / m + q * alpha
                den = 1 + m * q2
                balpha = -m * (barv @ q)
                ba = balpha / den
                bb = -balpha * alpha * m / den
                bc = ba * m
                bl = m * barv + bc * q
                bq = -m * alpha * barv + ba * reconstructed_p + 2 * bb * q + bc * reconstructed_l
                gm = (barv @ v) / m + ba * cross - balpha * alpha * q2 / den
                actual = (ba * q + bl * P, barp + bl * A + bq * Q,
                          bl * u, bl * reconstructed_p, bq * u, gm)
                torch.testing.assert_close(reconstructed_p, p, rtol=1e-11, atol=1e-11)
                for observed, expected in zip(actual, reference):
                    torch.testing.assert_close(observed, expected, rtol=1e-11, atol=1e-11)


if __name__ == '__main__':
    test_cached_inverse_and_adjoint()
    print('PASS: cached inverse and adjoint agree with independent FP64 autograd')
