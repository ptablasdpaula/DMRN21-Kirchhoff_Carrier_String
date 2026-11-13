"""Check the native first-order API against both independent PyTorch references."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'benchmarks')]
import autodiff_memory as reference
reference.load_torch()
import torch
from kc_string_forward import ForwardString, DualReference


def objective(audio):
    # A time-domain and a spectral loss, both outside the native simulator.
    normalized = audio / .002
    return normalized.square().mean() + .03 * normalized.mean() + .001 * torch.fft.rfft(normalized).abs().square().mean()


def main():
    model = ForwardString()
    dual_reference = DualReference(block_steps=32)
    # First check every output tangent directly against eager forward AD.
    theta = torch.ones(8, dtype=torch.float64, requires_grad=True)
    expected_jacobian = torch.func.jacfwd(lambda x: reference.simulate(x * reference.defaults(), 16, 128, 44100, 'reverse', 16))(theta)
    audio, jacobian = model.output_and_tangents(theta, 16 / 44100)
    torch.testing.assert_close(jacobian, expected_jacobian, rtol=1e-6, atol=1e-10)
    assert torch.autograd.gradcheck(lambda x: model(x, 16 / 44100), (theta,), eps=1e-5, atol=1e-7, rtol=1e-4)
    # Checkpointed reverse-mode remains a validation path only.
    reference.compile_step_and_validate(argparse.Namespace(method='checkpoint_reentrant', intervals=128, rate=44100))
    cases = [torch.ones(8, dtype=torch.float64),
             torch.tensor([1.02, .98, 1.01, 1.03, 1.01, .9, 1.1, .95], dtype=torch.float64)]
    for parameters in cases:
        variable = parameters.clone().requires_grad_()
        expected = reference.simulate(variable * reference.defaults(), 273, 128, 44100, 'checkpoint_reentrant', 128)
        expected_loss = objective(expected)
        expected_loss.backward()
        theta = parameters.clone().requires_grad_()
        actual = model(theta, 273 / 44100)
        actual_loss = objective(actual)
        actual_loss.backward()
        torch.testing.assert_close(actual, expected.detach(), rtol=1e-8, atol=1e-12)
        torch.testing.assert_close(actual_loss, expected_loss.detach(), rtol=1e-9, atol=1e-12)
        torch.testing.assert_close(theta.grad, variable.grad, rtol=1e-6, atol=1e-9)
        # Compiled nine-column PyTorch dual arithmetic is a second reference.
        dual_audio, dual_loss, dual_gradient = dual_reference.loss_and_gradient(parameters, 273 / 44100)
        torch.testing.assert_close(actual, dual_audio, rtol=1e-8, atol=1e-12)
        raw_audio, tangents = model.output_and_tangents(parameters, 273 / 44100)
        mse_gradient = (2 * raw_audio / .002**2 / raw_audio.numel()) @ tangents
        torch.testing.assert_close(mse_gradient, dual_gradient, rtol=1e-6, atol=1e-9)
        print('Native vs compiled dual/checkpointed reverse passed; max gradient error:', (theta.grad-variable.grad).abs().max().item())
    # Subsampling must select identical signal/tangent rows, including startup.
    audio, tangents = model.output_and_tangents(cases[0], 273 / 44100)
    for stride in (2, 7, 1000):
        sampled = ForwardString(output_stride=stride)
        a, j = sampled.output_and_tangents(cases[0], 273 / 44100)
        torch.testing.assert_close(a, audio[::stride], rtol=0, atol=0)
        torch.testing.assert_close(j, tangents[::stride], rtol=0, atol=0)
    assert model(cases[0], 0).numel() == 2
    # No shared mutable numerical state: independent simulations can run in threads.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        parallel = list(pool.map(lambda p: model.output_and_tangents(p, .001), cases))
    for parameters, (parallel_audio, parallel_tangents) in zip(cases, parallel):
        serial_audio, serial_tangents = model.output_and_tangents(parameters, .001)
        torch.testing.assert_close(parallel_audio, serial_audio, rtol=0, atol=0)
        torch.testing.assert_close(parallel_tangents, serial_tangents, rtol=0, atol=0)
    # Chain through a parameter transform and use ordinary optimizers/closures.
    for optimizer_class in (torch.optim.Adam, torch.optim.LBFGS):
        log_theta = torch.nn.Parameter(torch.zeros(8, dtype=torch.float64))
        options = dict(lr=1e-3)
        if optimizer_class is torch.optim.LBFGS:
            options['max_iter'] = 2
        optimizer = optimizer_class([log_theta], **options)
        def closure():
            optimizer.zero_grad()
            loss = objective(model(log_theta.exp(), .001))
            loss.backward()
            return loss
        optimizer.step(closure)
        assert torch.isfinite(log_theta).all() and log_theta.detach().abs().max() > 0
    # First-order contract: the saved numerical Jacobian is not a Hessian graph.
    theta = cases[0].clone().requires_grad_()
    first = torch.autograd.grad(model(theta, .001).sum(), theta, create_graph=True)[0]
    try:
        torch.autograd.grad(first.sum(), theta)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Second derivatives must not silently return an incorrect result')
    print('Jacobian, gradcheck, subsampling, Adam/L-BFGS and first-order contract checks passed.')


if __name__ == '__main__':
    main()
