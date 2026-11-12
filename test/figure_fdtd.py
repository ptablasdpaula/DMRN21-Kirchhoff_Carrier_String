"""Independent spatial dual-number reference checks for the plain AD baseline."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from experiments.figure_comparison.fdtd import simulate
from experiments.figure_comparison.physical import (
    gradient_scales, physical_values, theta_from_physical,
)
from src.kc_string_forward import PARAMETER_COUNT, advance, dual_dot, initialize, nominal_parameters, pickup_weights
from src.parameters import material_multipliers, reference_parameters


def dual_reference(theta, intervals, rate, steps, excitation):
    """(steps+2, 11) pickup samples: the value and its ten multiplier tangents."""
    state, coefficient, loss = initialize(theta.detach(), intervals, rate, excitation)
    weights = pickup_weights(theta.detach(), intervals, True)
    output = [dual_dot(weights, state[0]), dual_dot(weights, state[1])]
    for _ in range(steps):
        state, loss, _ = advance(state, coefficient, weights, loss)
        output.append(dual_dot(weights, state[1]))
    return torch.stack(output)


def test_dual_reference():
    torch.manual_seed(749)
    for intervals, excitation in ((8, 'first_mode'), (16, 'raised_cosine'), (176, 'multimode'),
                                  (176, 'raised_cosine'), (24, 'raised_cosine:0.2')):
        rate, steps = 132300, 36
        params = reference_parameters()
        params = params * params.new_tensor([1.05, 1.1, 0.9, 1.02, 1.1, 0.9])
        params.requires_grad_()
        theta = material_multipliers(params)
        actual = simulate(theta, intervals, rate, steps, excitation)
        expected = dual_reference(theta, intervals, rate, steps, excitation)
        torch.testing.assert_close(actual, expected[:, 0], rtol=1e-10, atol=1e-13)
        weight = torch.randn_like(actual)
        actual_grad, = torch.autograd.grad((actual * weight).sum(), params)
        theta_grad = weight @ expected[:, 1:]
        mapping = torch.autograd.functional.jacobian(material_multipliers, params.detach())
        expected_grad = theta_grad @ mapping
        # Scale heterogeneous physical units before comparing six directions.
        torch.testing.assert_close(actual_grad * params, expected_grad * params,
                                   rtol=5e-9, atol=1e-12)
        # Direct multipliers with an off-centre pluck and pickup exercise the
        # two position columns (the pluck only moves a raised cosine).
        theta = (material_multipliers(params.detach()) * params.new_tensor([1.]*8 + [.9, 1.1])).requires_grad_()
        actual = simulate(theta, intervals, rate, steps, excitation)
        expected = dual_reference(theta, intervals, rate, steps, excitation)
        torch.testing.assert_close(actual, expected[:, 0], rtol=1e-10, atol=1e-13)
        weight = torch.randn_like(actual)
        actual_grad, = torch.autograd.grad((actual * weight).sum(), theta)
        torch.testing.assert_close(actual_grad, weight @ expected[:, 1:], rtol=5e-9, atol=1e-12)


def test_checkpointed_chunks():
    """Chunked torch.utils.checkpoint rematerialisation reproduces the plain path exactly, in both of its forms."""
    torch.manual_seed(905)
    params = reference_parameters()
    theta = (material_multipliers(params) * params.new_tensor([1.]*8 + [.9, 1.1]))
    batch = torch.stack((theta, theta * theta.new_tensor([1., 1.01, 1., 1., 1., .9, 1., 1., 1.02, .97])))
    for intervals, excitation in ((176, 'raised_cosine'), (16, 'multimode')):
        for steps, chunk in ((36, 7), (36, 36), (36, 100), (5, 2), (0, 3), (1, 1)):
            a = batch.detach().clone().requires_grad_()
            b = batch.detach().clone().requires_grad_()
            plain = simulate(a, intervals, 132300, steps, excitation)
            chunked = simulate(b, intervals, 132300, steps, excitation, checkpoint_steps=chunk)
            assert chunked.shape == (2, steps + 2)
            torch.testing.assert_close(chunked, plain, rtol=1e-13, atol=1e-16)
            c = batch.detach().clone().requires_grad_()
            reentrant = simulate(c, intervals, 132300, steps, excitation, checkpoint_steps=chunk,
                                 checkpoint_reentrant=True)
            torch.testing.assert_close(reentrant, plain, rtol=1e-13, atol=1e-16)
            if steps == 0:
                continue
            weight = torch.randn_like(plain)
            grad_plain, = torch.autograd.grad((plain * weight).sum(), a)
            grad_chunked, = torch.autograd.grad((chunked * weight).sum(), b)
            torch.testing.assert_close(grad_chunked, grad_plain, rtol=1e-12, atol=1e-15)
            # The reentrant form only supports .backward(), not autograd.grad(). Each chunk runs its own
            # backward pass, so the parameter gradients are summed in a different order: equal to rounding
            # (about 4e-11 on components that are small by cancellation), not bit for bit.
            (reentrant * weight).sum().backward()
            torch.testing.assert_close(c.grad, grad_plain, rtol=5e-9, atol=1e-12)
    for bad in (0, -1, 2.0, True):
        try:
            simulate(batch, 16, 132300, 4, checkpoint_steps=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'checkpoint_steps={bad!r} accepted')


def test_sine_basis():
    """Stepping in the discrete sine basis reproduces the grid scheme: waveform and ten gradients, plain and chunked."""
    torch.manual_seed(518)
    params = reference_parameters()
    theta = (material_multipliers(params) * params.new_tensor([1.]*8 + [.9, 1.1]))
    batch = torch.stack((theta, theta * theta.new_tensor([1., 1.01, 1., 1., 1., .9, 1., 1., 1.02, .97])))
    for intervals, excitation, order in ((176, 'raised_cosine', 4), (16, 'multimode', 4), (24, 'raised_cosine:0.2', 2),
                                         (8, 'first_mode', 4)):
        for steps, chunk in ((36, None), (36, 7), (5, 2), (0, None), (1, 1)):
            a = batch.detach().clone().requires_grad_()
            b = batch.detach().clone().requires_grad_()
            grid = simulate(a, intervals, 132300, steps, excitation, order)
            sine = simulate(b, intervals, 132300, steps, excitation, order, checkpoint_steps=chunk, basis='sine')
            assert sine.shape == (2, steps + 2)
            torch.testing.assert_close(sine, grid, rtol=1e-9, atol=1e-13)
            weight = torch.randn_like(grid)
            grad_grid, = torch.autograd.grad((grid * weight).sum(), a)
            grad_sine, = torch.autograd.grad((sine * weight).sum(), b)
            torch.testing.assert_close(grad_sine, grad_grid, rtol=5e-8, atol=1e-12)
    try:
        simulate(batch, 16, 132300, 4, basis='modal')
    except ValueError:
        pass
    else:
        raise AssertionError("basis='modal' accepted")


def test_batch_and_empty_steps():
    params = reference_parameters()
    theta = material_multipliers(params)
    batch = torch.stack((theta, theta * theta.new_tensor([1., 1.01, 1., 1., 1., .9, 1., 1., 1.02, .97])))
    batch.requires_grad_()
    for steps in (0, 1, 20):
        out = simulate(batch, intervals=16, rate=132300, steps=steps)
        individual = torch.stack([simulate(row, intervals=16, rate=132300, steps=steps) for row in batch])
        torch.testing.assert_close(out, individual, rtol=1e-13, atol=1e-15)
        actual_grad, = torch.autograd.grad(out.square().sum(), batch, retain_graph=True)
        expected_grad, = torch.autograd.grad(individual.square().sum(), batch)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-11, atol=1e-15)


def test_physical_coordinates():
    from benchmarks.exact_methods import varied_material

    torch.manual_seed(414)
    values = physical_values(2).requires_grad_()
    assert values.shape == (2, PARAMETER_COUNT)
    torch.testing.assert_close(values.detach(), varied_material(2), rtol=0, atol=0)
    theta = theta_from_physical(values)
    material = theta * nominal_parameters(theta.device)
    # All ten native coordinates are directly trainable; the benchmark batch
    # moves every one of them, so the pluck and pickup sit off centre here.
    torch.testing.assert_close(material, values)
    assert bool((values[:, 8:] != .5).all())
    audio = simulate(theta, intervals=176, rate=132300, steps=36)
    weights = torch.randn_like(audio)
    actual_grad, = torch.autograd.grad((audio * weights).sum(), values)
    expected_grad = []
    for row, row_weights in zip(theta, weights):
        dual = dual_reference(row, 176, 132300, 36, 'multimode')
        torch.testing.assert_close(dual[:, 0], audio[len(expected_grad)], rtol=1e-10, atol=1e-13)
        theta_grad = row_weights @ dual[:, 1:]
        expected_grad.append(theta_grad / nominal_parameters())
    scale = gradient_scales(values.device)
    assert scale.shape == (PARAMETER_COUNT,)
    torch.testing.assert_close(actual_grad * scale, torch.stack(expected_grad) * scale,
                               rtol=5e-9, atol=1e-12)


def test_cuda_replay():
    from src.adjoint import simulate as replay
    torch.manual_seed(721)
    device = torch.device('cuda')
    values = physical_values(2).to(device)
    for excitation in ('multimode', 'raised_cosine'):
        a = values.detach().clone().requires_grad_()
        b = values.detach().clone().requires_grad_()
        theta_a = theta_from_physical(a)
        theta_b = theta_from_physical(b)
        output_a = simulate(theta_a, intervals=176, rate=132300, steps=48, excitation=excitation)
        output_b = replay(theta_b, intervals=176, rate=132300, steps=48, excitation=excitation)
        weight = torch.randn_like(output_a)
        grad_a, = torch.autograd.grad((output_a * weight).sum(), a)
        grad_b, = torch.autograd.grad((output_b * weight).sum(), b)
        torch.testing.assert_close(output_a, output_b, rtol=1e-8, atol=1e-12)
        scale = gradient_scales(device)
        torch.testing.assert_close(grad_a * scale, grad_b * scale, rtol=5e-7, atol=1e-11)
    print('PASS: CUDA spatial autodiff agrees with replay waveform and ten physical gradients (multimode, raised cosine)')


def test_cuda_compiled(modes=(None,)):
    """torch.compile blocks reproduce the eager loop: waveform and ten gradients, plain and checkpointed."""
    torch.manual_seed(312)
    device = torch.device('cuda')
    values = physical_values(2).to(device)
    scale = gradient_scales(device)
    for mode in modes:
        # 77 steps in 32-step chunks of 8-step blocks: whole blocks, a remainder block and a short last chunk.
        for chunk, reentrant, basis in ((None, False, 'grid'), (32, False, 'grid'), (32, True, 'grid'),
                                        (None, False, 'sine'), (32, True, 'sine')):
            a = values.detach().clone().requires_grad_()
            b = values.detach().clone().requires_grad_()
            eager = simulate(theta_from_physical(a), 176, 132300, 77, 'raised_cosine')
            compiled = simulate(theta_from_physical(b), 176, 132300, 77, 'raised_cosine', checkpoint_steps=chunk,
                                compile_block=8, compile_mode=mode, checkpoint_reentrant=reentrant, basis=basis)
            weight = torch.randn_like(eager)
            grad_a, = torch.autograd.grad((eager * weight).sum(), a)
            # The reentrant checkpoint only supports .backward(), not autograd.grad().
            (compiled * weight).sum().backward()
            grad_b = b.grad
            torch.testing.assert_close(compiled, eager, rtol=1e-9 if basis == 'sine' else 1e-10, atol=1e-13)
            torch.testing.assert_close(grad_b * scale, grad_a * scale, rtol=5e-8, atol=1e-12)
            print(f'ok: mode={mode or "default"} checkpoint_steps={chunk} reentrant={reentrant} basis={basis}', flush=True)
    print('PASS: torch.compile blocks agree with the eager spatial loop, waveform and ten physical gradients '
          f'(modes: {", ".join(m or "default" for m in modes)})')


if __name__ == '__main__':
    test_dual_reference()
    test_checkpointed_chunks()
    test_sine_basis()
    test_batch_and_empty_steps()
    test_physical_coordinates()
    print('PASS: plain spatial FDTD waveform and all ten material and six perceptual gradients match independent dual reference; '
          'batches isolated; checkpointed chunks (non-reentrant and reentrant) reproduce the plain values and gradients')
    if '--cuda' in sys.argv:
        if not torch.cuda.is_available():
            raise RuntimeError('--cuda requires a CUDA device')
        test_cuda_replay()
        if '--compiled' in sys.argv:
            test_cuda_compiled()
        if '--compiled-cudagraphs' in sys.argv:
            test_cuda_compiled(('reduce-overhead',))
