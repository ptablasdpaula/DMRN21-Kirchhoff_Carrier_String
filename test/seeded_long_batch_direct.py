"""Direct CUDA seeded/replay check for the saved four-second B128 result."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from benchmarks.exact_methods import BOX, GRID, coordinates_and_seeds, filter_audio, measure, metadata, varied_parameters
from benchmarks.reconstruction import relative_error, render
from src.accelerated import evaluate
from src.optimization import criterion_for
from src.synthesis import Renderer

torch.set_num_threads(1)

class SeededAudio(torch.autograd.Function):
    """The same coordinate seeding and contraction as the recorded benchmark."""
    @staticmethod
    def forward(ctx, values, steps, count):
        theta, seeds = coordinates_and_seeds(values)
        audio, jac = evaluate(theta, GRID.intervals, GRID.integration_rate, steps, 1,
                              'multimode', 'modal', True, seeds=seeds[:, :, :count])
        ctx.save_for_backward(jac)
        ctx.count = values.shape[1]
        return audio

    @staticmethod
    def backward(ctx, grad):
        jac, = ctx.saved_tensors
        selected = torch.einsum('bt,btd->bd', grad.contiguous(), jac)
        out = grad.new_zeros((grad.shape[0], ctx.count))
        out[:, :selected.shape[1]] = selected
        return out, None, None

def evaluate_case(values, renderer, criterion, method, count):
    p = values.detach().clone().requires_grad_()
    if method == 'seeded':
        raw = SeededAudio.apply(p, round(4 * GRID.integration_rate), count)
        audio = filter_audio(raw, renderer)
    else:
        audio = render(p, renderer, 'replay', 256)
    loss = torch.stack([criterion(row) for row in audio])
    grad, = torch.autograd.grad(loss.sum(), p)
    return audio.detach(), loss.detach(), grad.detach()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    assert torch.cuda.is_available()
    values = varied_parameters(128).cuda()
    target = Renderer(GRID, 4.).synthesize(BOX.defaults()).detach().cuda()
    renderer = Renderer(GRID, 4., device='cuda')
    criterion = criterion_for(target, GRID.sample_rate)
    ref_audio, ref_loss, ref_grad = evaluate_case(values, renderer, criterion, 'replay', 6)
    result = {'metadata': {**metadata(), 'status': 'running', 'case': 'batch 128, four seconds, full loss',
                           'reference': 'CUDA replay adjoint checkpoint 256'}, 'checks': [], 'rows': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for count in (2, 6):
        audio, loss, grad = evaluate_case(values, renderer, criterion, 'seeded', count)
        scaled = (values * grad)[:, :count]
        expected = (values * ref_grad)[:, :count]
        per_item = torch.linalg.vector_norm(scaled - expected, dim=1) / torch.clamp(torch.linalg.vector_norm(expected, dim=1), min=1e-30)
        check = {'directions': count, 'waveform_relative_l2': relative_error(audio, ref_audio),
                 'loss_relative_l2': relative_error(loss, ref_loss),
                 'scaled_gradient_relative_l2': relative_error(scaled, expected),
                 'max_per_item_gradient_relative_l2': per_item.max().item(),
                 'max_absolute_scaled_gradient_error': (scaled - expected).abs().max().item()}
        torch.testing.assert_close(audio, ref_audio, rtol=1e-6, atol=1e-10)
        torch.testing.assert_close(loss, ref_loss, rtol=5e-4, atol=1e-10)
        torch.testing.assert_close(scaled, expected, rtol=5e-4, atol=1e-8)
        assert not torch.count_nonzero(grad[:, count:]), 'Unselected parameter gradients must be zero.'
        check['status'] = 'passed'
        result['checks'].append(check)
        del audio, loss, grad, scaled, expected, per_item
        save()
    del ref_audio, ref_loss, ref_grad
    torch.cuda.empty_cache()
    for method, count in [('seeded', 2), ('seeded', 6), ('replay', 6)]:
        def run():
            return evaluate_case(values, renderer, criterion, method, count)
        row = {'method': method, 'directions': count, **measure(run, values.device, args.repeats)}
        result['rows'].append(row)
        save()
        print(json.dumps(row), flush=True)
    result['metadata']['status'] = 'complete'
    save()
    print('ALL DIRECT SEEDED VALIDATION CHECKS PASSED', flush=True)

if __name__ == '__main__':
    main()
