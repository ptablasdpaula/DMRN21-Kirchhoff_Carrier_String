"""GPU smoke test for the sound-matching application: batched render vs the CPU reference, gradients, a short fit."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from experiments.applications.common.boxes import decode_batch, sample_logits
from experiments.applications.common.optim import Schedule, fit_batch
from experiments.applications.common.render import BatchRenderer, theta_from_physical
from experiments.applications.sound_matching.boxes import family_box, stage_a_grid, EXCITATION
from experiments.applications.sound_matching.loss import make_criterion, per_item_loss
from src.kc_string_forward import ForwardString


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'CUDA is required'
    torch.set_num_threads(1)
    box, grid = family_box('steel'), stage_a_grid('steel')
    logits = sample_logits(box, 4, seed=1)
    physical = decode_batch(box, logits)
    physical[0, 8], physical[0, 9] = 0.3, 0.6     # off-centre pluck and pickup: both position gradients nonzero
    seconds = 0.05
    renderer = BatchRenderer(grid, seconds, EXCITATION, device='cuda')
    x = physical.cuda().requires_grad_()
    raw = renderer.raw(x)
    audio = renderer(x)
    assert raw.shape == (4, renderer.steps+2) and audio.shape == (4, renderer.samples), (raw.shape, audio.shape)
    reference = ForwardString(grid.intervals, grid.integration_rate, excitation=EXCITATION, order=grid.order)
    cpu_audio, _ = reference.output_and_tangents(theta_from_physical(physical[0]), seconds)
    torch.testing.assert_close(raw[0].detach().cpu(), cpu_audio, rtol=1e-6, atol=1e-12)
    waveform_error = float((raw[0].detach().cpu()-cpu_audio).norm()/cpu_audio.norm())
    (audio.square().mean()).backward()
    grad = x.grad.detach().cpu()
    assert bool(torch.isfinite(grad).all()) and bool((grad[0] != 0).all()), grad
    # A three-iteration fit of a self-generated target on 0.3 s of audio.
    renderer = BatchRenderer(grid, 0.3, EXCITATION, device='cuda')
    criterion = make_criterion(grid.sample_rate)
    with torch.no_grad():
        target = renderer(physical[:1].cuda())[0]
    def objective(z, indices):
        return per_item_loss(criterion, renderer(decode_batch(box, z)), target), None
    result = fit_batch(logits.cuda(), objective, schedule=Schedule(maximum_updates=3, stop_patience=5))
    assert result.evaluations == 4 and bool(torch.isfinite(result.best_loss).all())
    assert bool((result.best_loss <= result.initial_loss).all())
    report = {'status': 'passed', 'grid': renderer.describe(), 'waveform_relative_l2_vs_cpu': waveform_error,
              'gradient_abs_min': float(grad.abs().min()), 'initial_loss': result.initial_loss.tolist(),
              'best_loss': result.best_loss.tolist(), 'seconds_three_iterations': result.seconds}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
