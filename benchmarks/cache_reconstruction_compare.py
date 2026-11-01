"""Compare experimental scalar-cached reconstruction with exact references."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from benchmarks.exact_methods import BOX, GRID, SeededTangents, filter_audio, measure, metadata, theta_of, varied_parameters
from benchmarks.reconstruction import relative_error
from src.optimization import criterion_for
from src.synthesis import Renderer


def render(values, model, method, checkpoint):
    if method == 'seeded':
        audio = SeededTangents.apply(values, round(model.duration * GRID.integration_rate), 0)
        return filter_audio(audio, model)
    elif method == 'replay':
        from src.adjoint import simulate
    elif method == 'reconstruction':
        from experiments.reconstruction.reconstruct import simulate
    elif method == 'cached':
        from experiments.cached_reconstruction.reconstruct import simulate
    else:
        raise ValueError(method)
    audio = simulate(theta_of(values), GRID.intervals, GRID.integration_rate,
                     round(model.duration * GRID.integration_rate), 'multimode', checkpoint)
    return filter_audio(audio, model)


def loss_and_grad(values, model, criterion, method, checkpoint):
    x = values.detach().clone().requires_grad_()
    audio = render(x, model, method, checkpoint)
    losses = torch.stack([criterion(row) for row in audio])
    grad, = torch.autograd.grad(losses.sum(), x)
    return audio.detach(), losses.detach(), grad.detach()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--durations', default='.25,4')
    p.add_argument('--batches', default='1,128')
    p.add_argument('--checkpoint', type=int, default=256)
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('Requires a GPU allocated by hpc_sonic')
    device = torch.device('cuda')
    impl = ROOT / 'experiments/cached_reconstruction'
    source_hash = metadata()['source_sha256']
    h = hashlib.sha256()
    for path in sorted(impl.glob('*')):
        if path.suffix in ('.cu', '.py', '.cpp', '.h'):
            h.update(path.name.encode() + path.read_bytes())
    result = {'metadata': {**metadata(), 'cached_sha256': h.hexdigest(),
                           'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           'status': 'running'}, 'checks': [], 'rows': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')

    save()
    for duration in map(float, args.durations.split(',')):
        target = Renderer(GRID, duration).synthesize(BOX.defaults()).detach().to(device)
        model = Renderer(GRID, duration, device=device)
        criterion = criterion_for(target, GRID.sample_rate)
        for batch in map(int, args.batches.split(',')):
            values = varied_parameters(batch).to(device)
            ref = candidate = None
            try:
                ref = loss_and_grad(values, model, criterion, 'replay', args.checkpoint)
                for method in ('cached', 'seeded'):
                    candidate = loss_and_grad(values, model, criterion, method, args.checkpoint)
                    errors = dict(audio_relative_l2=relative_error(candidate[0], ref[0]),
                                  loss_relative_l2=relative_error(candidate[1], ref[1]),
                                  scaled_gradient_relative_l2=relative_error(values*candidate[2], values*ref[2]))
                    torch.testing.assert_close(candidate[0], ref[0], rtol=1e-6, atol=1e-10)
                    torch.testing.assert_close(candidate[1], ref[1], rtol=5e-4, atol=1e-10)
                    torch.testing.assert_close(values*candidate[2], values*ref[2], rtol=5e-4, atol=1e-8)
                    result['checks'].append(dict(method=method, batch=batch, duration=duration, status='passed', **errors))
            except (RuntimeError, AssertionError) as exc:
                result['checks'].append(dict(batch=batch, duration=duration, status='failed', error=str(exc)))
                save()
                print('FAILED', batch, duration, str(exc), flush=True)
                continue
            finally:
                del ref, candidate
            save()
            for method in ('seeded', 'replay', 'reconstruction', 'cached'):
                def fn():
                    x = values.detach().clone().requires_grad_()
                    audio = render(x, model, method, args.checkpoint)
                    torch.stack([criterion(row) for row in audio]).sum().backward()
                try:
                    row = {'method': method, 'batch': batch, 'duration': duration,
                           'checkpoint': args.checkpoint, **measure(fn, device, args.repeats)}
                except RuntimeError as exc:
                    row = {'method': method, 'batch': batch, 'duration': duration,
                           'checkpoint': args.checkpoint, 'error': str(exc)}
                result['rows'].append(row); save(); print(json.dumps(row), flush=True)
    result['metadata']['source_changed_during_run'] = metadata()['source_sha256'] != source_hash
    result['metadata']['status'] = ('complete_with_failures' if any(x['status'] != 'passed' for x in result['checks'])
                                    or any('error' in x for x in result['rows']) else 'complete')
    save()
    if result['metadata']['status'] != 'complete' or result['metadata']['source_changed_during_run']:
        raise RuntimeError('Comparison failed or source changed; inspect result JSON')


if __name__ == '__main__':
    main()
