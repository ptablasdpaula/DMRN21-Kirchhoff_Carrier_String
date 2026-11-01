"""Matched checkpoint reconstruction/replay latency, memory and gradient checks.

This research harness never changes the production solver. Every measurement is
paired with a checked replay reference at the same grid, precision, checkpoint
interval, duration, batch and target. JSON is incrementally written so failures
and completed observations survive worker interruption.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from benchmarks.exact_methods import BOX, GRID, filter_audio, measure, metadata, theta_of, varied_parameters
from src.optimization import criterion_for
from src.synthesis import Renderer


def reconstruction_digest():
    h = hashlib.sha256()
    for path in sorted((ROOT / 'experiments' / 'reconstruction').rglob('*')):
        if path.is_file() and path.suffix in ('.py', '.cu', '.cpp', '.h'):
            h.update(str(path.relative_to(ROOT)).encode() + path.read_bytes())
    return h.hexdigest()


def render(values, renderer, method, checkpoint):
    if method == 'replay':
        from src.adjoint import simulate
    elif method == 'reconstruction':
        from experiments.reconstruction.reconstruct import simulate
    else:
        raise ValueError(f'Unknown method {method}')
    raw = simulate(theta_of(values), GRID.intervals, GRID.integration_rate,
                   round(renderer.duration * GRID.integration_rate), 'multimode', checkpoint)
    return filter_audio(raw, renderer)


def checked_loss(values, renderer, criterion, method, checkpoint):
    p = values.detach().clone().requires_grad_()
    audio = render(p, renderer, method, checkpoint)
    losses = torch.stack([criterion(row) for row in audio])
    (grad,) = torch.autograd.grad(losses.sum(), p)
    return audio.detach(), losses.detach(), grad.detach()


def relative_error(a, b):
    denominator = torch.linalg.vector_norm(b).clamp_min(1e-30)
    return (torch.linalg.vector_norm(a - b) / denominator).item()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--validation', type=Path, required=True,
                   help='JSON produced by the reconstruction correctness suite')
    p.add_argument('--batches', default='1,8,64,128')
    p.add_argument('--durations', default='.25,4')
    p.add_argument('--checkpoints', default='64,256,1024')
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    batches = [int(v) for v in args.batches.split(',')]
    durations = [float(v) for v in args.durations.split(',')]
    checkpoints = [int(v) for v in args.checkpoints.split(',')]
    if args.repeats < 1 or any(v < 1 for v in batches + checkpoints) or any(not math.isfinite(v) or v <= 0 for v in durations):
        p.error('Choose positive batches, durations, checkpoints and repetitions')
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('This benchmark needs a CUDA allocation through hpc_sonic.')
    reference_meta = metadata()
    source_hash = reference_meta['source_sha256']
    experiment_hash = reconstruction_digest()
    test_hash = hashlib.sha256((ROOT / 'test' / 'reconstruction.py').read_bytes()).hexdigest()
    benchmark_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    harness_hash = hashlib.sha256((ROOT / 'benchmarks' / 'exact_methods.py').read_bytes()).hexdigest()
    worker_hash = hashlib.sha256((ROOT / 'hpc' / 'reconstruction-benchmark.sh').read_bytes()).hexdigest()
    evidence = json.loads(args.validation.read_text())
    if (evidence.get('status') != 'passed' or evidence.get('source_sha256') != source_hash
            or evidence.get('reconstruction_sha256') != experiment_hash
            or evidence.get('test_sha256') != test_hash
            or evidence.get('source_changed_during_run')):
        raise RuntimeError('Validation evidence is failed, stale or from a different source snapshot')
    result = {'metadata': {**reference_meta, 'reconstruction_sha256': experiment_hash,
                           'validation': str(args.validation), 'validation_evidence': evidence,
                           'benchmark_sha256': benchmark_hash, 'test_sha256': test_hash,
                           'harness_sha256': harness_hash, 'worker_sha256': worker_hash,
                           'status': 'running'}, 'checks': [], 'rows': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        from benchmarks.exact_methods import digest
        result['metadata']['source_changed_during_run'] = (digest() != source_hash or reconstruction_digest() != experiment_hash or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != benchmark_hash or hashlib.sha256((ROOT / 'test' / 'reconstruction.py').read_bytes()).hexdigest() != test_hash or hashlib.sha256((ROOT / 'benchmarks' / 'exact_methods.py').read_bytes()).hexdigest() != harness_hash or hashlib.sha256((ROOT / 'hpc' / 'reconstruction-benchmark.sh').read_bytes()).hexdigest() != worker_hash)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    def emit(row):
        result['rows'].append(row); save(); print(json.dumps(row, allow_nan=False), flush=True)
    save()
    device = torch.device('cuda', torch.cuda.current_device())
    for duration in durations:
        target = Renderer(GRID, duration).synthesize(BOX.defaults()).detach().to(device)
        renderer = Renderer(GRID, duration, device=device)
        criterion = criterion_for(target, GRID.sample_rate)
        for checkpoint in checkpoints:
            for batch in batches:
                values = varied_parameters(batch).to(device)
                ref_audio = ref_losses = ref_grad = output = losses = gradient = None
                check_passed = False
                try:
                    ref_audio, ref_losses, ref_grad = checked_loss(values, renderer, criterion, 'replay', checkpoint)
                    output, losses, gradient = checked_loss(values, renderer, criterion, 'reconstruction', checkpoint)
                    error_audio = relative_error(output, ref_audio)
                    error_loss = relative_error(losses, ref_losses)
                    error_grad = relative_error(values * gradient, values * ref_grad)
                    torch.testing.assert_close(output, ref_audio, rtol=1e-6, atol=1e-10)
                    torch.testing.assert_close(losses, ref_losses, rtol=5e-4, atol=1e-10)
                    torch.testing.assert_close(values * gradient, values * ref_grad, rtol=5e-4, atol=1e-8)
                    result['checks'].append(dict(batch=batch, duration=duration, checkpoint=checkpoint,
                        status='passed', waveform_relative_l2=error_audio, loss_relative_l2=error_loss,
                        scaled_gradient_relative_l2=error_grad))
                    check_passed = True
                    save()
                except (AssertionError, RuntimeError) as error:
                    result['checks'].append(dict(batch=batch, duration=duration, checkpoint=checkpoint,
                                                 status='failed', error=str(error)))
                    save(); print('CHECK FAILED', batch, duration, checkpoint, str(error), flush=True)
                finally:
                    del ref_audio, ref_losses, ref_grad, output, losses, gradient
                    if not check_passed: torch.cuda.empty_cache()
                for method in (('replay', 'reconstruction') if check_passed else ('replay',)):
                    pval = values.detach().clone().requires_grad_()
                    probe = torch.linspace(-1, 1, renderer.samples, dtype=torch.float64, device=device)
                    for kind in ('synthesis', 'vjp', 'loss_backward'):
                        if kind == 'synthesis':
                            def fn():
                                with torch.no_grad(): return render(values, renderer, method, checkpoint)
                        elif kind == 'vjp':
                            def fn():
                                pval.grad = None
                                (render(pval, renderer, method, checkpoint) * probe).sum().backward()
                        else:
                            def fn():
                                pval.grad = None
                                audio = render(pval, renderer, method, checkpoint)
                                torch.stack([criterion(row) for row in audio]).sum().backward()
                        try:
                            timing = measure(fn, device, args.repeats)
                            emit(dict(method=method, kind=kind, batch=batch, duration=duration,
                                      checkpoint=checkpoint, items_per_second=batch / timing['median_seconds'], **timing))
                        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                            if isinstance(error, torch.cuda.OutOfMemoryError): torch.cuda.empty_cache()
                            emit(dict(method=method, kind=kind, batch=batch, duration=duration,
                                      checkpoint=checkpoint, error=str(error)))
    failed_checks = any(row.get('status') != 'passed' for row in result['checks'])
    failed_rows = any('error' in row for row in result['rows'])
    result['metadata']['status'] = 'complete_with_failures' if failed_checks or failed_rows else 'complete'
    save()
    if result['metadata']['source_changed_during_run']:
        raise RuntimeError('Source changed while benchmarking; observations cannot be treated as matched.')
    print('RECONSTRUCTION BENCHMARK COMPLETE', flush=True)

if __name__ == '__main__':
    main()
