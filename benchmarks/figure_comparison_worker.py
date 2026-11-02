"""One isolated timing/validation case for figure_comparison_driver."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('JAX_ENABLE_X64', 'true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')


def emit(event, **values):
    print(json.dumps({'event': event, **values}, allow_nan=False), flush=True)


def torch_case(method, batch, duration, excitation='multimode', loss='mae'):
    import torch
    from benchmarks.exact_methods import BOX, GRID, filter_audio
    from benchmarks.reconstruction import relative_error
    from experiments.figure_comparison.physical import physical_values, theta_from_physical, gradient_scales
    from experiments.loss_batch.grouped import grouped_loss_backward
    from src.adjoint import simulate as replay
    from src.optimization import criterion_for
    from src.synthesis import Renderer
    if method in ('fdtd', 'fdtd_ckpt'):
        from experiments.figure_comparison.fdtd import simulate
    else:
        from experiments.scalar_cache.reconstruct import simulate
    torch.set_num_threads(1)
    assert torch.cuda.is_available(), 'CUDA is required'
    device = torch.device('cuda')
    target = Renderer(GRID, duration, excitation).synthesize(BOX.defaults()).detach().to(device)
    renderer = Renderer(GRID, duration, excitation, device=device)
    criterion = criterion_for(target, GRID.sample_rate, loss)
    values = physical_values(batch).to(device)
    scales = gradient_scales(device).cpu()
    steps = round(duration * GRID.integration_rate)

    def run(which):
        x = values.detach().clone().requires_grad_()
        args = (theta_from_physical(x), GRID.intervals, GRID.integration_rate, steps, excitation)
        if which == 'replay': raw = replay(*args, 256)
        elif which == 'fdtd': raw = simulate(*args)
        elif which == 'fdtd_ckpt': raw = simulate(*args, checkpoint_steps=256)
        else: raw = simulate(*args, 256, 3 if which == 'cached3' else 1)
        audio = filter_audio(raw, renderer)
        loss = grouped_loss_backward(audio, criterion, 16)
        return audio.detach(), loss.detach(), x.grad.detach()

    start = time.perf_counter(); ref = run('replay'); torch.cuda.synchronize()
    reference_seconds = time.perf_counter() - start
    ref_cpu = tuple(t.cpu() for t in ref)
    del ref
    torch.cuda.empty_cache(); torch.cuda.synchronize()
    start = time.perf_counter(); candidate = run(method); torch.cuda.synchronize()
    warmup = time.perf_counter() - start
    actual = tuple(t.cpu() for t in candidate)
    errors = {
        'audio_relative_l2': relative_error(actual[0], ref_cpu[0]),
        'loss_relative_l2': relative_error(actual[1], ref_cpu[1]),
        'scaled_gradient_relative_l2': relative_error(scales*actual[2], scales*ref_cpu[2]),
    }
    torch.testing.assert_close(actual[0], ref_cpu[0], rtol=1e-6, atol=1e-10)
    torch.testing.assert_close(actual[1], ref_cpu[1], rtol=5e-4, atol=1e-10)
    torch.testing.assert_close(scales*actual[2], scales*ref_cpu[2], rtol=5e-4, atol=1e-8)
    del candidate, ref_cpu, actual
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    emit('measure_begin')
    samples = []
    for _ in range(3):
        torch.cuda.synchronize(); start = time.perf_counter()
        result = run(method)
        torch.cuda.synchronize(); samples.append(time.perf_counter()-start)
        if not all(bool(torch.isfinite(x).all()) for x in result):
            raise RuntimeError('Non-finite measured output')
        del result
    emit('measure_end')
    return {'method': method, 'batch': batch, 'duration_seconds': duration, 'status': 'ok',
            'median_seconds': statistics.median(samples), 'repetitions_seconds': samples,
            'warmup_seconds': warmup, 'reference_seconds': reference_seconds,
            'peak_framework_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_framework_reserved_bytes': torch.cuda.max_memory_reserved(),
            'validation': errors, 'gpu_name': torch.cuda.get_device_name(),
            'gpu_uuid': str(getattr(torch.cuda.get_device_properties(0), 'uuid', 'unavailable')),
            'torch_version': torch.__version__, 'cuda_runtime': torch.version.cuda,
            'checkpoint_steps': None if method == 'fdtd' else 256,
            'rematerialisation': ('torch.utils.checkpoint (use_reentrant=False) over 256-step chunks of the spatial PyTorch loop'
                                  if method == 'fdtd_ckpt' else None), 'excitation': excitation,
            'gradient_coordinates': ['density', 'tension', 'radius', 'young_modulus', 'length', 'amplitude', 'sigma_0', 'sigma_1', 'pluck_position', 'pickup_position'], 'loss': loss, 'loss_group': 16}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=('fdtd', 'fdtd_ckpt', 'cached1', 'cached3', 'diaz'), required=True)
    ap.add_argument('--batch', type=int, required=True)
    ap.add_argument('--duration', type=float, default=4.)
    ap.add_argument('--excitation', default='multimode')
    ap.add_argument('--loss', default='mae', choices=('mae', 'cel'))
    args = ap.parse_args()
    try:
        if args.method == 'diaz':
            from experiments.figure_comparison.diaz_worker import run_case
            row = run_case(args.batch, args.duration, emit, args.excitation, args.loss)
        else:
            row = torch_case(args.method, args.batch, args.duration, args.excitation, args.loss)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        s = str(exc).lower()
        status = 'oom' if 'out of memory' in s or 'resource_exhausted' in s else 'error'
        row = {'method': args.method, 'batch': args.batch, 'status': status, 'loss': args.loss,
               'error': f'{type(exc).__name__}: {exc}'}
    emit('result', row=row)


if __name__ == '__main__':
    main()
