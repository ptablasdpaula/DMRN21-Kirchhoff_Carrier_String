"""CPU benchmark of the KC recurrence at a fixed spatial/time grid.

Run with Pixi. No production simulator files are modified. The differentiable
harness preserves the reference initialization and stencil/rank-one helpers.
Peak RSS is process memory, including Python/PyTorch and allocator caches, not
just tensor storage. Every timed measurement uses a fresh, warmed subprocess.

Examples:
    pixi run python benchmarks/autodiff_memory.py --validate
    pixi run python benchmarks/autodiff_memory.py --durations 0.06 0.25
    pixi run python benchmarks/autodiff_memory.py --durations 4 --repeats 1 --memory-limit-gib 6
    pixi run python benchmarks/autodiff_memory.py --compile-step --methods no_grad reverse checkpoint_reentrant --durations 1 4 --memory-limit-gib 6 --output .notebook-build/compile_memory.json

The default benchmark grid is 128 intervals at exactly 44,100 Hz, NOT the
production simulator's default 512x oversampling. All methods use the same
grid, initial state and midpoint mean-square loss. Eight independent relative
parameters are differentiated: density, tension, radius, Young's modulus,
length, excitation amplitude, sigma_0 and sigma_1 (not their T60 parameterization).
No optimizer update, compilation, audio export or process-startup time is timed.
With --compile-step, first-use compilation/warmup time is recorded separately.
Whole-process peak RSS includes compiler memory; additional peak subtracts the
post-compilation warmup peak. Inductor disk caches are allowed to be reused.
Reentrant checkpointing supports the loss.backward() use case measured here;
its restrictions matter if using torch.autograd.grad or higher-order derivatives.
"""
import argparse
import contextlib
import functools
import gc
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def load_torch():
    global torch, checkpoint, kc
    import torch
    from torch.utils.checkpoint import checkpoint
    sys.path.insert(0, str(ROOT / 'src'))
    import kc_string as kc
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def defaults():
    return torch.tensor([kc.material_density, kc.tension, kc.radius,
                         kc.young_modulus, kc.length, kc.amplitude,
                         kc.sigma_0, kc.sigma_1], dtype=torch.float64)


def prepare(parameters, intervals, sample_rate):
    density, tension, radius, modulus, length, amplitude, sigma_0, sigma_1 = parameters
    dt = 1 / sample_rate
    h = length / intervals
    area = math.pi * radius**2
    mass = density * area
    inertia = math.pi * radius**4 / 4
    c_squared = tension / mass
    K_squared = modulus * inertia / mass
    indices = torch.arange(1, intervals, dtype=torch.float64)
    previous = amplitude * torch.sin(math.pi * indices / intervals)
    second = kc.second_spatial_derivative(previous, h)
    fourth = kc.second_spatial_derivative(second, h)
    nonlinear = -torch.dot(previous, second) * second
    # Deliberately retain the reference's extra startup h factors.
    current = previous + 0.5 * dt**2 * (
        h * c_squared * second - h * K_squared * fourth
        + h * (0.5 * modulus * h / (length * density)) * nonlinear
    )
    inverse_mass = 1 / (mass + mass * sigma_0 * dt)
    q_scale = dt * torch.sqrt(h * modulus * area / length) * 0.5
    # Constant step coefficients are computed once, for every benchmark method.
    coefficients = (h, mass, tension * dt**2 + 2 * mass * sigma_1 * dt,
                    modulus * inertia * dt**2, mass * sigma_0 * dt - mass,
                    2 * mass * sigma_1 * dt, inverse_mass, q_scale)
    return previous, current, coefficients


def advance(previous, current, *coefficients, steps):
    h, mass, tension_loss, bending, previous_mass, previous_loss, inverse_mass, q_scale = coefficients
    midpoint = current.numel() // 2
    samples = []
    for _ in range(steps):
        second = kc.second_spatial_derivative(current, h)
        fourth = kc.second_spatial_derivative(second, h)
        previous_second = kc.second_spatial_derivative(previous, h)
        linear = (2 * mass * current + tension_loss * second - bending * fourth
                  + previous_mass * previous - previous_loss * previous_second)
        q = q_scale * second
        rhs = linear - q * torch.dot(q, previous)
        following = kc.solve_rank_one(rhs, q, inverse_mass)
        samples.append(following[midpoint])
        previous, current = current, following
    return previous, current, torch.stack(samples)


def simulate(parameters, steps, intervals, sample_rate, method, chunk):
    previous, current, coefficients = prepare(parameters, intervals, sample_rate)
    midpoint = current.numel() // 2
    blocks = [torch.stack((previous[midpoint], current[midpoint]))]
    for start in range(0, steps, chunk):
        operation = functools.partial(advance, steps=min(chunk, steps - start))
        if method in ('checkpoint', 'checkpoint_reentrant'):
            previous, current, samples = checkpoint(
                operation, previous, current, *coefficients,
                use_reentrant=method == 'checkpoint_reentrant', preserve_rng_state=False,
            )
        else:
            previous, current, samples = operation(previous, current, *coefficients)
        blocks.append(samples)
    return torch.cat(blocks)


def objective(output):
    # Fixed normalizer: amplitude is itself optimised, so do not normalize by it.
    return (output / kc.amplitude).square().mean()


def peak_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024**2 if sys.platform == 'darwin' else 1024)


def current_reference(steps, intervals, sample_rate):
    original_grid = kc.simulation_grid
    kc.simulation_grid = lambda _=1: (intervals, kc.length / intervals, 1 / sample_rate, sample_rate)
    try:
        # A tiny positive offset prevents integer-division roundoff dropping a step.
        return kc.simulate_string((steps + 1e-7) / sample_rate, 1)[0]
    finally:
        kc.simulation_grid = original_grid


def validate():
    load_torch()
    # Validate at both the optimisation grid and the original reference grid,
    # while keeping each run short. No Octave invocation is needed here.
    for intervals, rate in ((128, 44100), (kc.simulation_grid()[0], kc.simulation_grid()[3])):
        parameters = defaults()
        expected = current_reference(64, intervals, rate)
        with torch.no_grad():
            actual = simulate(parameters, 64, intervals, rate, 'no_grad', 32)
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-12)
    outputs = {}
    gradients = {}
    for method in ('reverse', 'checkpoint', 'checkpoint_reentrant'):
        relative = torch.ones(8, dtype=torch.float64, requires_grad=True)
        output = simulate(relative * defaults(), 256, 128, 44100, method, 32)
        objective(output).backward()
        outputs[method] = output.detach()
        gradients[method] = relative.grad.detach().clone()
    for method in ('checkpoint', 'checkpoint_reentrant'):
        torch.testing.assert_close(outputs[method], outputs['reverse'], rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(gradients[method], gradients['reverse'], rtol=1e-9, atol=1e-11)
    direction = torch.tensor([0.2, -0.3, 0.1, 0.25, -0.15, 0.3, 0.4, -0.2], dtype=torch.float64)
    analytic = torch.dot(gradients['reverse'], direction).item()
    differences = []
    for epsilon in (1e-3, 3e-4, 1e-4):
        with torch.no_grad():
            plus = objective(simulate((1 + epsilon * direction) * defaults(), 256, 128, 44100, 'no_grad', 32))
            minus = objective(simulate((1 - epsilon * direction) * defaults(), 256, 128, 44100, 'no_grad', 32))
        finite_difference = ((plus - minus) / (2 * epsilon)).item()
        differences.append({'epsilon': epsilon, 'absolute_error': abs(analytic - finite_difference)})
    assert differences[-1]['absolute_error'] < 1e-6, differences
    print(json.dumps({'status': 'passed', 'directional_derivative': analytic,
                      'central_difference_errors': differences, 'relative_parameter_gradients': gradients['reverse'].tolist()}))


def compile_step_and_validate(args):
    """Compile one recurrence step; retain Python time/checkpoint loops."""
    global advance
    requires_grad = args.method not in ('current', 'no_grad')
    def trial():
        relative = torch.ones(8, dtype=torch.float64, requires_grad=requires_grad)
        with contextlib.nullcontext() if requires_grad else torch.no_grad():
            output = simulate(relative * defaults(), 64, args.intervals, args.rate, args.method, 32)
            loss = objective(output)
        if requires_grad:
            loss.backward()
        return output.detach(), relative.grad
    expected, expected_gradient = trial()
    compiled_step = torch.compile(functools.partial(advance, steps=1),
                                  backend='inductor', fullgraph=True, dynamic=False)
    def compiled_advance(previous, current, *coefficients, steps):
        samples = []
        for _ in range(steps):
            previous, current, sample = compiled_step(previous, current, *coefficients)
            samples.append(sample)
        return previous, current, torch.cat(samples)
    advance = compiled_advance
    started = time.perf_counter()
    actual, actual_gradient = trial()
    first_use_seconds = time.perf_counter() - started
    torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-12)
    gradient_error = None
    if requires_grad:
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-9)
        gradient_error = (actual_gradient - expected_gradient).abs().max().item()
    return {'first_use_seconds': first_use_seconds,
            'short_output_max_error': (actual - expected).abs().max().item(),
            'short_gradient_max_error': gradient_error,
            'compiler': 'inductor', 'compiled_steps_per_call': 1,
            'compile_note': 'First use includes a 64-step validation run; disk caches may be reused. Peak RSS includes compilation.'}


def worker(args):
    load_torch()
    # Warm lazy checkpoint imports and a small backward in every process, so
    # checkpoint's first-use import/initialization is not charged to one method.
    warm = torch.ones(8, dtype=torch.float64, requires_grad=True)
    objective(simulate(warm * defaults(), 32, args.intervals, args.rate, 'checkpoint', 16)).backward()
    del warm
    compilation = compile_step_and_validate(args) if args.compile_step else {}
    gc.collect()
    parameters = torch.ones(8, dtype=torch.float64, requires_grad=args.method not in ('no_grad', 'current'))
    baseline = peak_mib()
    steps = round(args.duration * args.rate)
    started = time.perf_counter()
    context = torch.no_grad() if args.method in ('no_grad', 'current') else contextlib.nullcontext()
    with context:
        if args.method == 'current':
            output = current_reference(steps, args.intervals, args.rate)
        else:
            output = simulate(parameters * defaults(), steps, args.intervals, args.rate, args.method, args.chunk)
        loss = objective(output)
    forward_seconds = time.perf_counter() - started
    forward_peak = peak_mib()
    backward_seconds = None
    if args.method not in ('no_grad', 'current'):
        started = time.perf_counter()
        loss.backward()
        backward_seconds = time.perf_counter() - started
    final_peak = peak_mib()
    result = {
        'method': args.method, 'compiled': args.compile_step, **compilation,
        'duration_seconds': args.duration, 'steps': steps,
        'intervals': args.intervals, 'sample_rate': args.rate, 'chunk_steps': args.chunk,
        'forward_seconds': forward_seconds, 'backward_seconds': backward_seconds,
        'total_seconds': forward_seconds + (backward_seconds or 0),
        'baseline_peak_mib': baseline, 'forward_peak_mib': forward_peak,
        'total_peak_mib': final_peak, 'additional_peak_mib': max(0, final_peak - baseline),
        'loss': loss.item(), 'output_samples': output.numel(),
        'gradients': parameters.grad.tolist() if parameters.grad is not None else None,
        'torch_version': torch.__version__, 'dtype': 'float64', 'device': 'cpu', 'threads': 1,
    }
    assert math.isfinite(result['loss'])
    if result['gradients'] is not None:
        assert all(math.isfinite(value) for value in result['gradients'])
    print(json.dumps(result), flush=True)


def suite(args):
    import psutil
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(output_path.read_text()) if output_path.exists() else []
    for duration in args.durations:
        for chunk in args.chunks:
            for method in args.methods:
                if method in ('current', 'no_grad', 'reverse') and chunk != args.chunks[0]:
                    continue
                for repetition in range(args.repeats):
                    key = (duration, chunk, method, repetition, args.intervals, args.rate, args.compile_step)
                    if any((r['duration_seconds'], r['chunk_steps'], r['method'], r['repetition'],
                            r.get('intervals'), r.get('sample_rate'), r.get('compiled', False)) == key for r in records):
                        continue
                    if psutil.virtual_memory().available < 4 * 2**30:
                        raise RuntimeError('Less than 4 GiB system RAM available; stop before launching another trial.')
                    command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--method', method,
                               '--duration', str(duration), '--chunk', str(chunk),
                               '--intervals', str(args.intervals), '--rate', str(args.rate)]
                    if args.compile_step:
                        command.append('--compile-step')
                    print(f'RUN duration={duration}s method={method} chunk={chunk} repeat={repetition+1}', flush=True)
                    started = time.monotonic()
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    monitored = psutil.Process(process.pid)
                    aborted = None
                    observed_peak_mib = 0.0
                    while process.poll() is None:
                        try:
                            observed_peak_mib = max(observed_peak_mib, monitored.memory_info().rss / 2**20)
                            if observed_peak_mib > args.memory_limit_gib * 1024:
                                aborted = 'memory limit exceeded'
                            if time.monotonic() - started > args.timeout:
                                aborted = 'time limit exceeded'
                            if psutil.virtual_memory().available < 2 * 2**30:
                                aborted = 'system available RAM below 2 GiB'
                        except psutil.NoSuchProcess:
                            break
                        if aborted:
                            process.terminate()
                            break
                        time.sleep(0.05)
                    stdout, stderr = process.communicate(timeout=30)
                    if aborted:
                        record = {'duration_seconds': duration, 'chunk_steps': chunk, 'method': method,
                                  'repetition': repetition, 'status': aborted, 'compiled': args.compile_step,
                                  'intervals': args.intervals, 'sample_rate': args.rate,
                                  'observed_peak_mib': observed_peak_mib,
                                  'wall_seconds_until_stop': time.monotonic() - started,
                                  'memory_limit_gib': args.memory_limit_gib}
                    elif process.returncode:
                        raise RuntimeError(stderr + '\n' + stdout)
                    else:
                        record = json.loads(stdout.strip().splitlines()[-1])
                        record.update(repetition=repetition, status='completed')
                    records.append(record)
                    output_path.write_text(json.dumps(records, indent=2))
                    print(json.dumps(record), flush=True)
    print(f'Results saved to {output_path}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile-step', action='store_true', help='Compile the recurrence step with Inductor; use no_grad instead of current.')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--method', default='reverse')
    parser.add_argument('--duration', type=float, default=0.06)
    parser.add_argument('--chunk', type=int, default=128)
    parser.add_argument('--intervals', type=int, default=128)
    parser.add_argument('--rate', type=int, default=44100)
    parser.add_argument('--durations', nargs='+', type=float, default=[0.06, 1.0])
    parser.add_argument('--chunks', nargs='+', type=int, default=[128])
    parser.add_argument('--methods', nargs='+', default=['current', 'no_grad', 'reverse', 'checkpoint', 'checkpoint_reentrant'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--memory-limit-gib', type=float, default=3.0)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--output', default='.notebook-build/autodiff_memory.json')
    arguments = parser.parse_args()
    if arguments.compile_step and ('current' in arguments.methods if not arguments.worker else arguments.method == 'current'):
        parser.error('Use no_grad instead of current when compiling.')
    if arguments.validate:
        validate()
    elif arguments.worker:
        worker(arguments)
    else:
        suite(arguments)
