"""Eight simultaneous forward-mode tangents, using vmap(jvp) and Inductor.

CPU float64, 128 intervals, 44,100 Hz. The loss is the same mean-square
midpoint displacement as autodiff_memory.py. Two displacement states carry
8 parameter derivatives each. Loss and its derivatives accumulate online;
the audio output is retained, but its Jacobian is not. No backward graph.
Run: pixi run python benchmarks/forward_memory.py
Four-step compilation: pixi run python benchmarks/forward_memory.py --block-steps 4 --durations 4

Each displacement state has shape (127, 9): column zero is displacement,
the other columns are derivatives with respect to the eight relative physical
parameters in autodiff_memory.defaults(). For an update y = F(x, coefficients),
jvp propagates dy/dtheta = F_x dx/dtheta + F_coeff dcoeff/dtheta.
Initial conditions and coefficients are differentiated too. This is forward
automatic differentiation, not finite differences or eight separate simulations.
Online loss accumulation keeps derivative memory independent of duration for
this loss. Retaining audio costs O(time); retaining its full Jacobian for a
different, non-streamable loss would cost O(time * parameter_count).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import autodiff_memory as b


def initialize(relative):
    previous, current, coefficients = b.prepare(relative * b.defaults(), 128, 44100)
    midpoint = current.numel() // 2
    loss_sum = (previous[midpoint] / b.kc.amplitude)**2 + (current[midpoint] / b.kc.amplitude)**2
    return (previous, current, loss_sum, *coefficients)


def step(previous, current, loss_sum, *coefficients):
    previous, following, sample = b.advance(previous, current, *coefficients, steps=1)
    return previous, following, loss_sum + (sample[0] / b.kc.amplitude)**2


def propagate(primals, tangents):
    return b.torch.vmap(lambda *directions: b.torch.func.jvp(step, primals, directions),
                        out_dims=(None, 0))(*tangents)


def pack(primal, tangent):
    return b.torch.cat((primal.unsqueeze(-1), tangent.movedim(0, -1)), dim=-1)


def packed_propagate(states, coefficients, accumulated_loss):
    primals = (states[0, :, 0], states[1, :, 0], accumulated_loss[0], *coefficients[:, 0].unbind())
    tangents = (states[0, :, 1:].T, states[1, :, 1:].T, accumulated_loss[1:], *coefficients[:, 1:].unbind())
    # Avoid a PyTorch forward-AD view-layout assertion for strided packed views.
    primals = tuple(value.contiguous() for value in primals)
    tangents = tuple(value.contiguous() for value in tangents)
    values, derivatives = propagate(primals, tangents)
    following_states = b.torch.stack((pack(values[0], derivatives[0]), pack(values[1], derivatives[1])))
    return following_states, pack(values[2], derivatives[2]), values[1][63]


def packed_block(states, coefficients, accumulated_loss, steps):
    samples = []
    for _ in range(steps):
        states, accumulated_loss, sample = packed_propagate(states, coefficients, accumulated_loss)
        samples.append(sample)
    return states, accumulated_loss, b.torch.stack(samples)


def simulate(operation, steps, block_steps=1):
    torch = b.torch
    relative = torch.ones(8, dtype=torch.float64)
    identity = torch.eye(8, dtype=torch.float64)
    primals, tangents = torch.vmap(lambda direction: torch.func.jvp(initialize, (relative,), (direction,)),
                                  out_dims=(None, 0))(identity)
    states = torch.stack((pack(primals[0], tangents[0]), pack(primals[1], tangents[1])))
    coefficients = torch.stack([pack(p, t) for p, t in zip(primals[3:], tangents[3:])])
    accumulated_loss = pack(primals[2], tangents[2])
    output = torch.empty(steps + 2, dtype=torch.float64)
    output[0], output[1] = primals[0][63], primals[1][63]
    for index in range(0, steps, block_steps):
        if block_steps == 1:
            states, accumulated_loss, sample = operation(states, coefficients, accumulated_loss)
            output[index + 2] = sample
        else:
            count = min(block_steps, steps - index)
            states, accumulated_loss, samples = operation(states, coefficients, accumulated_loss, count)
            output[index + 2:index + 2 + count] = samples
    return output, accumulated_loss[0] / (steps + 2), accumulated_loss[1:] / (steps + 2)


def worker(duration, block_steps, implementation):
    b.load_torch()
    torch = b.torch
    # Same lazy-import warmup as the reverse-mode benchmark.
    warm = torch.ones(8, dtype=torch.float64, requires_grad=True)
    b.objective(b.simulate(warm * b.defaults(), 32, 128, 44100, 'checkpoint', 16)).backward()
    relative = torch.ones(8, dtype=torch.float64, requires_grad=True)
    reference = b.simulate(relative * b.defaults(), 256, 128, 44100, 'reverse', 128)
    b.objective(reference).backward()
    expected_output = reference.detach()
    expected_gradient = relative.grad.detach()
    del reference, warm, relative
    if implementation in ('dual', 'native'):
        from kc_string_forward import DualReference, advance_block, native_block
        evaluator = DualReference(block_steps=block_steps)
        if implementation == 'native':
            evaluator.operation = native_block
        def run(steps):
            return evaluator.loss_and_gradient(torch.ones(8, dtype=torch.float64), steps / 44100)
    else:
        compiled = torch.compile(packed_propagate if block_steps == 1 else packed_block,
                                 backend='inductor', fullgraph=True, dynamic=False)
        def run(steps):
            return simulate(compiled, steps, block_steps)
    with torch.no_grad():
        started = time.perf_counter()
        output, loss, gradients = run(256)
        remainder = round(duration * 44100) % block_steps
        if remainder:
            run(256 + remainder)
        first_use = time.perf_counter() - started
        torch.testing.assert_close(output, expected_output, rtol=1e-8, atol=1e-12)
        torch.testing.assert_close(gradients, expected_gradient, rtol=1e-6, atol=1e-9)
        validation_error = (gradients - expected_gradient).abs().max().item()
        baseline = b.peak_mib()
        started = time.perf_counter()
        output, loss, gradients = run(round(duration * 44100))
        elapsed = time.perf_counter() - started
    result = dict(duration_seconds=duration, method='compiled_forward_8_tangents_packed',
                  implementation=implementation,
                  compiled_steps_per_call=block_steps,
                  forward_and_gradient_seconds=elapsed, backward_seconds=None,
                  first_use_seconds=first_use, total_peak_mib=b.peak_mib(),
                  additional_peak_mib=b.peak_mib()-baseline, baseline_peak_mib=baseline,
                  loss=loss.item(), gradients=gradients.tolist(),
                  validation_max_gradient_error=validation_error,
                  output_samples=output.numel(), threads=1, dtype='float64', device='cpu')
    assert torch.isfinite(gradients).all() and torch.isfinite(loss)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--block-steps', type=int, choices=[1, 4, 32, 128], default=1)
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--implementation', choices=['jvp', 'dual', 'native'], default='jvp')
    parser.add_argument('--duration', type=float, default=0.06)
    parser.add_argument('--durations', type=float, nargs='+', default=[0.06, 1, 4])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', default='.notebook-build/forward_memory.json')
    args = parser.parse_args()
    if args.worker:
        worker(args.duration, args.block_steps, args.implementation)
    else:
        import psutil
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        records = json.loads(path.read_text()) if path.exists() else []
        for duration in args.durations:
            for repetition in range(args.repeats):
                if any(r['duration_seconds']==duration and r['repetition']==repetition and r.get('compiled_steps_per_call', 1)==args.block_steps and r.get('implementation', 'jvp')==args.implementation for r in records):
                    continue
                print('RUN',duration,repetition,flush=True)
                process = subprocess.Popen([sys.executable, __file__, '--worker', '--duration', str(duration), '--block-steps', str(args.block_steps), '--implementation', args.implementation],
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                monitor = psutil.Process(process.pid)
                started = time.monotonic()
                while process.poll() is None:
                    try:
                        if monitor.memory_info().rss > 6 * 2**30 or psutil.virtual_memory().available < 2 * 2**30 or time.monotonic()-started > args.timeout:
                            process.terminate()
                            process.communicate(timeout=30)
                            raise RuntimeError('Memory or time limit exceeded')
                    except psutil.NoSuchProcess:
                        break
                    time.sleep(0.05)
                stdout, stderr = process.communicate(timeout=30)
                if process.returncode:
                    raise RuntimeError(stderr + stdout)
                record=json.loads(stdout.strip().splitlines()[-1])
                record['repetition']=repetition
                records.append(record)
                path.write_text(json.dumps(records,indent=2)+'\n')
                print(json.dumps(record),flush=True)
