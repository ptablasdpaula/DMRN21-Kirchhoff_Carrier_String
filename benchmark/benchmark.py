"""Time and memory of one gradient step for the four implementations of Table 1.

    python benchmark/benchmark.py --method modes --batch 1 2 4 --output results.json

Each case runs in its own process. The parent samples the GPU's used memory with
``nvidia-smi`` while it runs (the table's peak memory) and records the child's
report: the median wall time of ``--repetitions`` gradient steps after one
warm-up step that includes compilation. Every case first checks the first
string's audio and gradient against ``scheme.reference``. A case that runs out
of memory is recorded with status "oom".

The rows the paper's table reads are ``method``, ``batch``, ``status``,
``median_seconds`` and ``peak_gpu_bytes``.
"""

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

# Table 1's rows: (row name used in the result files, JAX checkpoint, basis, unroll) or the kernel.
METHODS = {
    "auto": ("fdtd_jax", None, "grid", 32),
    "checkpoints": ("fdtd_jax_ckpt", 256, "grid", 32),
    "modes": ("fdtd_jax_sine_ckpt", 256, "sine", 64),
    "hand-written": ("replay", None, None, None),
}


def _jax_step(method, values, target):
    """A compiled function returning the batch's summed MAE, its audio and its gradient (JAX autodiff)."""
    import jax
    import jax.numpy as jnp

    from kc_string import jax_scheme, render, workload

    _, checkpoint, basis, unroll = METHODS[method]
    grid = workload.GRID
    taps = jnp.asarray(render.antialias_kernel(grid.oversampling).flatten().numpy())
    half = (taps.shape[0] - 1) // 2

    def decimate(raw):
        padded = jnp.pad(raw, ((0, 0), (half, half)), mode="edge")
        filtered = jax.lax.conv_general_dilated(
            padded[..., None],
            taps[:, None, None],
            (grid.oversampling,),
            "VALID",
            dimension_numbers=("NWC", "WIO", "NWC"),
        )
        return filtered[..., 0][:, : workload.SAMPLES]

    def loss(physical):
        raw = jax_scheme.render(
            physical,
            workload.STEPS,
            grid.intervals,
            grid.integration_rate,
            workload.EXCITATION,
            checkpoint=checkpoint,
            unroll=unroll,
            basis=basis,
        )
        audio = decimate(raw)
        return jnp.sum(jnp.mean(jnp.abs(audio - target), axis=-1)), raw

    step = jax.jit(jax.value_and_grad(loss, has_aux=True))

    def run():
        (value, raw), gradient = step(values)
        jax.block_until_ready((value, raw, gradient))
        return raw, gradient

    return run


def _kernel_step(values, target):
    """The batch's summed MAE backpropagated through the hand-written adjoint."""
    import torch

    from kc_string import adjoint, physics, render, workload

    taps = render.antialias_kernel(workload.GRID.oversampling, "cuda")
    target = torch.as_tensor(target, device="cuda")

    def run():
        theta = physics.multipliers(torch.as_tensor(values, device="cuda")).requires_grad_()
        raw = adjoint.simulate(
            theta,
            workload.GRID.intervals,
            workload.GRID.integration_rate,
            workload.STEPS,
            workload.EXCITATION,
            workload.CHECKPOINT,
        )
        audio = render.decimate(raw, workload.GRID.oversampling, workload.SAMPLES, taps)
        workload.mae(audio, target).backward()
        torch.cuda.synchronize()
        # In physical units, as JAX returns it.
        return raw.detach().cpu().numpy(), (theta.grad / physics.nominal_parameters("cuda")).cpu().numpy()

    return run


def _reference_check(raw, gradient, values, target):
    """Relative errors of the first string's audio and gradient against ``scheme.reference``."""
    import numpy as np
    import torch

    from kc_string import physics, render, scheme, workload

    theta = physics.multipliers(torch.as_tensor(values[0]))
    grid = workload.GRID
    audio, jacobian = scheme.reference(
        theta, grid.intervals, grid.integration_rate, workload.STEPS, workload.EXCITATION
    )
    signal = audio[None].clone().requires_grad_()
    decimated = render.decimate(signal, grid.oversampling, workload.SAMPLES)
    workload.mae(decimated, torch.as_tensor(target)).backward()
    expected = (signal.grad[0] @ jacobian / physics.nominal_parameters()).numpy()
    scale = physics.nominal_parameters().numpy()

    def relative(actual, wanted):
        return float(np.linalg.norm(actual - wanted) / np.linalg.norm(wanted))

    return {
        "audio_relative_l2": relative(np.asarray(raw[0]), audio.numpy()),
        "scaled_gradient_relative_l2": relative(np.asarray(gradient[0]) * scale, expected * scale),
    }


def worker(method, batch, repetitions):
    """One case in this process; prints its report as JSON."""
    import numpy as np

    from kc_string import workload

    values = workload.batch(batch).numpy()
    target = _target()
    run = _kernel_step(values, target) if method == "hand-written" else _jax_step(method, values, target)
    start = time.perf_counter()
    raw, gradient = run()
    warmup = time.perf_counter() - start
    check = _reference_check(np.asarray(raw), np.asarray(gradient), values, target)
    if check["audio_relative_l2"] > 1e-9 or check["scaled_gradient_relative_l2"] > 1e-7:
        raise AssertionError(f"{method} disagrees with the reference: {check}")
    seconds = []
    for _ in range(repetitions):
        start = time.perf_counter()
        run()
        seconds.append(time.perf_counter() - start)
    return {"warmup_seconds": warmup, "repetitions_seconds": seconds, "validation": check}


def _target():
    """The nominal string's audio at 44.1 kHz (the reference, so the target does not depend on the method)."""
    import torch

    from kc_string import physics, render, scheme, workload

    grid = workload.GRID
    theta = torch.ones(physics.PARAMETER_COUNT, dtype=torch.float64)
    audio, _ = scheme.reference(theta, grid.intervals, grid.integration_rate, workload.STEPS, workload.EXCITATION)
    return render.decimate(audio[None], grid.oversampling, workload.SAMPLES)[0].numpy()


def _sample_gpu_memory(stop, peak):
    while not stop.is_set():
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        if output.strip():
            peak[0] = max(peak[0], int(output.split()[0]) * 2**20)
        time.sleep(0.05)


def run_case(method, batch, repetitions):
    """A case in a child process while the GPU's memory is sampled."""
    stop, peak = threading.Event(), [0]
    sampler = threading.Thread(target=_sample_gpu_memory, args=(stop, peak), daemon=True)
    sampler.start()
    command = [sys.executable, __file__, "--worker", "--method", method, "--batch", str(batch)]
    command += ["--repetitions", str(repetitions)]
    child = subprocess.run(command, capture_output=True, text=True, check=False)
    stop.set()
    sampler.join()
    row = {"method": METHODS[method][0], "batch": batch, "peak_gpu_bytes": peak[0]}
    if child.returncode == 0:
        report = json.loads(child.stdout.strip().splitlines()[-1])
        return {**row, "status": "ok", "median_seconds": statistics.median(report["repetitions_seconds"]), **report}
    out_of_memory = any(word in child.stderr for word in ("out of memory", "RESOURCE_EXHAUSTED", "OutOfMemory"))
    if not out_of_memory:
        raise RuntimeError(f"{method} at batch {batch} failed:\n{child.stderr[-4000:]}")
    return {**row, "status": "oom"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--batch", type=int, nargs="+", default=None)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.method, args.batch[0], args.repetitions)))
        return
    from kc_string import workload

    rows = [run_case(args.method, batch, args.repetitions) for batch in args.batch or workload.BATCH_SIZES]
    text = json.dumps({"rows": rows}, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
