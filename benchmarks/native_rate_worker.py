"""JAX modal solver at 44.1 kHz: one isolated batch/loss/gradient case."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('JAX_ENABLE_X64', 'true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')


def emit(event, **kwargs):
    print(json.dumps({'event': event, **kwargs}, allow_nan=False), flush=True)


def run_case(batch, duration, modes, rate, artifact, excitation='multimode', loss='mae', checkpoint_steps=None,
             readout_in_scan=False):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch
    import jaxdiffmodal.time_integrators as integrators
    from benchmarks.exact_methods import BOX, GRID
    from experiments.native_rate_study.renderer import renderer
    from experiments.figure_comparison.physical import physical_values, gradient_scales
    from experiments.jax_loss_benchmark import cel, mae, prepare_target, prepare_mae_target, anti_alias
    from src.optimization import criterion_for
    from src.kc_string_forward import nominal_parameters
    from src.synthesis import Renderer

    torch.set_num_threads(1)
    if jax.default_backend() != 'gpu': raise RuntimeError('Requires JAX GPU backend')
    if rate not in (44100, GRID.integration_rate):
        raise ValueError(f'Only the native rate and the FDTD internal rate ({GRID.integration_rate} Hz) are supported')
    count = round(duration * 44100)
    internal_samples = round(duration*rate)
    if loss == 'cel' and count < 1024: raise ValueError('Target is too short for the CeL STFT')
    model = Renderer(GRID, duration, excitation)
    target = model.synthesize(BOX.defaults()).detach()
    physical = physical_values(batch)
    x = jnp.asarray(physical.numpy(), dtype=jnp.float64)
    prepare, group_loss = (prepare_target, cel) if loss == 'cel' else (prepare_mae_target, mae)
    prepared = jax.jit(prepare)(jnp.asarray(target.numpy(), dtype=jnp.float64))
    kernel = jnp.asarray(model.filter.reshape(-1).numpy(), dtype=jnp.float64)

    def loss_and_audio(v):
        # When oversampled, retain the two extra points used by the project
        # anti-alias filter; at 1x, output exactly the requested number of samples.
        samples = internal_samples + 2 if rate != 44100 else internal_samples
        raw = renderer(v, samples, rate, modes, GRID.intervals, excitation, checkpoint_steps, readout_in_scan)
        audio = anti_alias(raw, kernel, count, GRID.oversampling) if rate != 44100 else raw
        value = sum(group_loss(audio[i:i+16], prepared)*min(16, batch-i)
                    for i in range(0, batch, 16))
        return value, audio

    start = time.perf_counter()
    executable = jax.jit(jax.value_and_grad(loss_and_audio, has_aux=True)).lower(x).compile()
    compile_seconds = time.perf_counter()-start
    start = time.perf_counter()
    (value, audio), gradient = executable(x)
    jax.block_until_ready((value, audio, gradient))
    warmup = time.perf_counter()-start
    audio_cpu, grad_cpu, loss_cpu = np.asarray(audio), np.asarray(gradient), float(value)
    if not (np.isfinite(audio_cpu).all() and np.isfinite(grad_cpu).all() and np.isfinite(loss_cpu)):
        raise RuntimeError('Non-finite output or gradients; stability not established')

    # This diagnostic is deliberately outside the timed workload and does not
    # claim exact equality: the modal integrator, discretization and filter are
    # different. The reference is exactly the same physical parameter vector.
    ref_params = physical[0].detach().clone().requires_grad_()
    ref_audio = model._filter_audio(model.model(ref_params / nominal_parameters(), duration))
    loss_fn = criterion_for(target, GRID.sample_rate, loss)
    ref_loss = loss_fn(ref_audio)
    ref_grad, = torch.autograd.grad(ref_loss, ref_params)
    ref_audio_np, ref_grad_np = ref_audio.detach().numpy(), ref_grad.detach().numpy()
    scales = gradient_scales('cpu').numpy()
    relative = lambda a,b: float(np.linalg.norm(a-b)/max(np.linalg.norm(b), np.finfo(float).tiny))
    diagnostic = {
        'reference': f'{GRID.integration_rate} Hz oversampled FDTD (order {GRID.order}, {GRID.intervals} intervals), filtered to 44100 Hz; first batch item',
        'first_item_audio_relative_l2_vs_fdtd': relative(audio_cpu[0], ref_audio_np),
        'first_item_fft_magnitude_relative_l2_vs_fdtd': relative(np.abs(np.fft.rfft(audio_cpu[0])), np.abs(np.fft.rfft(ref_audio_np))),
        'first_item_scaled_gradient_relative_l2_vs_fdtd': relative(grad_cpu[0]*scales, ref_grad_np*scales),
        'first_item_loss': float(loss_fn(torch.from_numpy(np.array(audio_cpu[0], copy=True))).detach()),
        'first_item_fdtd_loss': float(ref_loss.detach()),
        'first_item_peak_abs': float(np.max(np.abs(audio_cpu[0]))),
        'first_item_rms': float(np.sqrt(np.mean(audio_cpu[0]**2))),
        'first_item_fdtd_peak_abs': float(np.max(np.abs(ref_audio_np))),
        'first_item_modal_gradient': grad_cpu[0].tolist(),
        'first_item_fdtd_gradient': ref_grad_np.tolist(),
        'finite': True,
    }
    # Save compact first-item diagnostics outside timing. Comparing these
    # artifacts across rates/mode counts never creates extra timed GPU memory.
    if artifact is not None:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(artifact, audio=audio_cpu[0], gradient=grad_cpu[0],
                            reference_audio=ref_audio_np, reference_gradient=ref_grad_np,
                            gradient_scales=scales, physical_parameters=physical[0].numpy(),
                            rate=rate, modes=modes)
    del audio_cpu, grad_cpu, audio, gradient, ref_audio, ref_grad, ref_loss, ref_params, ref_audio_np, ref_grad_np

    emit('measure_begin')
    samples = []
    for _ in range(3):
        start = time.perf_counter(); result = executable(x); jax.block_until_ready(result)
        samples.append(time.perf_counter()-start); del result
    emit('measure_end')
    device = jax.devices()[0]
    memory = device.memory_stats() or {}
    direct_url = Path(integrators.__file__).parent.parent / 'jaxdiffmodal-0.1.0.dist-info' / 'direct_url.json'
    name = f'diaz_{rate}' + ('_ckptro' if checkpoint_steps and readout_in_scan else '_ckpt' if checkpoint_steps else '')
    integrator = ('jaxdiffmodal.time_integrators.solve_tf' if not checkpoint_steps else
                  'experiments.native_rate_study.renderer.solve_tf_checkpointed: the upstream recurrence as a nested '
                  f'jax.lax.scan with jax.checkpoint on every {checkpoint_steps}-step segment'
                  + (', pickup read inside the scan (no modal trajectory returned)' if readout_in_scan else
                     ', full modal trajectory and velocities returned as upstream'))
    return {'method':name, 'batch':batch, 'status':'ok', 'duration_seconds':duration, 'checkpoint_steps':checkpoint_steps,
            'readout_in_scan':bool(readout_in_scan),
            'median_seconds':statistics.median(samples), 'repetitions_seconds':samples,
            'compile_seconds':compile_seconds, 'warmup_seconds':warmup,
            'peak_framework_allocated_bytes':memory.get('peak_bytes_in_use'),
            'framework_memory_stats_scope':'JAX cumulative device stats including compilation/warm-up',
            'validation':diagnostic, 'gpu_name':device.device_kind, 'jax_version':jax.__version__,
            'internal_rate':rate, 'output_rate':44100, 'output_samples':count,
            'oversampling':rate//44100, 'antialias_filter':rate!=44100, 'modes':modes, 'excitation':excitation,
            'fdtd_spatial_order':GRID.order,
            'initial_shape_note':f'{excitation} displacement sampled on the fixed {GRID.intervals}-interval grid and projected by h*(basis @ shape)',
            'first_item_diagnostic_artifact':str(artifact) if artifact is not None else None,
            'gradient_coordinates':['density', 'tension', 'radius', 'young_modulus', 'length', 'amplitude', 'sigma_0', 'sigma_1', 'pluck_position', 'pickup_position'],
            'loss':loss, 'loss_group':16, 'integrator':integrator,
            'upstream_vcs_info':json.loads(direct_url.read_text()).get('vcs_info',{}) if direct_url.exists() else {},
            'upstream_integrator_sha256':hashlib.sha256(Path(integrators.__file__).read_bytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--batch', type=int, required=True)
    p.add_argument('--duration', type=float, default=4.)
    p.add_argument('--modes', type=int, default=175)
    p.add_argument('--rate', type=int, default=44100, help='44100 or the FDTD internal rate')
    p.add_argument('--artifact', type=Path)
    p.add_argument('--excitation', default='multimode')
    p.add_argument('--loss', default='mae', choices=('mae', 'cel'))
    p.add_argument('--checkpoint-steps', type=int, default=None,
                   help='Rematerialise the upstream recurrence in segments of this many steps (jax.checkpoint)')
    p.add_argument('--readout-in-scan', action='store_true',
                   help='With --checkpoint-steps: read the pickup inside the scan instead of returning the modal trajectory')
    a = p.parse_args()
    try:
        row = run_case(a.batch, a.duration, a.modes, a.rate, a.artifact, a.excitation, a.loss, a.checkpoint_steps,
                       a.readout_in_scan)
    except Exception as exc:
        traceback.print_exc(); s = str(exc).lower()
        status = 'oom' if 'out of memory' in s or 'resource_exhausted' in s else 'error'
        row = {'method':f'diaz_{a.rate}' + ('_ckptro' if a.checkpoint_steps and a.readout_in_scan else '_ckpt' if a.checkpoint_steps else ''),
               'batch':a.batch, 'status':status,
               'excitation':a.excitation, 'loss':a.loss, 'checkpoint_steps':a.checkpoint_steps,
               'modes':a.modes, 'duration_seconds':a.duration, 'error':f'{type(exc).__name__}: {exc}'}
    emit('result', row=row)


if __name__ == '__main__': main()
