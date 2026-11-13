"""Isolated full-workload worker for the genuine upstream JAX modal solver."""
import hashlib
import importlib.metadata
import json
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from benchmarks.exact_methods import BOX, GRID
from experiments.figure_comparison.diaz import renderer
from experiments.figure_comparison.physical import physical_values, gradient_scales
from experiments.jax_loss_benchmark import anti_alias, cel, mae, prepare_target, prepare_mae_target
from src.optimization import criterion_for
from src.kc_string_forward import nominal_parameters
from src.synthesis import Renderer


def _relative(a, b):
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(aa-bb) / max(np.linalg.norm(bb), np.finfo(float).tiny))


def run_case(batch, duration, emit, excitation='multimode', loss='mae'):
    """Measure full synthesis/filter/loss/ten-coordinate VJP, returning one row.

    The upstream solver returns full modal trajectories. We intentionally do
    not rewrite it to stream only audio; an actual allocation failure is an OOM.
    """
    import jaxdiffmodal.time_integrators as integrators
    import jaxdiffmodal.ftm as ftm

    torch.set_num_threads(1)
    if jax.default_backend() != 'gpu':
        raise RuntimeError('Díaz benchmark requires JAX CUDA GPU backend')
    model = Renderer(GRID, duration, excitation)
    target = model.synthesize(BOX.defaults()).detach()
    all_values = physical_values(batch)
    # CPU-only arrays are transferred to JAX. No PyTorch CUDA context/pool is
    # created in this isolated process; NVML therefore describes JAX itself.
    x = jnp.asarray(all_values.numpy(), dtype=jnp.float64)
    prepare, group_loss = (prepare_target, cel) if loss == 'cel' else (prepare_mae_target, mae)
    prepared = jax.jit(prepare)(jnp.asarray(target.numpy(), dtype=jnp.float64))
    kernel = jnp.asarray(model.filter.reshape(-1).numpy(), dtype=jnp.float64)
    steps = round(duration * GRID.integration_rate)

    def forward_and_loss(v):
        raw = renderer(v, steps, GRID.intervals, GRID.integration_rate, excitation)
        audio = anti_alias(raw, kernel, target.shape[-1], GRID.oversampling)
        # Each common-target group is independently normalised; multiply each
        # group mean by its size to equal the sum of per-string loss values.
        groups = [group_loss(audio[start:start+16], prepared)
                  * min(16, batch-start) for start in range(0, batch, 16)]
        return sum(groups), audio

    candidate = jax.jit(jax.value_and_grad(forward_and_loss, has_aux=True))
    started = time.perf_counter()
    executable = candidate.lower(x).compile()
    compile_seconds = time.perf_counter()-started
    started = time.perf_counter()
    (value, audio), gradient = executable(x)
    jax.block_until_ready((value, audio, gradient))
    warmup_seconds = time.perf_counter()-started
    audio_cpu = np.asarray(audio)
    grad_cpu = np.asarray(gradient)
    loss_cpu = float(value)
    if not (np.isfinite(audio_cpu).all() and np.isfinite(grad_cpu).all()
            and np.isfinite(loss_cpu)):
        raise RuntimeError('Upstream modal solver produced non-finite audio, loss or gradient')

    # Record actual discretization difference for the first input. This is a
    # diagnostic, not an equality assertion: continuous modal TF and FDTD have
    # different spatial/time discretizations and initial time-step schemes.
    # Build the reference directly from the eight material values, including
    # density and length, and differentiate them outside the timed workload.
    reference_parameters = all_values[0].detach().clone().requires_grad_()
    reference_raw = model.model(reference_parameters / nominal_parameters(), duration)
    reference = model._filter_audio(reference_raw)
    candidate0 = torch.from_numpy(np.array(audio_cpu[0], copy=True))
    loss_fn = criterion_for(target, GRID.sample_rate, loss)
    diaz_item_loss = float(loss_fn(candidate0).detach())
    ref_item_loss = float(loss_fn(reference).detach())
    reference_loss = loss_fn(reference)
    reference_grad, = torch.autograd.grad(reference_loss, reference_parameters)
    reference_grad_np = reference_grad.detach().numpy()
    scales = gradient_scales('cpu').numpy()
    diagnostics = {'first_item_audio_relative_l2_vs_fdtd': _relative(audio_cpu[0], reference.detach().numpy()),
                   'first_item_modal_loss': diaz_item_loss,
                   'first_item_fdtd_loss': ref_item_loss,
                   'first_item_scaled_gradient_relative_l2_vs_fdtd': _relative(grad_cpu[0]*scales, reference_grad_np*scales),
                   'first_item_modal_gradient': grad_cpu[0].tolist(),
                   'first_item_fdtd_gradient': reference_grad_np.tolist(),
                   'gradients_finite': True}
    del audio_cpu, grad_cpu, gradient, audio, value, candidate0, reference
    del reference_raw, reference_loss, reference_grad, reference_parameters

    emit('measure_begin')
    samples = []
    for _ in range(3):
        started = time.perf_counter()
        result = executable(x)
        jax.block_until_ready(result)
        samples.append(time.perf_counter()-started)
        del result
    emit('measure_end')
    device = jax.devices()[0]
    memory = device.memory_stats() or {}
    direct_url = Path(integrators.__file__).parent.parent / 'jaxdiffmodal-0.1.0.dist-info' / 'direct_url.json'
    revision = json.loads(direct_url.read_text()).get('vcs_info', {}) if direct_url.exists() else {}
    return {'method': 'diaz', 'batch': batch, 'duration_seconds': duration, 'status': 'ok',
            'median_seconds': statistics.median(samples), 'repetitions_seconds': samples,
            'compile_seconds': compile_seconds, 'warmup_seconds': warmup_seconds,
            'peak_framework_allocated_bytes': memory.get('peak_bytes_in_use'),
            'framework_memory_stats_scope': 'JAX cumulative device stats, not reset between compilation/warm-up and measurement',
            'validation': diagnostics, 'gpu_name': device.device_kind,
            'jax_version': jax.__version__, 'gradient_coordinates': 10,
            'coordinate_names': ['density', 'tension', 'radius', 'young_modulus', 'length', 'amplitude', 'sigma_0', 'sigma_1', 'pluck_position', 'pickup_position'],
            'loss': loss, 'loss_group': 16, 'modes': GRID.intervals-1, 'excitation': excitation,
            'integrator': 'jaxdiffmodal.time_integrators.solve_tf',
            'upstream_vcs_info': revision,
            'upstream_integrator_sha256': hashlib.sha256(Path(integrators.__file__).read_bytes()).hexdigest(),
            'upstream_ftm_sha256': hashlib.sha256(Path(ftm.__file__).read_bytes()).hexdigest()}
