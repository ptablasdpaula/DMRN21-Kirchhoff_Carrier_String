"""Matched single/batched exact-recurrence benchmarks; internal research harness.

Every item uses a distinct valid parameter vector and an independently evaluated
CeL objective against the same original-CPU target. Output JSON grows after every
row so successful observations survive scheduler limits or out-of-memory cases.
"""
import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from src.accelerated import evaluate, kernel
from src.grid import GridConfig
from src.kc_string_forward import ForwardString, nominal_parameters
from src.optimization import criterion_for
from src.parameters import BenchmarkBox, BenchmarkMaterialBox
from src.synthesis import Renderer

# The benchmark's strings are small perturbations of the nominal string, so the
# narrow box admits 176 intervals at 2x oversampling with the five-point stencil.
# BOX keeps the perceptual coordinates of the older harnesses; the timed sweep
# perturbs the ten physical parameters directly (MATERIAL_BOX, +-3 %), and the
# same grid is validated against that box's stability corner here (0.93).
BOX = BenchmarkBox()
GRID = GridConfig(BOX, oversampling=2, intervals=176)
MATERIAL_BOX = BenchmarkMaterialBox()
MATERIAL_GRID = GridConfig(MATERIAL_BOX, oversampling=GRID.oversampling, intervals=GRID.intervals)
assert MATERIAL_GRID.integration_rate == GRID.integration_rate


def theta_of(values):
    return torch.vmap(BOX.to_material)(values) / nominal_parameters(values.device)


def coordinates_and_seeds(values):
    nominal = nominal_parameters(values.device)
    def transform(row):
        return BOX.to_material(row) / nominal
    return torch.vmap(transform)(values), torch.vmap(torch.func.jacfwd(transform))(values)


class RawTangents(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, steps, variant):
        audio, jac = evaluate(theta, GRID.intervals, GRID.integration_rate, steps, 1,
                              'multimode', 'modal', True, variant=variant)
        ctx.save_for_backward(jac)
        return audio

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        (jac,) = ctx.saved_tensors
        return torch.einsum('bt,btd->bd', grad, jac), None, None


class SeededTangents(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, steps, variant):
        theta, seeds = coordinates_and_seeds(values)
        audio, jac = evaluate(theta, GRID.intervals, GRID.integration_rate, steps, 1,
                              'multimode', 'modal', True, seeds=seeds, variant=variant)
        ctx.save_for_backward(jac)
        return audio

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        (jac,) = ctx.saved_tensors
        return torch.einsum('bt,btd->bd', grad, jac), None, None


def varied_parameters(n):
    default = BOX.defaults()
    i = torch.arange(n, dtype=torch.float64)[:, None]
    j = torch.arange(default.numel(), dtype=torch.float64)[None, :]
    values = default[None, :] * torch.exp(.045 * torch.sin((i+1)*(j+1)*.73))
    low, high = BOX.bounds()
    values = torch.maximum(low, torch.minimum(high, values))
    for row in values: BOX.validate(row)
    return values


def varied_material(n):
    """n strings with every one of the ten physical parameters moved by up to +-3 %.

    The same deterministic pattern as ``varied_parameters`` applied to the SI
    parameters of the nominal string, so item i is the same string in every
    batch size and run; the target is the nominal string itself.
    """
    from src.parameters import BENCHMARK_MATERIAL_SPREAD
    default = MATERIAL_BOX.defaults()
    i = torch.arange(n, dtype=torch.float64)[:, None]
    j = torch.arange(default.numel(), dtype=torch.float64)[None, :]
    values = default[None, :] * torch.exp(BENCHMARK_MATERIAL_SPREAD * torch.sin((i+1)*(j+1)*.73))
    low, high = MATERIAL_BOX.bounds()
    values = torch.maximum(low, torch.minimum(high, values))
    for row in values: MATERIAL_BOX.validate(row)
    return values


def filter_audio(raw, renderer):
    if renderer.filter is not None:
        raw = F.conv1d(F.pad(raw[:, None, :], (renderer.half_width, renderer.half_width),
                             mode='replicate'), renderer.filter, stride=GRID.oversampling)[:, 0, :]
    return raw[:, :renderer.samples]


def audio_for(values, renderer, method, *, tangents):
    """Full batch audio, with ordinary differentiability when tangents is true."""
    steps = round(renderer.duration * GRID.integration_rate)
    if method == 'cpu-native':
        theta = theta_of(values)
        raw = torch.stack([renderer.model(row, renderer.duration) for row in theta])
    elif method == 'cuda-adjoint':
        from src.adjoint import simulate
        raw = simulate(theta_of(values), GRID.intervals, GRID.integration_rate, steps, 'multimode')
    elif method == 'cuda-seeded':
        if tangents:
            raw = SeededTangents.apply(values, steps, 0)
        else:
            raw, _ = evaluate(theta_of(values), GRID.intervals, GRID.integration_rate, steps,
                              1, 'multimode', 'modal', False, variant=0)
    else:
        variant = {'cpu-modal': 0, 'cuda-modal': 0, 'cuda-recompute': 1,
                   'cuda-register': 2}[method]
        theta = theta_of(values)
        if tangents:
            raw = RawTangents.apply(theta, steps, variant)
        else:
            raw, _ = evaluate(theta, GRID.intervals, GRID.integration_rate, steps, 1,
                              'multimode', 'modal', False, variant=variant)
    return filter_audio(raw, renderer)


def explicit_tangents(values, duration, method):
    steps = round(duration * GRID.integration_rate)
    theta = theta_of(values)
    if method == 'cpu-native':
        model = ForwardString(GRID.intervals, GRID.integration_rate, excitation='multimode')
        return tuple(torch.stack(items) for items in zip(*(model.output_and_tangents(row, duration) for row in theta)))
    if method == 'cuda-seeded':
        theta, seeds = coordinates_and_seeds(values)
        return evaluate(theta, GRID.intervals, GRID.integration_rate, steps, 1, 'multimode',
                        'modal', True, seeds=seeds, variant=0)
    variant = 1 if method == 'cuda-recompute' else 0
    return evaluate(theta, GRID.intervals, GRID.integration_rate, steps, 1, 'multimode',
                    'modal', True, variant=variant)


def sync(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)


def measure(fn, device, repeats):
    sync(device); warm_started=time.perf_counter(); fn(); sync(device)
    warmup_seconds=time.perf_counter()-warm_started
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
    wall, events = [], []
    for _ in range(repeats):
        sync(device)
        if device.type == 'cuda':
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        t = time.perf_counter(); fn()
        if device.type == 'cuda': end.record()
        sync(device); wall.append(time.perf_counter()-t)
        if device.type == 'cuda': events.append(start.elapsed_time(end)/1000)
    return {'warmup_seconds': warmup_seconds, 'median_seconds': statistics.median(wall), 'min_seconds': min(wall),
            'max_seconds': max(wall), 'repetitions_seconds': wall,
            'cuda_event_seconds': events,
            'peak_cuda_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None}


def digest():
    h = hashlib.sha256()
    for f in sorted((ROOT/'src').glob('*')):
        if f.suffix in ('.py', '.cpp', '.cu', '.h'):
            h.update(f.name.encode()+f.read_bytes())
    return h.hexdigest()


def metadata():
    cpu_name = platform.processor()
    if Path('/proc/cpuinfo').exists():
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                cpu_name=line.split(':',1)[1].strip(); break
    return {'source_sha256': digest(), 'revision': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
            'source_status': subprocess.check_output(['git','status','--short'], cwd=ROOT, text=True),
            'host': platform.node(), 'cpu': cpu_name, 'job': os.environ.get('SLURM_JOB_ID'),
            'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
            'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            'dtype': 'float64', 'cpu_threads': torch.get_num_threads(),
            'intervals': GRID.intervals, 'output_rate': GRID.sample_rate,
            'internal_rate': GRID.integration_rate, 'oversampling': GRID.oversampling,
            'spatial_order': GRID.order, 'parameter_box': 'BenchmarkBox exp(+-0.05) (perceptual harness); timed sweep: BenchmarkMaterialBox exp(+-0.03) on ten parameters', 'excitation': 'multimode',
            'target': 'same original CPU reference, all parameter vectors',
            'loss': 'independent STFTCumulativeEnergyLoss per item, summed for backward'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--validation',type=Path,help='Current-source numerical validation JSON')
    p.add_argument('--validation-short',type=Path,help='Short-batch numerical validation JSON')
    p.add_argument('--methods', default='cpu-native,cpu-modal,cuda-modal,cuda-recompute,cuda-register,cuda-seeded')
    p.add_argument('--batches', default='1,2,4,8,16,32,64')
    p.add_argument('--durations', default='.25,4')
    p.add_argument('--kinds', default='synthesis,tangents,vjp,loss_backward')
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--quick', action='store_true')
    args = p.parse_args()
    args.methods = args.methods.split(',')
    args.batches = [int(x) for x in args.batches.split(',')]
    args.durations = [float(x) for x in args.durations.split(',')]
    args.kinds = args.kinds.split(',')
    choices = ('cpu-native','cpu-modal','cuda-modal','cuda-recompute','cuda-register','cuda-seeded','cuda-adjoint')
    if any(m not in choices for m in args.methods): p.error(f'methods: {choices}')
    if any(k not in ('synthesis','tangents','vjp','loss_backward') for k in args.kinds): p.error('invalid kind')
    if args.repeats < 1 or any(b < 1 for b in args.batches) or any(d <= 0 for d in args.durations): p.error('positive values required')
    if args.quick: args.batches, args.durations, args.repeats = [1,4], [.25], min(args.repeats,2)
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {'metadata': metadata(), 'rows': []}
    result['metadata']['benchmark_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.validation:
        evidence=json.loads(args.validation.read_text())
        if evidence['status']!='passed' or evidence['source_sha256']!=digest() or evidence['source_changed_during_run']:
            raise RuntimeError('Validation does not match the current source')
        result['metadata']['validation']=str(args.validation)
        result['metadata']['validation_evidence']=evidence
    if args.validation_short:
        evidence=json.loads(args.validation_short.read_text())
        if evidence['status']!='passed' or evidence['source_sha256']!=digest() or evidence['source_changed_during_run']:
            raise RuntimeError('Short validation does not match the current source')
        result['metadata']['validation_short']=str(args.validation_short)
        result['metadata']['validation_short_evidence']=evidence
    def save(): args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    def add(row):
        result['rows'].append(row); save(); print(json.dumps(row, allow_nan=False), flush=True)
    save()
    for duration in args.durations:
        # The identical target is computed once with the original CPU solver, outside timing.
        target = Renderer(GRID, duration).synthesize(BOX.defaults()).detach()
        for method in args.methods:
            device = torch.device('cuda' if method.startswith('cuda') else 'cpu')
            if device.type == 'cuda' and not torch.cuda.is_available():
                add({'method':method,'duration':duration,'error':'CUDA unavailable'}); continue
            if device.type == 'cuda':
                resources=result['metadata'].setdefault('kernel_resources',{})
                if method not in resources:
                    if method=='cuda-modal':
                        kind,regs,local,shared,maximum=kernel(True).resources(True,GRID.intervals-1,256)
                        resources[method]=dict(columns=9,storage=kind,registers_per_thread=regs,local_bytes_per_thread=local,shared_bytes=shared,max_threads=maximum)
                    elif method in ('cuda-recompute','cuda-register'):
                        variant=1 if method=='cuda-recompute' else 2
                        resources[method]=[]
                        for columns in ((1,9) if variant==1 else (1,)):
                            regs,local,shared,maximum=kernel(True).variant_resources(variant,columns,GRID.intervals-1,256)
                            resources[method].append(dict(columns=columns,registers_per_thread=regs,local_bytes_per_thread=local,shared_bytes=shared,max_threads=maximum))
                    else:
                        resources[method]='not provided by extension'
            renderer = Renderer(GRID, duration, device=device, backend='native')
            criterion = criterion_for(target.to(device), GRID.sample_rate)
            initial = varied_parameters(max(args.batches))
            for batch in args.batches:
                values = initial[:batch].to(device)
                steps = round(duration * GRID.integration_rate)
                for kind in args.kinds:
                    if (method == 'cuda-register' and kind != 'synthesis') or (method == 'cuda-adjoint' and kind == 'tangents'): continue
                    try:
                        if kind == 'synthesis':
                            def fn():
                                with torch.no_grad(): return audio_for(values, renderer, method, tangents=False)
                        elif kind == 'tangents':
                            def fn():
                                with torch.no_grad(): return explicit_tangents(values, duration, method)
                        else:
                            pval = values.detach().clone().requires_grad_()
                            if kind == 'vjp':
                                # Differentiation of a nontrivial arbitrary audio-space VJP.
                                probe = torch.linspace(-1, 1, renderer.samples, device=device, dtype=torch.float64)
                                def fn():
                                    pval.grad = None
                                    (audio_for(pval, renderer, method, tangents=True)*probe).sum().backward()
                            else:
                                def fn():
                                    pval.grad = None
                                    audio = audio_for(pval, renderer, method, tangents=True)
                                    torch.stack([criterion(audio[i]) for i in range(batch)]).sum().backward()
                        timing = measure(fn, device, args.repeats)
                        add(dict(method=method, kind=kind, batch=batch, duration=duration,
                                 tangent_coordinates=('perceptual' if method=='cuda-seeded' else 'material') if kind=='tangents' else None,
                                 items_per_second=batch/timing['median_seconds'], **timing))
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        add(dict(method=method, kind=kind, batch=batch, duration=duration, error='CUDA out of memory'))
                        continue
            del criterion
    result['metadata']['source_changed_during_run'] = digest() != result['metadata']['source_sha256']
    save(); print('BENCHMARK COMPLETE', flush=True)

if __name__ == '__main__': main()
