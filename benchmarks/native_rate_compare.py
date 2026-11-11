"""Isolated 44.1-kHz JAX and oversampled cached-FDTD comparisons on one GPU.

The FDTD grid and internal rate come from ``benchmarks.exact_methods.GRID``;
``diaz_internal`` names the JAX comparator at that internal rate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.figure_comparison_driver import ProcessMemory, rss_bytes, run_case, source_fingerprints, upstream_provenance


def fingerprints():
    result = source_fingerprints()
    for file in (Path(__file__), ROOT/'benchmarks/native_rate_worker.py', ROOT/'benchmarks/native_rate_plot.py',
                 ROOT/'experiments/native_rate_study/renderer.py',
                 ROOT/'hpc/native-rate-pilot.sh', ROOT/'hpc/native-rate-sweep.sh', ROOT/'hpc/native-rate-jit.sh'):
        if file.exists(): result[str(file.relative_to(ROOT))] = hashlib.sha256(file.read_bytes()).hexdigest()
    return result


def native_case(batch, duration, timeout, python, outdir, modes, rate, interval=.01, host_limit=32*10**9,
                excitation='multimode', loss='mae', checkpoint_steps=None, readout_in_scan=False, scheme='modal',
                unroll=8, basis='grid'):
    outdir.mkdir(parents=True, exist_ok=True)
    tag = '-ckptro' if checkpoint_steps and readout_in_scan else '-ckpt' if checkpoint_steps else ''
    stem = f'fdtd-jax-{basis}-{rate}-b{batch}{tag}' if scheme == 'fdtd' else f'diaz-{rate}-m{modes}-b{batch}{tag}'
    logpath = outdir/f'{stem}.log'
    artifact = outdir/f'{stem}.npz'
    command = [python, str(ROOT/'benchmarks/native_rate_worker.py'), '--batch', str(batch), '--duration', str(duration),
               '--modes', str(modes), '--rate', str(rate), '--artifact', str(artifact), '--excitation', excitation,
               '--loss', loss]
    if checkpoint_steps: command += ['--checkpoint-steps', str(checkpoint_steps)]
    if readout_in_scan: command += ['--readout-in-scan']
    if scheme == 'fdtd': command += ['--scheme', 'fdtd', '--unroll', str(unroll), '--basis', basis]
    proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, start_new_session=True)
    lines = queue.Queue()
    def pump():
        for line in proc.stdout: lines.put(line)
    reader = threading.Thread(target=pump, daemon=True); reader.start()
    try:
        memory, memory_error = ProcessMemory(), None
    except (OSError, RuntimeError, AttributeError) as exc:
        memory, memory_error = None, str(exc)
    start = time.monotonic(); peak = None; samples = 0; peak_host = 0
    measuring = False; timed_out = False; host_limit_hit = False; result = None; captured = []
    with logpath.open('w') as log:
        while proc.poll() is None or reader.is_alive() or not lines.empty():
            try:
                while True:
                    line = lines.get_nowait(); log.write(line); log.flush(); captured.append(line)
                    try: item = json.loads(line)
                    except (ValueError, TypeError): continue
                    if item.get('event') == 'measure_begin': measuring = True
                    if item.get('event') == 'measure_end': measuring = False
                    if item.get('event') == 'result': result = item['row']
            except queue.Empty: pass
            if memory is not None and proc.poll() is None:
                try:
                    v = memory.read(proc.pid)
                    if measuring and v is not None: peak = max(peak or 0, v); samples += 1
                except RuntimeError as exc: memory_error, memory = str(exc), None
            if proc.poll() is None: peak_host = max(peak_host, rss_bytes(proc.pid))
            if proc.poll() is None and (time.monotonic()-start > timeout or peak_host > host_limit):
                timed_out = time.monotonic()-start > timeout; host_limit_hit = not timed_out
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            if proc.poll() is None: time.sleep(interval)
        proc.wait(); reader.join(); proc.stdout.close()
    if result is None:
        text = ''.join(captured).lower()
        status = ('timeout' if timed_out else 'host_memory_limit' if host_limit_hit else
                  'oom' if 'out of memory' in text or 'resource_exhausted' in text else 'error')
        result = {'method':(f'fdtd_jax{"_sine" if basis == "sine" else ""}{"_ckpt" if checkpoint_steps else ""}' if scheme == 'fdtd' else
                            f'diaz_{rate}{"_ckptro" if checkpoint_steps and readout_in_scan else "_ckpt" if checkpoint_steps else ""}'),
                  'batch':batch, 'modes':None if scheme == 'fdtd' else modes,
                  'status':status, 'error':f'exit={proc.returncode}; see {logpath}'}
    result.update(peak_gpu_bytes=peak, gpu_memory_metric='NVML sampled process memory',
                  gpu_memory_samples=samples, gpu_memory_sample_interval_seconds=interval,
                  gpu_memory_error=memory_error, log=str(logpath), worker_exit_code=proc.returncode,
                  peak_sampled_host_rss_bytes=peak_host,
                  gpu_devices=getattr(memory,'observed',{}) if memory is not None else {})
    if result['status'] == 'ok' and (peak is None or proc.returncode):
        result['status'] = 'error'
        result['error'] = 'No comparable NVML measurement' if peak is None else f'worker exited {proc.returncode}'
    return result


def comparison_pairs(modes, internal_rate=132300):
    """(candidate, reference, label) artifact names: each count vs the primary, and the primary across rates."""
    primary=modes[0]
    pairs=[(f'diaz-44100-m{m}-b1.npz',f'diaz-44100-m{primary}-b1.npz',f'{m} vs {primary} modes at 44.1 kHz') for m in modes[1:]]
    pairs.append((f'diaz-44100-m{primary}-b1.npz',f'diaz-{internal_rate}-m{primary}-b1.npz',
                  f'44.1 vs {internal_rate/1000:g} kHz, {primary} modes'))
    return pairs


def compare_artifacts(logs, modes=(175,123,8), internal_rate=132300):
    """Compare first-item audio/10-gradients; missing artifacts stay missing."""
    import numpy as np
    def relative(a,b): return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),np.finfo(float).tiny))
    checks=[]
    for candidate,reference,label in comparison_pairs(list(modes), internal_rate):
        cp,rp=logs/candidate,logs/reference
        if not cp.exists() or not rp.exists():
            checks.append({'comparison':label,'status':'unavailable'});continue
        with np.load(cp) as c,np.load(rp) as r:
            if c['audio'].shape!=r['audio'].shape or not np.array_equal(c['physical_parameters'],r['physical_parameters']):
                checks.append({'comparison':label,'status':'incompatible'});continue
            a,b=c['audio'],r['audio'];scale=r['gradient_scales']
            checks.append({'comparison':label,'status':'measured',
                'audio_relative_l2':relative(a,b),
                'fft_magnitude_relative_l2':relative(np.abs(np.fft.rfft(a)),np.abs(np.fft.rfft(b))),
                'scaled_eight_gradient_relative_l2':relative(c['gradient']*scale,r['gradient']*scale),
                'candidate_artifact':str(cp),'reference_artifact':str(rp)})
    return checks


def host_cpu():
    """CPU model of the host: the eager PyTorch baselines are bound by it, not by the GPU."""
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return None


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batches', default='1,2,4,8,16,32,64,128')
    p.add_argument('--methods', default='cached3,cached1,diaz_44100,fdtd')
    p.add_argument('--duration', type=float, default=4.)
    p.add_argument('--modes', default='100', help='Modal mode counts; the first is the primary 44.1 kHz configuration')
    p.add_argument('--excitation', default='multimode')
    p.add_argument('--loss', default='mae', choices=('mae', 'cel'),
                   help='mae: sum over items of the mean |waveform - target|; cel: the STFT cumulative-energy loss')
    p.add_argument('--host-limit-gb', type=float, default=32.)
    p.add_argument('--timeout', type=float, default=1800.)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--plot-prefix',type=Path,help='Update the figure and tables after each case')
    p.add_argument('--compile-block', type=int, default=64,
                   help='fdtd_compiled, fdtd_ckpt_compiled: time steps unrolled into one torch.compile graph')
    p.add_argument('--compile-mode', default='default', choices=('default', 'reduce-overhead'),
                   help='fdtd_compiled, fdtd_ckpt_compiled: torch.compile mode')
    p.add_argument('--reentrant', action='store_true', help='fdtd_ckpt_compiled: reentrant torch.utils.checkpoint')
    p.add_argument('--unroll', type=int, default=8,
                   help='fdtd_jax, fdtd_jax_ckpt: time steps unrolled per iteration of the compiled scan')
    a = p.parse_args()
    from benchmarks.exact_methods import GRID
    internal = GRID.integration_rate; internal_method = f'diaz_{internal}'
    sizes = [int(x) for x in a.batches.split(',')]
    methods = [internal_method if m == 'diaz_internal' else m for m in a.methods.split(',')]
    modes = [int(x) for x in a.modes.split(',')]
    torch_methods = ('cached3','cached1','replay','fdtd','fdtd_ckpt','fdtd_ckpt_reentrant','fdtd_compiled','fdtd_ckpt_compiled')
    jax_fdtd_methods = ('fdtd_jax','fdtd_jax_ckpt','fdtd_jax_sine','fdtd_jax_sine_ckpt')
    if any(m not in (*torch_methods,*jax_fdtd_methods,internal_method,'diaz_44100','diaz_44100_ckpt','diaz_44100_ckptro') for m in methods):
        raise ValueError('Unknown method')
    compile_extra = ['--compile-block', str(a.compile_block), '--compile-mode', a.compile_mode] + (['--reentrant'] if a.reentrant else [])
    if not modes or any(not 1 <= m <= GRID.intervals-1 for m in modes) or len(set(modes)) != len(modes):
        raise ValueError(f'Mode counts must be distinct and lie in 1..{GRID.intervals-1}')
    host_limit = int(a.host_limit_gb*1e9)
    result = {'metadata':{'status':'running', 'job_id':os.getenv('SLURM_JOB_ID'),
              'hostname':os.uname().nodename, 'host_cpu':host_cpu(), 'gpu_visibility':os.getenv('CUDA_VISIBLE_DEVICES'),
              'duration_seconds':a.duration, 'batch_sizes':sizes, 'methods':methods,
              'excitation':a.excitation, 'loss':a.loss,
              'loss_definition':('sum over batch items of the mean absolute error of the 44.1 kHz waveform against the target'
                                 if a.loss == 'mae' else 'sum over batch items of the STFT cumulative-energy loss against the target'),
              'parameters':'ten: eight material parameters plus pluck and pickup positions (fractions of L), batch perturbed by up to +-3 % each (BenchmarkMaterialBox)',
              'cached_rate':internal, 'modal_rate':44100, 'output_rate':GRID.sample_rate,
              'oversampling':GRID.oversampling, 'intervals':GRID.intervals, 'spatial_order':GRID.order,
              'native_modal_modes':modes, 'primary_modal_modes':modes[0],
              'baseline_rematerialisation':{
                  'fdtd_ckpt':'torch.utils.checkpoint (use_reentrant=False) over 256-step chunks of the spatial PyTorch loop',
                  'fdtd_ckpt_reentrant':'the same eager chunks with use_reentrant=True: each chunk first runs under no_grad and is re-run with its own backward pass',
                  'diaz_44100_ckpt':'upstream solve_tf recurrence as a nested jax.lax.scan with jax.checkpoint on each 256-step segment, full modal trajectory returned as upstream',
                  'diaz_44100_ckptro':'the same checkpointed recurrence with the pickup read inside the scan, so no modal trajectory is materialised',
                  'fdtd_ckpt_compiled':f'torch.utils.checkpoint (use_reentrant={a.reentrant}) over 256-step chunks of compiled blocks',
                  'fdtd_jax_ckpt':'nested jax.lax.scan with jax.checkpoint on each 256-step segment, pickup read inside the scan'},
              'baseline_compilation':{
                  'fdtd_compiled':f'torch.compile (TorchInductor, mode {a.compile_mode}) of the spatial PyTorch loop in blocks of {a.compile_block} unrolled steps; every step kept for the reverse pass',
                  'fdtd_ckpt_compiled':f'the same compiled blocks ({a.compile_block} steps, mode {a.compile_mode}) inside 256-step checkpointed chunks',
                  'fdtd_jax':f'the same spatial scheme as a jax.lax.scan (unroll={a.unroll}) under jax.jit; every step kept for the reverse pass',
                  'fdtd_jax_ckpt':f'the same jax.jit scan (unroll={a.unroll}) with jax.checkpoint on each 256-step segment',
                  'fdtd_jax_sine':f'the same scheme stepped in the discrete sine basis (the CUDA adjoint\'s formulation) as a jax.jit scan (unroll={a.unroll}); every step kept',
                  'fdtd_jax_sine_ckpt':f'the sine-basis jax.jit scan (unroll={a.unroll}) with jax.checkpoint on each 256-step segment',
                  'timing':'compilation is excluded from the medians: it happens in the warm-up call (torch.compile) or is timed separately as compile_seconds (jax.jit)'},
              'memory_metric':'NVML sampled process memory incl. context/reservations',
              'sample_interval_seconds':.01, 'host_rss_limit_gb':a.host_limit_gb,
              'timeout_seconds_per_case':a.timeout, 'source_fingerprints':fingerprints(),
              'jaxdiffmodal_provenance':upstream_provenance()}, 'rows':[], 'cross_comparisons':[]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(a.output)
        if a.plot_prefix:
            if result['metadata']['status']=='invalid_source_changed':
                for ext in ('.png','.pdf','.csv','.md'):a.plot_prefix.with_suffix(ext).unlink(missing_ok=True)
                a.plot_prefix.with_name(a.plot_prefix.name+'-modes.md').unlink(missing_ok=True)
            else:
                from benchmarks.native_rate_plot import render
                render(result,a.plot_prefix)
    save(); logs=a.output.parent/(a.output.stem+'-logs')
    for method in methods:
        for batch in sizes:
            if method in torch_methods:
                row = run_case(method,batch,a.duration,a.timeout,a.python,logs,.01,host_limit,a.excitation,a.loss,
                               extra=compile_extra if method.endswith('_compiled') else ())
                row.update(internal_rate=internal,output_rate=GRID.sample_rate,oversampling=GRID.oversampling,
                           spatial_order=GRID.order,antialias_filter=GRID.oversampling>1,excitation=a.excitation,loss=a.loss)
                result['rows'].append(row);save();print(json.dumps(row),flush=True)
            elif method in jax_fdtd_methods:
                row=native_case(batch,a.duration,a.timeout,a.python,logs,modes[0],internal,excitation=a.excitation,loss=a.loss,
                                checkpoint_steps=256 if method.endswith('_ckpt') else None,scheme='fdtd',unroll=a.unroll,
                                basis='sine' if '_sine' in method else 'grid')
                row.update(internal_rate=internal,output_rate=GRID.sample_rate,oversampling=GRID.oversampling,
                           spatial_order=GRID.order,antialias_filter=GRID.oversampling>1,excitation=a.excitation,loss=a.loss)
                result['rows'].append(row);save();print(json.dumps(row),flush=True)
            else:
                # One oversampled primary-mode JAX comparison is enough to isolate
                # the rate; higher-batch oversampled runs exhaust GPU memory.
                if method == internal_method and batch != 1: continue
                mode_list = [modes[0]] if method == internal_method else modes
                for mode in mode_list:
                    row=native_case(batch,a.duration,a.timeout,a.python,logs,mode,
                                    internal if method == internal_method else 44100, excitation=a.excitation, loss=a.loss,
                                    checkpoint_steps=256 if method.endswith(('_ckpt', '_ckptro')) else None,
                                    readout_in_scan=method.endswith('_ckptro'))
                    result['rows'].append(row)
                    result['cross_comparisons']=compare_artifacts(logs, modes, internal)
                    save();print(json.dumps(row),flush=True)
    changed=fingerprints()!=result['metadata']['source_fingerprints']
    result['metadata'].update(source_changed_during_run=changed,status='invalid_source_changed' if changed else 'complete')
    save()


if __name__ == '__main__': main()
