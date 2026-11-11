"""Run each comparison in a fresh process on the same allocated GPU.

GPU process memory is sampled through NVML only during warmed evaluations;
this includes the CUDA context and allocator reservations, and may miss very
short peaks. Incremental JSON preserves completed and failed cases.
"""
import argparse
import ctypes
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


class _ProcessInfo(ctypes.Structure):
    _fields_ = [('pid', ctypes.c_uint), ('usedGpuMemory', ctypes.c_ulonglong),
                ('gpuInstanceId', ctypes.c_uint), ('computeInstanceId', ctypes.c_uint)]


class ProcessMemory:
    """Read NVML compute-process memory without adding a CUDA context."""
    def __init__(self):
        self.lib = ctypes.CDLL('libnvidia-ml.so.1')
        self._check(self.lib.nvmlInit_v2())
        count = ctypes.c_uint()
        self._check(self.lib.nvmlDeviceGetCount_v2(ctypes.byref(count)))
        self.handles = []
        for i in range(count.value):
            handle = ctypes.c_void_p()
            self._check(self.lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(handle)))
            self.handles.append(handle)
        self.observed = {}
        self.lib.nvmlDeviceGetUUID.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
        self.lib.nvmlDeviceGetName.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
        self.query = self.lib.nvmlDeviceGetComputeRunningProcesses_v2
        self.query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(_ProcessInfo)]

    @staticmethod
    def _check(code):
        if code:
            raise RuntimeError(f'NVML error {code}')

    def read(self, pid):
        result = None
        for handle in self.handles:
            n = ctypes.c_uint(0)
            code = self.query(handle, ctypes.byref(n), None)
            if code not in (0, 7):  # 7 is insufficient size, as expected.
                self._check(code)
            if not n.value:
                continue
            n.value += 8  # Allow other jobs to appear between calls.
            entries = (_ProcessInfo * n.value)()
            code = self.query(handle, ctypes.byref(n), entries)
            if code == 7:
                continue
            self._check(code)
            for entry in entries[:n.value]:
                if entry.pid == pid and entry.usedGpuMemory != 2**64-1:
                    result = (result or 0) + int(entry.usedGpuMemory)
                    uuid = ctypes.create_string_buffer(96)
                    name = ctypes.create_string_buffer(128)
                    self._check(self.lib.nvmlDeviceGetUUID(handle, uuid, len(uuid)))
                    self._check(self.lib.nvmlDeviceGetName(handle, name, len(name)))
                    self.observed[uuid.value.decode()] = name.value.decode()
        return result


def _pump(stream, output):
    for line in stream:
        output.put(line)


def source_fingerprints():
    paths = [p for d in ('src', 'experiments/figure_comparison', 'experiments/scalar_cache',
                         'experiments/loss_batch') for p in (ROOT/d).glob('*')
             if p.suffix in ('.py', '.cu', '.cpp', '.h')]
    paths += [ROOT/'benchmarks'/name for name in ('figure_comparison_driver.py',
              'figure_comparison_worker.py', 'figure_comparison_plot.py', 'exact_methods.py', 'reconstruction.py')]
    paths += [ROOT/'experiments/jax_loss_benchmark.py', ROOT/'hpc/figure-comparison.sh',
              ROOT/'hpc/figure-comparison-shard.sh']
    result = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(paths) if p.exists()}
    dependency = Path('/data/scratch/acw794/kc-string-gpu/diaz-deps/jaxdiffmodal')
    for p in sorted(dependency.glob('*.py')):
        result[f'upstream/jaxdiffmodal/{p.name}'] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def upstream_provenance():
    path = Path('/data/scratch/acw794/kc-string-gpu/diaz-deps/jaxdiffmodal-0.1.0.dist-info/direct_url.json')
    return json.loads(path.read_text()) if path.exists() else None


def rss_bytes(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError):
        pass
    return 0


def run_case(method, batch, duration, timeout, python, output_dir, poll_interval, host_limit_bytes,
             excitation='multimode', loss='mae', extra=()):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f'{method}-b{batch}.log'
    cmd = [python, str(ROOT/'benchmarks/figure_comparison_worker.py'),
           '--method', method, '--batch', str(batch), '--duration', str(duration),
           '--excitation', excitation, '--loss', loss, *extra]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, start_new_session=True)
    lines = queue.Queue()
    reader = threading.Thread(target=_pump, args=(proc.stdout, lines), daemon=True)
    reader.start()
    try:
        memory = ProcessMemory()
        memory_error = None
    except (OSError, RuntimeError, AttributeError) as exc:
        memory = None
        memory_error = str(exc)
    start = time.monotonic()
    measuring = False
    max_bytes = None
    samples = 0
    result = None
    timed_out = False
    host_limit_hit = False
    peak_host = 0
    captured = []
    with log_path.open('w') as log:
        while proc.poll() is None or reader.is_alive() or not lines.empty():
            try:
                while True:
                    line = lines.get_nowait()
                    log.write(line); log.flush(); captured.append(line)
                    try: message = json.loads(line)
                    except (ValueError, TypeError): continue
                    if message.get('event') == 'measure_begin': measuring = True
                    if message.get('event') == 'measure_end': measuring = False
                    if message.get('event') == 'result': result = message['row']
            except queue.Empty:
                pass
            if memory is not None and proc.poll() is None:
                try:
                    value = memory.read(proc.pid)
                    if measuring and value is not None:
                        max_bytes = max(max_bytes or 0, value); samples += 1
                except RuntimeError as exc:
                    memory_error = str(exc); memory = None
            if proc.poll() is None:
                peak_host = max(peak_host, rss_bytes(proc.pid))
            if proc.poll() is None and (time.monotonic()-start > timeout or peak_host > host_limit_bytes):
                timed_out = time.monotonic()-start > timeout
                host_limit_hit = not timed_out
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            if proc.poll() is None:
                time.sleep(poll_interval)
        proc.wait()
        reader.join()
        proc.stdout.close()
    if result is None:
        text = ''.join(captured).lower()
        if timed_out: status = 'timeout'
        elif host_limit_hit: status = 'host_memory_limit'
        elif 'out of memory' in text or 'cuda_error_out_of_memory' in text or 'resource_exhausted' in text: status = 'oom'
        else: status = 'error'
        result = {'method': method, 'batch': batch, 'status': status,
                  'error': f'exit={proc.returncode}; see {log_path}'}
    result.update(peak_gpu_bytes=max_bytes, gpu_memory_metric='NVML sampled process memory',
                  gpu_memory_samples=samples, gpu_memory_sample_interval_seconds=poll_interval,
                  gpu_memory_error=memory_error, log=str(log_path), worker_exit_code=proc.returncode,
                  peak_sampled_host_rss_bytes=peak_host,
                  gpu_devices=getattr(memory, 'observed', {}) if memory is not None else {})
    if result['status'] == 'ok' and (max_bytes is None or proc.returncode):
        result['status'] = 'error'
        result['error'] = 'No comparable NVML measurement' if max_bytes is None else f'worker exited {proc.returncode}'
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--batches', default='1,2,4,8,16,32,64,128')
    ap.add_argument('--methods', default='cached3,cached1,diaz,fdtd')
    ap.add_argument('--duration', type=float, default=4.)
    ap.add_argument('--timeout', type=float, default=1800.)
    ap.add_argument('--poll-interval', type=float, default=.01)
    ap.add_argument('--host-limit-gb', type=float, default=32.)
    ap.add_argument('--excitation', default='multimode')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--plot-prefix', type=Path, help='Update figure and tables after each case')
    args = ap.parse_args()
    result = {'metadata': {'status': 'running', 'duration': args.duration, 'excitation': args.excitation,
                          'batch_sizes': [int(b) for b in args.batches.split(',')],
                          'methods': args.methods.split(','), 'job_id': os.getenv('SLURM_JOB_ID'),
                          'hostname': os.uname().nodename, 'gpu_visibility': os.getenv('CUDA_VISIBLE_DEVICES'),
                          'git_revision': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                          'memory_metric': 'NVML sampled per-process allocation, including CUDA context and reservations',
                          'memory_sample_interval_seconds': args.poll_interval,
                          'timeout_seconds_per_case': args.timeout,
                          'host_rss_limit_gb': args.host_limit_gb,
                          'jaxdiffmodal_provenance': upstream_provenance(),
                          'source_fingerprints': source_fingerprints()}, 'rows': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temp = args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n'); temp.replace(args.output)
        if args.plot_prefix:
            if result['metadata']['status'] == 'invalid_source_changed':
                for suffix in ('.png', '.pdf', '.csv', '.md'):
                    args.plot_prefix.with_suffix(suffix).unlink(missing_ok=True)
                return
            from benchmarks.figure_comparison_plot import normalize_rows, write_tables, create_figure
            import matplotlib.pyplot as plt
            args.plot_prefix.parent.mkdir(parents=True, exist_ok=True)
            index = normalize_rows(result)
            write_tables(index, args.plot_prefix)
            fig, _ = create_figure(index)
            for suffix in ('.png', '.pdf'):
                fig.savefig(args.plot_prefix.with_suffix(suffix), dpi=300, bbox_inches='tight')
            plt.close(fig)
    save()
    for method in result['metadata']['methods']:
        for batch in result['metadata']['batch_sizes']:
            row = run_case(method, batch, args.duration, args.timeout, args.python,
                           args.output.parent / (args.output.stem + '-logs'), args.poll_interval,
                           int(args.host_limit_gb * 1e9), args.excitation)
            result['rows'].append(row); save(); print(json.dumps(row), flush=True)
    result['metadata']['source_changed_during_run'] = source_fingerprints() != result['metadata']['source_fingerprints']
    result['metadata']['status'] = 'invalid_source_changed' if result['metadata']['source_changed_during_run'] else 'complete'
    save()


if __name__ == '__main__':
    main()
