"""Merge native-rate sweep JSONs that share one workload into a single figure source.

Usage: merge_sweeps.py OUTPUT.json BASE.json EXTRA.json [EXTRA.json ...]

The base is the original sweep (e.g. job 34855605: cached3, fdtd, diaz_44100);
the extras add methods measured later under the same settings (e.g. the
rematerialised baselines fdtd_ckpt and diaz_44100_ckpt, possibly one JSON per
batch). Every input must be complete, must not have changed its sources during
the run, and must agree on duration, loss, excitation, grid, rates and mode
counts. Rows are concatenated; a (method, batch[, modes]) measured twice is an
error. The merged metadata keeps the base's fields, lists every input under
``sources`` (job, node, GPU, fingerprints, methods, batches) and sets
``batch_sizes`` to the union so that the figure script can check completeness
per method.
"""
import json
from pathlib import Path
import sys

SHARED = ('duration_seconds', 'loss', 'excitation', 'cached_rate', 'modal_rate', 'output_rate',
          'oversampling', 'intervals', 'spatial_order', 'primary_modal_modes', 'native_modal_modes')


def load(path):
    data = json.loads(Path(path).read_text())
    meta = data['metadata']
    if meta.get('status') != 'complete' or meta.get('source_changed_during_run'):
        raise ValueError(f'{path}: incomplete or sources changed during the run')
    return data


def merge(base_path, extra_paths):
    base = load(base_path)
    rows = list(base['rows'])
    sources = []
    seen = {(r.get('method'), r.get('batch'), r.get('modes')) for r in rows}
    gpus = {r.get('gpu_name') for r in rows if r.get('gpu_name')}
    batches = set(base['metadata']['batch_sizes'])

    def describe(path, data):
        meta = data['metadata']
        return {'file': Path(path).name, 'job_id': meta.get('job_id'), 'hostname': meta.get('hostname'),
                'host_cpu': meta.get('host_cpu'),
                'methods': meta.get('methods'), 'batch_sizes': meta.get('batch_sizes'),
                'gpu_names': sorted({r.get('gpu_name') for r in data['rows'] if r.get('gpu_name')}),
                'source_fingerprints': meta.get('source_fingerprints'),
                'jaxdiffmodal_provenance': meta.get('jaxdiffmodal_provenance')}

    sources.append(describe(base_path, base))
    for path in extra_paths:
        data = load(path)
        meta = data['metadata']
        for key in SHARED:
            if meta.get(key) != base['metadata'].get(key):
                raise ValueError(f'{path}: {key} = {meta.get(key)!r} differs from the base {base["metadata"].get(key)!r}')
        for row in data['rows']:
            key = (row.get('method'), row.get('batch'), row.get('modes'))
            if key in seen:
                raise ValueError(f'{path}: duplicate measurement {key}')
            seen.add(key); rows.append(row)
            if row.get('gpu_name'): gpus.add(row['gpu_name'])
        batches |= set(meta['batch_sizes'])
        sources.append(describe(path, data))
    if len(gpus) > 1:
        raise ValueError(f'Inputs were measured on different GPU models: {sorted(gpus)}')
    methods = []
    for source in sources:
        for m in source['methods'] or ():
            if m not in methods: methods.append(m)
    merged = {'metadata': {**base['metadata'], 'methods': methods, 'batch_sizes': sorted(batches),
                           'merged_from': [s['file'] for s in sources], 'sources': sources,
                           'baseline_rematerialisation': next((load(p)['metadata'].get('baseline_rematerialisation')
                                                               for p in extra_paths if load(p)['metadata'].get('baseline_rematerialisation')), None)},
              'rows': rows, 'cross_comparisons': base.get('cross_comparisons', [])}
    return merged


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    output, base, *extras = sys.argv[1:]
    merged = merge(base, extras)
    Path(output).write_text(json.dumps(merged, indent=1, allow_nan=False) + '\n')
    counts = {}
    for row in merged['rows']:
        counts.setdefault(row.get('method'), []).append((row.get('batch'), row.get('status')))
    for method, items in counts.items():
        print(method, sorted(items))
    print('Wrote', output)


if __name__ == '__main__':
    main()
