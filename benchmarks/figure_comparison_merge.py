"""Combine independent benchmark shards without concealing differences.

By default, known GPU *models* must agree. Distinct physical device UUIDs are
recorded, not hidden; use --require-same-uuid to insist on the very same card.
Failed cases remain rows with their original status. Missing cases remain
missing. Running shards are allowed as explicitly provisional snapshots.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CASE_METHODS = {'fdtd', 'cached3', 'cached1', 'diaz'}
CASE_BATCHES = {1, 2, 4, 8, 16, 32, 64, 128}
PHYSICAL_COORDINATES = ('density', 'tension', 'radius', 'young_modulus', 'length',
                        'amplitude', 'sigma_0', 'sigma_1', 'pluck_position', 'pickup_position')
COMPARABLE_METADATA = ('duration', 'memory_metric', 'memory_sample_interval_seconds',
                       'timeout_seconds_per_case', 'host_rss_limit_gb',
                       'git_revision', 'jaxdiffmodal_provenance')


def _name(value):
    return value.strip() if isinstance(value, str) and value.strip() else None


def _valid_uuid(value):
    value = _name(value)
    if not value or value.lower() in {'none', 'unknown', 'unavailable'}:
        return None
    # NVML prefixes GPU UUIDs with "GPU-" while PyTorch reports bare UUIDs.
    # Treat those spellings as the same physical device.
    if re.fullmatch(r'(?:GPU-)?[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', value, re.IGNORECASE):
        return re.sub(r'^GPU-', '', value, flags=re.IGNORECASE).lower()
    return value


def _names(values):
    """Case and whitespace variations do not describe different GPU models."""
    unique = {}
    for value in values:
        if name := _name(value):
            key = re.sub(r'\s+', ' ', name).casefold()
            unique.setdefault(key, name)
    return list(unique.values())


def merge_shards(paths, require_same_uuid=False):
    if not paths:
        raise ValueError('At least one shard is required')
    fingerprints = None
    comparable = None
    all_rows, sources, models, uuids = [], [], [], []
    seen_cases = set()
    unknown_model = []
    unknown_uuid = []
    completed = True
    for raw_path in paths:
        path = Path(raw_path)
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or not isinstance(data.get('metadata'), dict) or not isinstance(data.get('rows'), list):
            raise ValueError(f'{path}: expected metadata object and rows array')
        meta, rows = data['metadata'], data['rows']
        state = meta.get('status')
        if state not in ('running', 'complete') or meta.get('source_changed_during_run'):
            raise ValueError(f'{path}: invalid or failed source provenance ({state!r})')
        completed &= state == 'complete'
        current = meta.get('source_fingerprints')
        if not isinstance(current, dict) or not current or not all(isinstance(k, str) and isinstance(v, str) and v for k, v in current.items()):
            raise ValueError(f'{path}: missing source fingerprints')
        if fingerprints is None:
            fingerprints = current
        elif fingerprints != current:
            raise ValueError(f'{path}: source fingerprints differ from the other shards')
        current = {key: meta.get(key) for key in COMPARABLE_METADATA}
        if current['duration'] is None or current['memory_metric'] is None:
            raise ValueError(f'{path}: missing duration or memory metric')
        if comparable is None:
            comparable = current
        elif comparable != current:
            differences = [key for key in COMPARABLE_METADATA if comparable[key] != current[key]]
            raise ValueError(f'{path}: incompatible benchmark metadata: {", ".join(differences)}')

        # NVML observes UUID->model mappings directly, even when a worker
        # fails before it can report its framework device. Cross-check those
        # identities against any Torch/JAX or shard-level device fields.
        observed_devices = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            devices = row.get('gpu_devices', {})
            if not isinstance(devices, dict):
                raise ValueError(f'{path}: gpu_devices must be a UUID-to-model mapping')
            for device_uuid, device_model in devices.items():
                if not _valid_uuid(device_uuid) or not _name(device_model):
                    raise ValueError(f'{path}: invalid NVML GPU identity {device_uuid!r}: {device_model!r}')
                observed_devices.append((_valid_uuid(device_uuid), _name(device_model)))
        shard_models = _names([name for _, name in observed_devices] +
                              [meta.get('gpu_name'), meta.get('gpu_model'), meta.get('gpu')] +
                              [row.get('gpu_name') for row in rows if isinstance(row, dict)])
        if len(shard_models) > 1:
            raise ValueError(f'{path}: conflicting GPU models inside shard: {shard_models}')
        model = shard_models[0] if shard_models else None
        shard_uuids = sorted({_valid_uuid(value) for value in
                              [uid for uid, _ in observed_devices] + [meta.get('gpu_uuid')] +
                              [r.get('gpu_uuid') for r in rows if isinstance(r, dict)]
                              if _valid_uuid(value)})
        if len(shard_uuids) > 1:
            raise ValueError(f'{path}: conflicting physical GPU UUIDs inside shard: {shard_uuids}')
        uuid = shard_uuids[0] if shard_uuids else None
        if model: models.append(model)
        elif rows: unknown_model.append(str(path))
        if uuid: uuids.append(uuid)
        elif rows: unknown_uuid.append(str(path))

        for row in rows:
            if not isinstance(row, dict): raise ValueError(f'{path}: row must be an object')
            method, batch, status = row.get('method'), row.get('batch'), row.get('status')
            if method not in CASE_METHODS or type(batch) is not int or batch not in CASE_BATCHES or not isinstance(status, str) or not status:
                raise ValueError(f'{path}: invalid case {row!r}')
            if (method, batch) in seen_cases:
                raise ValueError(f'{path}: duplicate case {method} batch {batch}')
            seen_cases.add((method, batch))
            if row.get('duration_seconds', comparable['duration']) != comparable['duration']:
                raise ValueError(f'{path}: row duration differs from shard metadata')
            if status == 'ok':
                coordinates = row.get('coordinate_names', row.get('gradient_coordinates'))
                if coordinates != 10 and (not isinstance(coordinates, list) or tuple(coordinates) != PHYSICAL_COORDINATES):
                    raise ValueError(f'{path}: successful case does not declare all ten gradients')
                for metric in ('median_seconds', 'peak_gpu_bytes'):
                    value = row.get(metric)
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < float('inf'):
                        raise ValueError(f'{path}: successful case lacks positive {metric}')
                if not model:
                    raise ValueError(f'{path}: successful case has no GPU model identity')
            copied = dict(row)
            copied['shard_path'] = str(path)
            copied['shard_job_id'] = meta.get('job_id')
            copied['shard_hostname'] = meta.get('hostname')
            if model and not copied.get('gpu_name'): copied['gpu_name'] = model
            if uuid and not _valid_uuid(copied.get('gpu_uuid')): copied['gpu_uuid'] = uuid
            all_rows.append(copied)
        sources.append({'path': str(path), 'status': state, 'job_id': meta.get('job_id'),
                        'hostname': meta.get('hostname'), 'gpu_name': model, 'gpu_uuid': uuid,
                        'gpu_visibility': meta.get('gpu_visibility'), 'row_count': len(rows)})

    distinct_models = _names(models)
    if len(distinct_models) > 1:
        raise ValueError(f'GPU models differ across shards: {distinct_models}')
    distinct_uuids = sorted(set(uuids))
    if require_same_uuid and (unknown_uuid or len(distinct_uuids) != 1):
        raise ValueError('Cannot prove one physical GPU UUID across all shards')
    expected = {(method, batch) for method in CASE_METHODS for batch in CASE_BATCHES}
    missing = sorted(expected - seen_cases)
    metadata = {'status': 'complete' if completed and not missing else 'running',
                **comparable, 'source_fingerprints': fingerprints,
                'source_changed_during_run': False,
                'gpu_name': distinct_models[0] if distinct_models else None,
                'gpu_uuids': distinct_uuids,
                'physical_gpu_policy': 'same UUID required' if require_same_uuid else 'same recorded GPU model; UUIDs may differ',
                'gpu_model_unverified_shards': unknown_model,
                'gpu_uuid_unverified_shards': unknown_uuid,
                'missing_cases': [{'method': method, 'batch': batch} for method, batch in missing],
                'shards': sources}
    return {'metadata': metadata, 'rows': sorted(all_rows, key=lambda r: (r['method'], r['batch']))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('shards', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot-prefix', type=Path, help='Also render PNG/PDF and CSV/Markdown tables')
    parser.add_argument('--require-same-uuid', action='store_true')
    args = parser.parse_args(argv)
    merged = merge_shards(args.shards, args.require_same_uuid)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + '.tmp')
    tmp.write_text(json.dumps(merged, indent=2, allow_nan=False) + '\n'); tmp.replace(args.output)
    if args.plot_prefix:
        from benchmarks.figure_comparison_plot import main as plot_main
        plot_main([str(args.output), '--output-prefix', str(args.plot_prefix)])


if __name__ == '__main__':
    main()
