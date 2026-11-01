"""Checks for honest provenance and failed-case handling in shard merging."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.figure_comparison_merge import merge_shards


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.meta = {'status':'complete', 'duration':4., 'memory_metric':'NVML',
                     'memory_sample_interval_seconds':.01, 'timeout_seconds_per_case':1800.,
                     'host_rss_limit_gb':32., 'git_revision':'revision',
                     'jaxdiffmodal_provenance':{'commit':'abc'}, 'source_fingerprints':{'source':'hash'},
                     'source_changed_during_run':False, 'job_id':'111', 'hostname':'a',
                     'gpu_name':'NVIDIA A100 80GB PCIe', 'gpu_uuid':'GPU-ONE'}
        self.row = {'method':'cached3', 'batch':1, 'status':'ok', 'duration_seconds':4.,
                    'median_seconds':1., 'peak_gpu_bytes':1000, 'gradient_coordinates':10}

    def tearDown(self): self.temp.cleanup()

    def save(self, name, metadata=None, rows=None):
        path = self.root / name
        path.write_text(json.dumps({'metadata':metadata or self.meta,
                                    'rows':rows if rows is not None else [self.row]}))
        return path

    def test_completed_and_failed_preserved_with_accurate_provenance(self):
        first = self.save('one.json')
        other_meta = dict(self.meta, job_id='222', hostname='b', gpu_uuid='GPU-TWO')
        failed = {'method':'diaz', 'batch':128, 'status':'oom'}
        other = self.save('two.json', other_meta, [failed])
        result = merge_shards([first, other])
        self.assertEqual(result['metadata']['status'], 'running')
        self.assertEqual(result['metadata']['gpu_uuids'], ['GPU-ONE', 'GPU-TWO'])
        self.assertEqual(len(result['metadata']['missing_cases']), 30)
        self.assertEqual(next(r for r in result['rows'] if r['method']=='diaz')['status'], 'oom')
        self.assertEqual(result['metadata']['shards'][1]['hostname'], 'b')
        with self.assertRaisesRegex(ValueError, 'one physical GPU UUID'):
            merge_shards([first, other], require_same_uuid=True)

    def test_same_uuid_and_successful_gradient_names(self):
        first = self.save('one.json')
        row = dict(self.row, method='diaz', batch=2, gradient_coordinates=10,
                   coordinate_names=['density','tension','radius','young_modulus',
                                     'length','amplitude','sigma_0','sigma_1',
                                     'pluck_position','pickup_position'])
        second = self.save('two.json', dict(self.meta, job_id='222'), [row])
        result = merge_shards([first, second], require_same_uuid=True)
        self.assertEqual(result['metadata']['gpu_uuids'], ['GPU-ONE'])
        self.assertEqual(len(result['rows']), 2)

    def test_pytorch_and_nvml_uuid_spelling_match(self):
        bare = '2d7784b2-430c-7bfd-ed55-e07814e62968'
        row = dict(self.row, gpu_uuid=bare, gpu_devices={'GPU-' + bare:'NVIDIA A100 80GB PCIe'})
        first = self.save('one.json', dict(self.meta, gpu_uuid=None), [row])
        result = merge_shards([first], require_same_uuid=True)
        self.assertEqual(result['metadata']['gpu_uuids'], [bare])

    def test_failed_row_nvml_identity_and_consistency(self):
        bare = '2d7784b2-430c-7bfd-ed55-e07814e62968'
        meta = dict(self.meta, gpu_uuid=None, gpu_name=None)
        row = {'method':'diaz','batch':1,'status':'oom',
               'gpu_devices':{'GPU-' + bare:'NVIDIA A100 80GB PCIe'}}
        first = self.save('one.json', meta, [row])
        result = merge_shards([first], require_same_uuid=True)
        self.assertEqual(result['metadata']['gpu_uuids'], [bare])
        self.assertEqual(result['metadata']['gpu_name'], 'NVIDIA A100 80GB PCIe')
        conflict = dict(row, gpu_name='NVIDIA H100 80GB HBM3')
        first = self.save('one.json', meta, [conflict])
        with self.assertRaisesRegex(ValueError, 'conflicting GPU models'):
            merge_shards([first])

    def test_malformed_nvml_identity_rejected(self):
        row = dict(self.row, gpu_devices={'GPU-ONE':None})
        path = self.save('one.json', rows=[row])
        with self.assertRaisesRegex(ValueError, 'invalid NVML GPU identity'):
            merge_shards([path])

    def test_unknown_uuid_cannot_claim_same_card(self):
        first = self.save('one.json', dict(self.meta, gpu_uuid=None))
        self.assertEqual(merge_shards([first])['metadata']['gpu_uuid_unverified_shards'], [str(first)])
        with self.assertRaisesRegex(ValueError, 'one physical GPU UUID'):
            merge_shards([first], require_same_uuid=True)

    def test_mixed_model_rejected(self):
        first = self.save('one.json')
        second = self.save('two.json', dict(self.meta, gpu_name='NVIDIA H100 80GB HBM3'),
                           [dict(self.row, batch=2)])
        with self.assertRaisesRegex(ValueError, 'models differ'):
            merge_shards([first, second])

    def test_missing_model_for_success_is_rejected(self):
        first = self.save('one.json', dict(self.meta, gpu_name=None))
        with self.assertRaisesRegex(ValueError, 'no GPU model'):
            merge_shards([first])

    def test_failed_unknown_model_is_explicit(self):
        shard = self.save('one.json', dict(self.meta, gpu_name=None, gpu_uuid=None),
                          [{'method':'diaz','batch':1,'status':'oom'}])
        result = merge_shards([shard])
        self.assertEqual(result['metadata']['gpu_model_unverified_shards'], [str(shard)])
        self.assertIsNone(result['metadata']['gpu_name'])

    def test_incompatible_source_and_metadata_rejected(self):
        first = self.save('one.json')
        for key, value in (('source_fingerprints', {'source':'other'}), ('duration', .25),
                           ('memory_metric', 'torch'), ('timeout_seconds_per_case', 60)):
            second = self.save('two.json', dict(self.meta, **{key:value}), [dict(self.row, batch=2)])
            with self.subTest(key=key), self.assertRaises(ValueError):
                merge_shards([first, second])

    def test_changed_source_and_duplicates_rejected(self):
        first = self.save('one.json')
        second = self.save('two.json', dict(self.meta, source_changed_during_run=True), [dict(self.row, batch=2)])
        with self.assertRaisesRegex(ValueError, 'provenance'):
            merge_shards([first, second])
        second = self.save('two.json')
        with self.assertRaisesRegex(ValueError, 'duplicate case'):
            merge_shards([first, second])

    def test_incorrect_coordinate_and_measurement_rejected(self):
        for changed in ({'gradient_coordinates':6}, {'gradient_coordinates':8},
                        {'coordinate_names':['density','tension','radius','young_modulus',
                                             'length','amplitude','sigma_0','sigma_1']},
                        {'peak_gpu_bytes':None},
                        {'median_seconds':0}, {'duration_seconds':.25}):
            path = self.save('one.json', rows=[dict(self.row, **changed)])
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                merge_shards([path])

    def test_running_shard_is_provisional(self):
        path = self.save('one.json', dict(self.meta, status='running'))
        self.assertEqual(merge_shards([path])['metadata']['status'], 'running')


if __name__ == '__main__':
    unittest.main()
