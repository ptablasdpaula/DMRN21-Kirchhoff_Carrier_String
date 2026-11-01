"""Test case isolation, statuses and the sampled-memory event protocol."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks import figure_comparison_driver as driver


class FakeNVML:
    def read(self, pid): return 123456789


class DriverTests(unittest.TestCase):
    def run_script(self, script, timeout=2):
        original = subprocess.Popen
        def popen(_cmd, **kwargs):
            return original([sys.executable, '-u', '-c', script], **kwargs)
        with tempfile.TemporaryDirectory() as tmp, patch.object(driver.subprocess, 'Popen', popen), patch.object(driver, 'ProcessMemory', FakeNVML):
            return driver.run_case('fdtd', 1, 4., timeout, sys.executable, Path(tmp), .005, 10**12)

    def test_success_samples_only_measurement(self):
        row = {'method':'fdtd', 'batch':1, 'status':'ok', 'median_seconds':.1}
        script = ("import json,time; print(json.dumps({'event':'measure_begin'}),flush=True);"
                  "time.sleep(.15);print(json.dumps({'event':'measure_end'}),flush=True);"
                  f"print(json.dumps({{'event':'result','row':{row!r}}}),flush=True)")
        result = self.run_script(script)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['peak_gpu_bytes'], 123456789)
        self.assertGreater(result['gpu_memory_samples'], 0)

    def test_oom_is_not_zero_measurement(self):
        result = self.run_script("raise RuntimeError('CUDA out of memory')")
        self.assertEqual(result['status'], 'oom')
        self.assertIsNone(result['peak_gpu_bytes'])

    def test_timeout_is_not_oom(self):
        result = self.run_script('import time; time.sleep(2)', timeout=.1)
        self.assertEqual(result['status'], 'timeout')
        self.assertIsNone(result['peak_gpu_bytes'])


if __name__ == '__main__':
    unittest.main()
