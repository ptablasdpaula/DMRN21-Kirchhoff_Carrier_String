import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from benchmarks.native_rate_plot import normalized,render


class PlotTests(unittest.TestCase):
    def test_native_175_only_on_main_plot_and_modes_in_table(self):
        rows=[{'method':'cached3','batch':1,'status':'ok','median_seconds':1.,'peak_gpu_bytes':1e9,'gpu_name':'A100'},
              {'method':'diaz_44100','modes':175,'batch':1,'status':'ok','median_seconds':2.,'peak_gpu_bytes':2e9,'gpu_name':'A100'},
              {'method':'diaz_44100','modes':8,'batch':1,'status':'ok','median_seconds':.5,'peak_gpu_bytes':.5e9,'gpu_name':'A100'},
              {'method':'diaz_132300','modes':175,'batch':1,'status':'ok','median_seconds':3.,'peak_gpu_bytes':3e9,'gpu_name':'A100'}]
        payload={'metadata':{'status':'running'},'rows':rows}
        index=normalized(payload)
        self.assertEqual(index['diaz',1]['median_seconds'],2.)
        with tempfile.TemporaryDirectory() as tmp:
            prefix=Path(tmp)/'figure';render(payload,prefix)
            self.assertTrue(prefix.with_suffix('.png').exists())
            text=prefix.with_suffix('.md').read_text()
            self.assertIn('1.000 s / 1.000 GB',text)
            self.assertIn('132.3 kHz',text)
            self.assertIn('0.500 s / 0.500 GB',prefix.with_name('figure-modes.md').read_text())

    def test_primary_modes_and_rate_from_metadata(self):
        rows=[{'method':'cached3','batch':1,'status':'ok','median_seconds':.4,'peak_gpu_bytes':.6e9,'gpu_name':'A100'},
              {'method':'fdtd','batch':1,'status':'ok','median_seconds':200.,'peak_gpu_bytes':3e9,'gpu_name':'A100'},
              {'method':'diaz_44100','modes':100,'batch':1,'status':'ok','median_seconds':.5,'peak_gpu_bytes':1e9,'gpu_name':'A100'},
              {'method':'diaz_44100','modes':123,'batch':1,'status':'ok','median_seconds':.6,'peak_gpu_bytes':1.1e9,'gpu_name':'A100'},
              {'method':'diaz_88200','modes':100,'batch':1,'status':'ok','median_seconds':1.,'peak_gpu_bytes':2e9,'gpu_name':'A100'}]
        payload={'metadata':{'status':'running','native_modal_modes':[100,123],'primary_modal_modes':100,
                             'cached_rate':88200,'oversampling':2,'spatial_order':4,'duration_seconds':1,
                             'excitation':'raised_cosine'},'rows':rows}
        index=normalized(payload)
        self.assertEqual(index['diaz',1]['median_seconds'],.5)
        with tempfile.TemporaryDirectory() as tmp:
            prefix=Path(tmp)/'figure';render(payload,prefix)
            text=prefix.with_suffix('.md').read_text()
            self.assertIn('88.2 kHz',text);self.assertIn('spatial order 4',text)
            modes=prefix.with_name('figure-modes.md').read_text()
            self.assertIn('100 modes',modes);self.assertIn('0.600 s / 1.100 GB',modes)

    def test_invalid_run_not_plotted(self):
        with self.assertRaises(ValueError):normalized({'metadata':{'status':'invalid_source_changed'},'rows':[]})


if __name__=='__main__':unittest.main()
