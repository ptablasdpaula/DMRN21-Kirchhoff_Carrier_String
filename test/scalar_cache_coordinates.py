"""Reuse the established independent coordinate stress test for each cache width."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.scalar_cache.reconstruct import simulate
import test.cached_coordinate_stress as stress

base_args=sys.argv
for count in (1,2,3):
    stress.reconstruct=lambda *args, _count=count:simulate(*args,cache_count=_count)
    sys.argv=[base_args[0], '--output', base_args[1].replace('.json',f'-{count}.json')]
    stress.main()
