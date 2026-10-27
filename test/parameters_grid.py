"""Physical mapping and independent checks of whole-box grid selection."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import math
import torch
from src.parameters import ParameterBox, reference_parameters, material_multipliers, material_parameters
from src.grid import GridConfig
from src import kc_string as reference

def main():
    box=ParameterBox(); values=reference_parameters()
    torch.testing.assert_close(material_multipliers(values),torch.ones(8,dtype=torch.float64),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(box.decode(box.encode(values)),values,rtol=1e-12,atol=1e-14)
    assert torch.autograd.gradcheck(material_parameters,(values.requires_grad_(),))
    grid=GridConfig(box)
    assert grid.corner_score <= 1
    corner=[box.ranges[i][1] for i in [0,1,5]]
    assert grid.score(*corner,intervals=grid.intervals+2)>1
    if grid.oversampling>1:
        assert grid.score(*corner,intervals=grid.minimum_intervals,oversampling=grid.oversampling-1)>1
    c=2*box.length*corner[0]; K=c*box.length*math.sqrt(corner[1])/math.pi
    h=reference.minimum_grid_spacing(1/grid.integration_rate,c,K,corner[2])*1.01
    assert math.isclose(grid.intervals,2*math.floor(box.length/h/2))
    generator=torch.Generator().manual_seed(1)
    for _ in range(100):
        p=box.decode(torch.randn(6,generator=generator,dtype=torch.float64)*3)
        assert grid.score(p[0],p[1],p[5])<=grid.corner_score
    for kwargs in [dict(sample_rate=0),dict(max_oversampling=1),dict(max_intervals=4)]:
        try: GridConfig(**kwargs)
        except ValueError: pass
        else: raise AssertionError(kwargs)
    print(grid.describe()); print('Parameter mapping, gradient and grid checks passed.')

if __name__=='__main__': main()
