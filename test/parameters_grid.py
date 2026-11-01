"""Physical mapping and independent checks of whole-box grid selection."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import math
import torch
from src.parameters import BenchmarkBox, BenchmarkMaterialBox, ParameterBox, reference_parameters, material_multipliers, material_parameters
from src.grid import GridConfig, reference_intervals
from src.kc_string_forward import PARAMETER_COUNT, eigenvalue_factor
from src import kc_string as reference

def main():
    box=ParameterBox(); values=reference_parameters()
    # Perceptual coordinates map to the eight material parameters plus the nominal pluck and pickup positions.
    torch.testing.assert_close(material_multipliers(values),torch.ones(PARAMETER_COUNT,dtype=torch.float64),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(box.decode(box.encode(values)),values,rtol=1e-12,atol=1e-14)
    assert torch.autograd.gradcheck(material_parameters,(values.requires_grad_(),))
    for magnitude in (-1000.,1000.):
        box.validate(box.decode(torch.full((6,),magnitude,dtype=torch.float64)))
    # The five-point stencil tightens the spacing bound by sqrt(4/3): 148 -> 128 at 44.1 kHz.
    assert reference_intervals(2)==148 and reference_intervals(4)==128
    for order,expected in ((4,(3,152)),(2,(3,176))):
        grid=GridConfig(box,order=order)
        assert (grid.oversampling,grid.intervals)==expected,(order,grid.oversampling,grid.intervals)
        assert grid.corner_score <= 1 and not grid.explicit
        corner=[box.ranges[i][1] for i in [0,1,5]]
        assert grid.score(*corner,intervals=grid.intervals+2)>1
        if grid.oversampling>1:
            assert grid.score(*corner,intervals=grid.minimum_intervals,oversampling=grid.oversampling-1)>1
        c=2*box.length*corner[0]; K=c*box.length*math.sqrt(corner[1])/math.pi
        h=reference.minimum_grid_spacing(1/grid.integration_rate,c,K,corner[2])*1.01*math.sqrt(eigenvalue_factor(order))
        assert math.isclose(grid.intervals,2*math.floor(box.length/h/2))
        generator=torch.Generator().manual_seed(1)
        for _ in range(100):
            p=box.decode(torch.randn(6,generator=generator,dtype=torch.float64)*3)
            assert grid.score(p[0],p[1],p[5])<=grid.corner_score
        print(grid.describe())
    # The benchmark fixes 176 intervals at 2x on its narrow box; the full box rejects that grid.
    benchmark=GridConfig(BenchmarkBox(),oversampling=2,intervals=176)
    assert benchmark.explicit and benchmark.integration_rate==88200 and benchmark.corner_score<=1
    assert all(low<=v<=high for v,(low,high) in zip(values.tolist(),BenchmarkBox().ranges))
    # The timed sweep moves all ten physical parameters by up to +-3 %; the same grid holds at that box's corner.
    material_box=BenchmarkMaterialBox();nominal=material_box.defaults()
    material=GridConfig(material_box,oversampling=2,intervals=176)
    assert material.explicit and material.integration_rate==88200 and material.corner_score<=1
    assert len(material_box.names)==PARAMETER_COUNT and nominal.shape==(PARAMETER_COUNT,)
    assert all(low<=v<=high for v,(low,high) in zip(nominal.tolist(),material_box.ranges))
    assert float(material.score_values(nominal))<=material.corner_score
    for kwargs in [dict(sample_rate=0),dict(max_oversampling=1),dict(max_intervals=4),dict(order=3),
                   dict(oversampling=2),dict(oversampling=2,intervals=176),dict(oversampling=2,intervals=175)]:
        try: GridConfig(box,**kwargs)
        except ValueError: pass
        else: raise AssertionError(kwargs)
    print(benchmark.describe()); print(material.describe()); print('Parameter mapping, gradient and grid checks passed.')

if __name__=='__main__': main()
