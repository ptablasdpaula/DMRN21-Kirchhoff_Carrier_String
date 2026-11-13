"""Direct materials: whole-box stability, native waveform/gradients, fits and metadata."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import itertools
import json
import tempfile
import torch
from src.parameters import MaterialParameterBox,ParameterBox,perceptual_from_material
from src.grid import GridConfig
from src.synthesis import Renderer,modal_diagnostics
from src.kc_string_forward import ForwardString,nominal_parameters
from src.optimization import fit,save_result

def main():
    torch.set_num_threads(1)
    box=MaterialParameterBox();p=box.defaults();grid=GridConfig(box)
    torch.testing.assert_close(box.to_material(p),p,rtol=0,atol=0)
    torch.testing.assert_close(box.decode(box.encode(p)),p,rtol=1e-12,atol=1e-14)
    for magnitude in (-1000.,1000.):box.validate(box.decode(torch.full((8,),magnitude,dtype=torch.float64)))
    scores=[]
    for corner in itertools.product(*box.ranges):
        values=torch.tensor(corner,dtype=torch.float64)
        scores.append(float(grid.score_values(values)))
    assert max(scores)<=1 and abs(max(scores)-grid.corner_score)<1e-12
    def worst(N,OF):
        return max(grid.physical_score(c*c,K*K,sigma1,L,intervals=N,oversampling=OF)
                   for c,K,sigma1,L in box.stability_cases())
    assert worst(grid.intervals+2,grid.oversampling)>1
    if grid.oversampling>1:assert worst(grid.minimum_intervals,grid.oversampling-1)>1
    low,high=box.bounds();gen=torch.Generator().manual_seed(4)
    for _ in range(100):
        values=low+(high-low)*torch.rand(8,generator=gen,dtype=torch.float64)
        assert float(grid.score_values(values))<=grid.corner_score
    renderer=Renderer(grid,.003)
    relative=torch.ones(8,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x:renderer(p*x),(relative,),eps=1e-5,atol=1e-7,rtol=2e-3)
    # Independent native call with direct material multipliers, then the same output filter.
    raw=ForwardString(grid.intervals,grid.integration_rate,excitation='multimode')(p/nominal_parameters(),.003)
    if renderer.filter is not None:
        raw=torch.nn.functional.conv1d(torch.nn.functional.pad(raw.reshape(1,1,-1),
            (renderer.half_width,renderer.half_width),mode='replicate'),renderer.filter,stride=grid.oversampling).flatten()
    torch.testing.assert_close(renderer(p),raw[:renderer.samples],rtol=0,atol=0)
    assert len(modal_diagnostics(p,grid))==8
    renderer=Renderer(grid,.04);target=renderer.synthesize(p);initial=box.perturb(p)
    # Fit all eight in either optimizer; then verify exact freezing in a subset fit.
    for method in ('adam','lbfgs'):
        result=fit(renderer,target,initial,optimizer_name=method,max_steps=2)
        assert len(result['mask'])==8 and result['iterations']==2
        assert result['best_loss']<=result['history'][0]['loss']
        mask=[False]*8;mask[1]=mask[5]=True
        frozen=fit(renderer,target,initial,mask=mask,optimizer_name=method,max_steps=2)
        keep=~torch.tensor(mask)
        assert torch.equal(frozen['best_physical'][keep],initial[keep])
        assert renderer.grid is grid
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'fit.json';save_result(path,result,renderer,p)
            saved=json.loads(path.read_text())
            assert saved['coordinate_system']=='material' and saved['parameter_names']==list(box.names)
    print(grid.describe());print('Material bounds, all corners, gradients, synthesis, both fits and metadata passed.')

if __name__=='__main__':main()
