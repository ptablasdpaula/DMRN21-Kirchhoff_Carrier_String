"""Bounded fits retain a fixed grid, frozen parameters and optimizer patience."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.parameters import ParameterBox,reference_parameters
from src.grid import GridConfig
from src.synthesis import Renderer
from src.optimization import fit,should_stop

def main():
    torch.set_num_threads(1)
    box=ParameterBox();grid=GridConfig(box);renderer=Renderer(grid,.04)
    truth=reference_parameters();target=renderer.synthesize(truth);initial=box.perturb(truth)
    for name in ('adam','lbfgs'):
        result=fit(renderer,target,initial,mask=[False,False,True,False,False,False],optimizer_name=name,max_steps=2)
        fixed=torch.tensor([True,True,False,True,True,True])
        assert torch.equal(result['best_physical'][fixed],initial[fixed])
        assert result['best_loss']<=result['history'][0]['loss']
        assert result['iterations']==2 and renderer.grid is grid
        # A differentiable constant objective must trigger the actual patience loop.
        result=fit(renderer,target,initial,optimizer_name=name,max_steps=30,criterion=lambda audio:audio.sum()*0+1)
        assert result['iterations']==20
    result=fit(renderer,target,initial,mask=[False]*6,max_steps=1)
    assert result['iterations']==0 and torch.equal(result['best_physical'],initial)
    assert not should_stop([dict(best_loss=1.)]*20)
    assert should_stop([dict(best_loss=1.)]*21)
    assert not should_stop([dict(best_loss=1.)]*20+[dict(best_loss=.8)])
    low,high=box.bounds()
    endpoint=initial.clone();endpoint[4]=0
    result=fit(renderer,target,endpoint,max_steps=0)
    assert result['effective_initial'][4]>0
    assert bool(((result['best_physical']>=low)&(result['best_physical']<=high)).all())
    print('Bounded fitting, independent starts, frozen values and 20-step patience passed.')

if __name__=='__main__':main()
