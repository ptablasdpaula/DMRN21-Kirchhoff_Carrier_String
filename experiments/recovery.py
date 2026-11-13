"""Reproducible six-parameter, full-duration recovery with independent optimizers."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.parameters import ParameterBox, reference_parameters, NAMES
from src.grid import GridConfig
from src.synthesis import Renderer
from src.optimization import fit,save_result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--duration',type=float,default=4.)
    parser.add_argument('--max-steps',type=int,default=500)
    parser.add_argument('--output',type=Path,default=Path('experiments/results'))
    args=parser.parse_args()
    torch.set_num_threads(1)
    box=ParameterBox();grid=GridConfig(box);renderer=Renderer(grid,args.duration)
    target_parameters=reference_parameters();initial=box.perturb(target_parameters)
    target=renderer.synthesize(target_parameters)
    print(grid.describe(),flush=True)
    for name in ('adam','lbfgs'):
        def report(row):
            if row['iteration']%10==0: print(name,row['iteration'],row['best_loss'],flush=True)
        result=fit(renderer,target,initial,optimizer_name=name,max_steps=args.max_steps,progress=report)
        prediction=renderer.synthesize(result['best_physical'])
        from src.optimization import criterion_for
        verified=criterion_for(target,grid.sample_rate)(prediction).item()
        assert abs(verified-result['best_loss']) < 1e-10
        save_result(args.output/f'recovery_{name}.json',result,renderer,target_parameters)
        print(name,result['stop_reason'],result['seconds'],flush=True)
        for key,start,truth,final in zip(NAMES,initial,target_parameters,result['best_physical']):
            print(f'{key}: start={start:.7g}, target={truth:.7g}, recovered={final:.7g}, error={100*(final/truth-1):+.3f}%')

if __name__=='__main__':main()
