"""Matched full-duration existing optimisers on exact-recurrence research paths."""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from benchmarks.exact_methods import BOX,GRID,audio_for,digest
from src.optimization import criterion_for,fit,save_result
from src.parameters import reference_parameters
from src.synthesis import Renderer

class ResearchRenderer(Renderer):
    def __init__(self, duration, method):
        super().__init__(GRID,duration,device='cuda',backend='native')
        self.method=method
        self.backend=method  # Result provenance; the wrapped renderer remains native.
    def __call__(self, values):
        values=values.to(self.device)
        tangents=torch.is_grad_enabled() and values.requires_grad
        return audio_for(values.unsqueeze(0),self,self.method,tangents=tangents)[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',required=True,choices=('cuda-recompute','cuda-seeded','cuda-adjoint'))
    parser.add_argument('--duration',type=float,default=4.)
    parser.add_argument('--max-steps',type=int,default=500)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    source_hash=digest()
    target_parameters=reference_parameters()
    initial=BOX.perturb(target_parameters)
    # One native CPU target for both runs; all compilation and objective warmup precede fit timing.
    target_cpu=Renderer(GRID,args.duration).synthesize(target_parameters).detach()
    args.output.mkdir(parents=True,exist_ok=True)
    info=dict(source_sha256=source_hash,host=platform.node(),job=os.environ.get('SLURM_JOB_ID'),
              gpu=torch.cuda.get_device_name(),torch=torch.__version__,duration=args.duration,
              candidate=args.candidate,max_steps=args.max_steps,runs=[])
    for method in ('cuda-modal',args.candidate):
        renderer=ResearchRenderer(args.duration,method)
        target=target_cpu.to(renderer.device)
        criterion=criterion_for(target,GRID.sample_rate)
        start=time.perf_counter()
        warm=initial.to(renderer.device).detach().clone().requires_grad_()
        criterion(renderer(warm)).backward();torch.cuda.synchronize()
        warm_seconds=time.perf_counter()-start
        del warm
        for name in ('adam','lbfgs'):
            def report(row):
                if row['iteration']%10==0: print(method,name,row['iteration'],row['best_loss'],flush=True)
            result=fit(renderer,target,initial,optimizer_name=name,max_steps=args.max_steps,
                       progress=report,criterion=criterion)
            result['hardware']=dict(host=platform.node(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,
                                    job=os.environ.get('SLURM_JOB_ID'),source_sha256=source_hash,
                                    research_method=method,warmup_seconds=warm_seconds)
            prediction=renderer.synthesize(result['best_physical'])
            verified=criterion_for(target,GRID.sample_rate)(prediction).item()
            assert abs(verified-result['best_loss'])<1e-10,(method,name,verified,result['best_loss'])
            output=args.output/f'{method}-{name}.json'
            save_result(output,result,renderer,target_parameters)
            info['runs'].append(dict(method=method,optimizer=name,seconds=result['seconds'],
                                     iterations=result['iterations'],evaluations=result['evaluations'],
                                     best_loss=result['best_loss'],stop_reason=result['stop_reason'],
                                     result=str(output),warmup_seconds=warm_seconds))
            info['source_changed_during_run']=digest()!=source_hash
            (args.output/'summary.json').write_text(json.dumps(info,indent=2)+'\n')
            print(method,name,result['seconds'],result['stop_reason'],flush=True)
    if info['source_changed_during_run']:
        raise RuntimeError('Source changed while fitting; observations cannot be treated as matched.')

if __name__=='__main__':main()
