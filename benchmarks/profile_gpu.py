"""Profile a warmed end-to-end evaluation without changing its loss or grid."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.parameters import ParameterBox
from src.grid import GridConfig
from src.synthesis import Renderer
from src.optimization import criterion_for
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=True)
for backend in ('spatial','modal'):
    box=ParameterBox();renderer=Renderer(GridConfig(box),duration=.25,device='cuda',backend=backend)
    x=box.defaults().cuda().requires_grad_();target=renderer.synthesize(box.perturb(x.detach()))
    criterion=criterion_for(target,44100)
    def step():
        x.grad=None;loss=criterion(renderer(x));loss.backward()
    step();torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True,profile_memory=True) as profile:
        step();torch.cuda.synchronize()
    profile.export_chrome_trace(str(args.output/f'{backend}.json'))
    (args.output/f'{backend}.txt').write_text(profile.key_averages().table(sort_by='self_device_time_total',row_limit=35))
    print(backend,profile.key_averages().table(sort_by='self_device_time_total',row_limit=8),flush=True)
