"""Measure normal and conservative whole-batch replay fallback separately."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from benchmarks.exact_methods import BOX,GRID,filter_audio,measure,metadata,theta_of,varied_parameters
from benchmarks.reconstruction import relative_error
from experiments.scalar_cache.fallback import simulate
from src.adjoint import simulate as replay
from src.optimization import criterion_for
from src.synthesis import Renderer

torch.set_num_threads(1)
output=Path(sys.argv[1]);output.parent.mkdir(parents=True,exist_ok=True)
result={'metadata':{**metadata(),'status':'running'},'checks':[],'rows':[]}
def save():output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
for batch in (1,128):
    duration=4.;values=varied_parameters(batch).cuda();target=Renderer(GRID,duration).synthesize(BOX.defaults()).detach().cuda()
    renderer=Renderer(GRID,duration,device='cuda');criterion=criterion_for(target,GRID.sample_rate)
    def run(method):
        x=values.detach().clone().requires_grad_();theta=theta_of(x);steps=round(duration*GRID.integration_rate)
        if method=='replay': raw=replay(theta,GRID.intervals,GRID.integration_rate,steps,'multimode',256)
        else: raw=simulate(theta,GRID.intervals,GRID.integration_rate,steps,'multimode',256,3,method=='forced_fallback')
        audio=filter_audio(raw,renderer);loss=torch.stack([criterion(row) for row in audio]).sum();loss.backward()
        return audio.detach(),loss.detach(),x.grad.detach()
    ref=run('replay')
    for method in ('normal','forced_fallback'):
        out=run(method)
        torch.testing.assert_close(out[0],ref[0],rtol=1e-6,atol=1e-10);torch.testing.assert_close(out[1],ref[1],rtol=5e-4,atol=1e-10)
        torch.testing.assert_close(values*out[2],values*ref[2],rtol=5e-4,atol=1e-8)
        check=dict(batch=batch,method=method,status='passed',audio_relative_l2=relative_error(out[0],ref[0]),
                   loss_relative_l2=relative_error(out[1],ref[1]),scaled_gradient_relative_l2=relative_error(values*out[2],values*ref[2]))
        result['checks'].append(check);del out
    del ref;save()
    for method in ('replay','normal','forced_fallback'):
        row=dict(batch=batch,duration=duration,method=method,**measure(lambda:run(method),values.device,3))
        result['rows'].append(row);save();print(json.dumps(row),flush=True)
result['metadata']['status']='complete';save()
