"""Profile matched grouped-loss workloads without changing benchmark timings."""
import argparse, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from torch.profiler import profile,ProfilerActivity,record_function
from benchmarks.cache_reconstruction_compare import render
from benchmarks.exact_methods import BOX,GRID,varied_parameters,metadata
from experiments.loss_batch.grouped import grouped_loss_backward
from src.optimization import criterion_for
from src.synthesis import Renderer

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--batches',default='1,128');p.add_argument('--duration',type=float,default=4);p.add_argument('--group',type=int,default=16);a=p.parse_args()
    torch.set_num_threads(1);device=torch.device('cuda');model=Renderer(GRID,a.duration,device=device)
    target=Renderer(GRID,a.duration).synthesize(BOX.defaults()).detach().to(device);criterion=criterion_for(target,GRID.sample_rate)
    result={'metadata':metadata(),'profiles':[]};a.output.parent.mkdir(parents=True,exist_ok=True)
    for batch in map(int,a.batches.split(',')):
      values=varied_parameters(batch).to(device)
      for method in ('seeded','cached'):
       def call():
        x=values.detach().clone().requires_grad_()
        with record_function('string_synthesis_and_filter'):
            audio=render(x,model,method,256)
        with record_function('grouped_loss_and_simulator_backward'):
            grouped_loss_backward(audio,criterion,a.group)
       call();torch.cuda.synchronize()
       with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],profile_memory=True) as prof:
        t=time.perf_counter();call();torch.cuda.synchronize();elapsed=time.perf_counter()-t
       rows=[]
       for ev in prof.key_averages():
        rows.append({'name':ev.key,'count':ev.count,'self_cpu_us':ev.self_cpu_time_total,
                     'self_device_us':getattr(ev,'self_device_time_total',0),
                     'device_memory_bytes':getattr(ev,'device_memory_usage',0)})
       rows.sort(key=lambda x:x['self_device_us'],reverse=True)
       record={'batch':batch,'method':method,'group':a.group,'profiled_seconds':elapsed,'events':rows[:50]}
       result['profiles'].append(record);a.output.write_text(json.dumps(result,indent=2)+'\n')
       print(json.dumps({'batch':batch,'method':method,'profiled_seconds':elapsed,'top':rows[:8]}),flush=True)
if __name__=='__main__':main()
