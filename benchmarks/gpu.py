"""Reproducible matched CPU/CUDA latency, batch throughput and launch tuning."""
import argparse,hashlib,json,os,platform,statistics,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.kc_string_forward import ForwardString,nominal_parameters, PARAMETER_COUNT
from src.accelerated import evaluate
from src.parameters import ParameterBox
from src.grid import GridConfig
from src.synthesis import Renderer
from src.optimization import criterion_for

def sync(device):
    if device=='cuda':torch.cuda.synchronize()

def measure(fn,device,repeats):
    fn();sync(device);times=[];events=[]
    for _ in range(repeats):
        sync(device)
        if device=='cuda':
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
        t=time.perf_counter();fn()
        if device=='cuda':end.record()
        sync(device);times.append(time.perf_counter()-t)
        if device=='cuda':events.append(start.elapsed_time(end)/1000)
    return dict(median_seconds=statistics.median(times),min_seconds=min(times),max_seconds=max(times),
                repetitions=times,cuda_event_seconds=events)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--devices',default='cpu,cuda',help='Comma-separated CPU/CUDA measurement devices')
    parser.add_argument('--repeats',type=int,default=5);parser.add_argument('--quick',action='store_true')
    args=parser.parse_args();args.devices=args.devices.split(',')
    if not args.devices or any(d not in ('cpu','cuda') for d in args.devices): parser.error('Use cpu,cuda or one of those devices')
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    source_hash=hashlib.sha256()
    for path in sorted(Path('src').glob('*')):
        if path.suffix in ('.py','.cpp','.cu','.h'):source_hash.update(path.name.encode()+path.read_bytes())
    result=dict(metadata=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
                cpu=platform.processor(),host=platform.node(),job=os.environ.get('SLURM_JOB_ID'),
                revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                source_sha256=source_hash.hexdigest(),dtype='float64',cpu_threads=1,source_status=subprocess.check_output(['git','status','--short'],text=True)),rows=[])
    def save(row):
        result['rows'].append(row);args.output.write_text(json.dumps(result,indent=2));print(json.dumps(row),flush=True)
    durations=[.06,.25] if args.quick else [.06,.25,1.,4.]
    configurations=[(128,44100),(176,132300)]
    for N,rate in configurations:
        for device in args.devices:
            for backend in (('native','spatial','modal') if device=='cpu' else ('spatial','modal')):
                theta=torch.ones(PARAMETER_COUNT,dtype=torch.float64,device=device,requires_grad=True)
                model=ForwardString(N,rate,excitation='multimode',backend=backend)
                sync(device);t=time.perf_counter();model(theta,32/rate).sum().backward();sync(device)
                save(dict(kind='first_use',N=N,rate=rate,device=device,backend=backend,seconds=time.perf_counter()-t))
                for duration in durations:
                    with torch.no_grad():
                        row=measure(lambda:model(theta,duration),device,args.repeats)
                    save(dict(kind='synthesis',N=N,rate=rate,device=device,backend=backend,duration=duration,**row))
                    if device=='cuda':torch.cuda.reset_peak_memory_stats()
                    row=measure(lambda:model(theta,duration),device,args.repeats)
                    save(dict(kind='forward_tangents',N=N,rate=rate,device=device,backend=backend,duration=duration,
                              peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,**row))
                    a,j=model.output_and_tangents(theta,duration);g=torch.ones_like(a)
                    save(dict(kind='contraction',N=N,rate=rate,device=device,backend=backend,duration=duration,
                              **measure(lambda:g@j,device,args.repeats)))
    box=ParameterBox();grid=GridConfig(box)
    for device in args.devices:
        for backend in (('native','spatial','modal') if device=='cpu' else ('spatial','modal')):
            p=box.defaults().to(device).requires_grad_();renderer=Renderer(grid,.25 if args.quick else 4.,device=device,backend=backend)
            target=renderer.synthesize(box.perturb(p.detach()));loss=criterion_for(target,grid.sample_rate)
            def closure():
                p.grad=None;value=loss(renderer(p));value.backward();return value
            save(dict(kind='full_evaluation',device=device,backend=backend,duration=renderer.duration,
                      **measure(closure,device,args.repeats)))
            raw=renderer.model((box.to_material(p.detach()) / nominal_parameters(p.device)),renderer.duration).detach().requires_grad_()
            def filter_loss():
                raw.grad=None;audio=raw
                if renderer.filter is not None:
                    padded=torch.nn.functional.pad(audio.reshape(1,1,-1),(renderer.half_width,renderer.half_width),mode='replicate')
                    audio=torch.nn.functional.conv1d(padded,renderer.filter,stride=grid.oversampling).flatten()
                value=loss(audio[:renderer.samples]);value.backward()
            save(dict(kind='filter_loss_backward',device=device,backend=backend,duration=renderer.duration,
                      **measure(filter_loss,device,args.repeats)))
    if 'cuda' in args.devices:
        from src.accelerated import kernel
        for backend in ('spatial','modal'):
            for threads in (128,256):
                values=kernel(True).resources(backend=='modal',175,threads)
                save(dict(kind='kernel_resources',backend=backend,threads=threads,
                    storage=['register','shared','global','channel_parallel'][values[0]],launched_threads=288 if values[0]==3 else threads,registers_per_thread=values[1],
                    local_bytes_per_thread=values[2],shared_bytes_per_block=values[3],max_threads=values[4]))
        # Tune on identical default-grid single-item differentiable work.
        for backend in ('spatial','modal'):
            theta=torch.ones(PARAMETER_COUNT,dtype=torch.float64,device='cuda',requires_grad=True)
            for threads in (128,256):
                for chunk in (1024,8192,0):
                    model=ForwardString(176,132300,excitation='multimode',backend=backend,threads=threads,chunk_steps=chunk)
                    save(dict(kind='tuning',device='cuda',backend=backend,threads=threads,chunk_steps=chunk,duration=.25,
                              **measure(lambda:model(theta,.25),'cuda',args.repeats)))
    for backend in ('spatial','modal'):
        for batch in ([1,4] if args.quick else [1,2,4,8,16,32,64]):
            for device in args.devices:
                theta=torch.ones(batch,PARAMETER_COUNT,dtype=torch.float64,device=device)
                # The same evaluator includes initialization and internal-rate tangent storage.
                fn=lambda:evaluate(theta,176,132300,round(.25*132300),1,'multimode',backend,True)
                try:
                    if device=='cuda':torch.cuda.reset_peak_memory_stats()
                    row=measure(fn,device,args.repeats)
                    save(dict(kind='batch',device=device,backend=backend,batch=batch,duration=.25,
                              items_per_second=batch/row['median_seconds'],
                              peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,**row))
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache();save(dict(kind='batch',device=device,backend=backend,batch=batch,error='out of memory'));break
    tuning=[r for r in result['rows'] if r['kind']=='tuning']
    result['recommended_launch']={backend:{key:row[key] for key in ('threads','chunk_steps','median_seconds')}
        for backend in ('spatial','modal') if any(r['backend']==backend for r in tuning)
        for row in [min((r for r in tuning if r['backend']==backend),key=lambda r:r['median_seconds'])]}
    args.output.write_text(json.dumps(result,indent=2))
    print('BENCHMARK COMPLETE',flush=True)
if __name__=='__main__':main()
