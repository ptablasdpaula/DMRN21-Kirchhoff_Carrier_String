"""Exact segment-scan scaling at 175 toy modes; *not* a production speed test.

The full-rank dense affine maps may consume substantial GPU memory. Upstream
loss is padded with zero weights when steps aren't divisible by the chunk size,
which leaves the original loss and gradients unchanged.
"""
import argparse,gc,hashlib,json,os,platform,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from experiments.parallel_segments.toy_scan import make_problem,forward,segment_scan,dense_storage_bytes

def digest():
    h=hashlib.sha256()
    for p in (Path(__file__),ROOT/'experiments/parallel_segments/toy_scan.py'):
        h.update(p.name.encode()+p.read_bytes())
    return h.hexdigest()

def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)

def measure(fn,device,repeats):
    sync(device);t=time.perf_counter();fn();sync(device);warm=time.perf_counter()-t
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    elapsed=[]
    for _ in range(repeats):
        sync(device);t=time.perf_counter();fn();sync(device);elapsed.append(time.perf_counter()-t)
    return dict(warmup_seconds=warm,median_seconds=statistics.median(elapsed),repetitions_seconds=elapsed,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)

def baseline(problem,device):
    # Fresh leaf tensors and computation graph on every invocation.
    data=dict(problem);p=problem['p0'].detach().clone().requires_grad_();u=problem['u0'].detach().clone().requires_grad_()
    theta=torch.zeros(p.shape[0],8,device=device,dtype=torch.float64,requires_grad=True)
    data['p0'],data['u0']=p,u
    audio,_=forward(data,theta,data['upstream'].shape[-1])
    return torch.autograd.grad((audio*data['upstream']).sum(),(p,u,theta))

def candidate(problem,device,chunk,return_debug=False):
    theta=torch.zeros(problem['p0'].shape[0],8,device=device,dtype=torch.float64)
    with torch.no_grad():
        _,checkpoints=forward(problem,theta,chunk)
        return segment_scan(problem,theta,checkpoints,chunk,return_debug)

def scan_only(problem,device,chunk,checkpoints):
    theta=torch.zeros(problem['p0'].shape[0],8,device=device,dtype=torch.float64)
    with torch.no_grad():return segment_scan(problem,theta,checkpoints,chunk)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--modes',type=int,default=175);p.add_argument('--batch',type=int,default=1)
    p.add_argument('--steps',default='256,1024,4096');p.add_argument('--chunks',default='10,64,256')
    p.add_argument('--repeats',type=int,default=2);args=p.parse_args();device=torch.device(args.device)
    if args.modes<1 or args.batch<1 or args.repeats<1:p.error('modes, batch and repeats must be positive')
    steps_list=[int(x) for x in args.steps.split(',')];chunks=[int(x) for x in args.chunks.split(',')]
    if any(x<1 for x in steps_list+chunks):p.error('steps and chunks must be positive')
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(1);initial_hash=digest()
    result=dict(metadata=dict(source_sha256=initial_hash,torch=torch.__version__,device=str(device),
           gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,host=platform.node(),
           job=os.environ.get('SLURM_JOB_ID'),dtype='float64',modes=args.modes,batch=args.batch,
           note='Synthetic stable coefficients, eager PyTorch (not fused), arbitrary upstream VJP; no audio filter or CeL. Do not compare absolute runtime with production kernels.'),rows=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    def record(row):result['rows'].append(row);save();print(json.dumps(row),flush=True)
    save()
    for steps in steps_list:
        problem=make_problem(args.modes,args.batch,steps,device=str(device))
        try:
            reference=baseline(problem,device);sync(device)
            timings=measure(lambda:baseline(problem,device),device,args.repeats)
            record(dict(kind='sequential_autograd',steps=steps,status='passed',**timings))
        except torch.cuda.OutOfMemoryError:
            gc.collect();torch.cuda.empty_cache();record(dict(kind='sequential_autograd',steps=steps,status='out_of_memory'))
            for chunk in chunks:record(dict(kind='segment_scan',steps=steps,chunk=chunk,status='skipped_reference_oom'))
            continue
        for chunk in chunks:
            data=dict(problem);padding=(-steps)%chunk
            data['upstream']=F.pad(problem['upstream'],(0,padding)) if padding else problem['upstream']
            try:
                gi,gt,diag=candidate(data,device,chunk,True)
                torch.testing.assert_close(gi[:,:args.modes],reference[0],rtol=1e-6,atol=1e-10)
                torch.testing.assert_close(gi[:,args.modes:],reference[1],rtol=1e-6,atol=1e-10)
                torch.testing.assert_close(gt,reference[2],rtol=1e-6,atol=1e-10)
                del gi,gt
                with torch.no_grad():
                    theta=torch.zeros(args.batch,8,device=device,dtype=torch.float64)
                    _,checks=forward(data,theta,chunk)
                scantime=measure(lambda:scan_only(data,device,chunk,checks),device,args.repeats)
                del checks
                fulltime=measure(lambda:candidate(data,device,chunk),device,args.repeats)
                record(dict(kind='segment_scan',steps=steps,padded_steps=steps+padding,chunk=chunk,status='passed',
                     relative_boundary_error=diag['relative_boundary_error'],segment_map_bytes=diag['segment_map_bytes'],
                     parameter_map_bytes=diag['parameter_map_bytes'],scan_only=scantime,forward_and_scan=fulltime,
                     projected_full_4s_map_bytes=dense_storage_bytes(args.modes+1,args.batch,529200,chunk)))
            except torch.cuda.OutOfMemoryError:
                gc.collect();torch.cuda.empty_cache();record(dict(kind='segment_scan',steps=steps,chunk=chunk,status='out_of_memory'))
            except (ValueError,RuntimeError,AssertionError) as error:
                record(dict(kind='segment_scan',steps=steps,chunk=chunk,status='failed',error=str(error)))
        del reference,problem;gc.collect()
    result['metadata']['source_changed_during_run']=digest()!=initial_hash
    if result['metadata']['source_changed_during_run']:save();raise RuntimeError('Source changed during scaling benchmark')
    save();print('PARALLEL SCAN SCALE COMPLETE',flush=True)

if __name__=='__main__':main()
