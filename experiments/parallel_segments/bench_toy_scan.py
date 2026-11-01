"""Small correctness-first CPU/CUDA segment-map scaling experiment.

This uses eager PyTorch, not a fused CUDA kernel; absolute timings do not
predict optimized-kernel performance. CPU and GPU graphs are the same.
"""
import argparse,hashlib,json,os,platform,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from experiments.parallel_segments.toy_scan import make_problem,forward,segment_scan,dense_storage_bytes

def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',default='cuda');parser.add_argument('--repeats',type=int,default=3);args=parser.parse_args()
    device=torch.device(args.device);torch.set_num_threads(1)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    base=ROOT/'experiments/parallel_segments'
    def digest():
        h=hashlib.sha256()
        for name in ('toy_scan.py','test_toy_scan.py','bench_toy_scan.py'):
            file=base/name;h.update(file.name.encode()+file.read_bytes())
        return h.hexdigest()
    result={'metadata':{'source_sha256':digest(),'host':platform.node(),'torch':torch.__version__,'job':os.environ.get('SLURM_JOB_ID'),'device':str(device),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,'dtype':'float64','note':'eager PyTorch toy, not an optimized CUDA kernel; trajectory checkpoint generation excluded'},'rows':[],'full_grid_memory':{}}
    for batch in (1,8,128):
        for chunk in (64,256,1024):
            result['full_grid_memory'][f'b{batch}_chunk{chunk}']=dense_storage_bytes(176,batch,529200,chunk)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    save()
    for n in (4,16,32,64):
        for batch in (1,4):
            steps=256
            for chunk in (8,32,128):
                problem=make_problem(n,batch,steps,device=str(device));theta=torch.zeros(batch,8,device=device,dtype=torch.float64)
                p=problem['p0'].detach().clone().requires_grad_();u=problem['u0'].detach().clone().requires_grad_();thetag=theta.clone().requires_grad_();problem['p0'],problem['u0']=p,u
                audio,checks=forward(problem,thetag,chunk)
                expected=torch.autograd.grad((audio*problem['upstream']).sum(),(p,u,thetag))
                with torch.no_grad():
                    gi,gt,diag=segment_scan(problem,theta,checks.detach(),chunk,True)
                torch.testing.assert_close(gi[:,:n],expected[0],rtol=2e-8,atol=1e-10)
                torch.testing.assert_close(gi[:,n:],expected[1],rtol=2e-8,atol=1e-10)
                torch.testing.assert_close(gt,expected[2],rtol=2e-8,atol=1e-10)
                del audio,expected,gi,gt
                if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
                times=[]
                for _ in range(args.repeats):
                    sync(device);started=time.perf_counter()
                    with torch.no_grad():segment_scan(problem,theta,checks.detach(),chunk)
                    sync(device);times.append(time.perf_counter()-started)
                row=dict(modes=n,batch=batch,steps=steps,chunk=chunk,segments=steps//chunk,
                         median_seconds=statistics.median(times),repetitions=times,
                         peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,**diag)
                result['rows'].append(row);save();print(json.dumps(row),flush=True)
    result['metadata']['source_changed_during_run']=digest()!=result['metadata']['source_sha256']
    if result['metadata']['source_changed_during_run']:
        save();raise RuntimeError('Toy source changed during the run.')
    save()
    print('TOY SEGMENT SCAN COMPLETE',flush=True)
if __name__=='__main__':main()
