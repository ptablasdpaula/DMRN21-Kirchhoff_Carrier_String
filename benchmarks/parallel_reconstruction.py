"""Matched replay/sequential/parallel-state-reconstruction benchmark.

The candidate parallelises checkpoint-segment state reconstruction, not the
boundary-gradient propagation. Per-case numeric guards run before timing.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from benchmarks.exact_methods import BOX, GRID, digest, filter_audio, measure, metadata, theta_of, varied_parameters
from benchmarks.reconstruction import reconstruction_digest, relative_error
from src.optimization import criterion_for
from src.synthesis import Renderer


def parallel_digest():
    h=hashlib.sha256()
    for name in ('parallel_reconstruction.cu','parallel_reconstruction.py'):
        path=ROOT/'experiments'/'parallel_segments'/name
        h.update(name.encode()+path.read_bytes())
    return h.hexdigest()


def render(values,renderer,method,checkpoint,group):
    if method=='replay':
        from src.adjoint import simulate
        extra=()
    elif method=='sequential':
        from experiments.reconstruction.reconstruct import simulate
        extra=()
    elif method=='parallel':
        from experiments.parallel_segments.parallel_reconstruction import simulate
        extra=(group,)
    else:
        raise ValueError(method)
    raw=simulate(theta_of(values),GRID.intervals,GRID.integration_rate,
                 round(renderer.duration*GRID.integration_rate),'multimode',checkpoint,*extra)
    return filter_audio(raw,renderer)


def check_loss(values,renderer,criterion,method,checkpoint,group):
    p=values.detach().clone().requires_grad_()
    audio=render(p,renderer,method,checkpoint,group)
    losses=torch.stack([criterion(row) for row in audio])
    (gradient,)=torch.autograd.grad(losses.sum(),p)
    return audio.detach(),losses.detach(),gradient.detach()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--validation',type=Path,required=True)
    parser.add_argument('--batches',default='1,128')
    parser.add_argument('--durations',default='4')
    parser.add_argument('--checkpoints',default='10,64,256')
    parser.add_argument('--groups',default='1,16,64')
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    batches=[int(v) for v in args.batches.split(',')]
    checkpoints=[int(v) for v in args.checkpoints.split(',')]
    groups=[int(v) for v in args.groups.split(',')]
    durations=[float(v) for v in args.durations.split(',')]
    if args.repeats<1 or any(v<1 for v in batches+checkpoints+groups) or any(not math.isfinite(v) or v<=0 for v in durations):
        parser.error('All dimensions, durations and repetitions must be positive.')
    torch.set_num_threads(1)
    if not torch.cuda.is_available(): raise RuntimeError('GPU allocation required')
    source_hash=digest();reconstruction_hash=reconstruction_digest();parallel_hash=parallel_digest()
    test_hash=hashlib.sha256((ROOT/'test'/'parallel_reconstruction.py').read_bytes()).hexdigest()
    own_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    helper_hash=hashlib.sha256((ROOT/'benchmarks'/'exact_methods.py').read_bytes()).hexdigest()
    worker_hash=hashlib.sha256((ROOT/'hpc'/'parallel-reconstruction-benchmark.sh').read_bytes()).hexdigest()
    evidence=json.loads(args.validation.read_text())
    if (evidence.get('status')!='passed' or evidence.get('source_sha256')!=source_hash
       or evidence.get('reconstruction_sha256')!=reconstruction_hash or evidence.get('parallel_sha256')!=parallel_hash
       or evidence.get('test_sha256')!=test_hash or evidence.get('source_changed_during_run')):
        raise RuntimeError('Validation evidence is failed, stale, or from a different source snapshot.')
    result={'metadata':{**metadata(),'reconstruction_sha256':reconstruction_hash,'parallel_sha256':parallel_hash,
      'test_sha256':test_hash,'benchmark_sha256':own_hash,'harness_sha256':helper_hash,'worker_sha256':worker_hash,
      'validation':str(args.validation),'validation_evidence':evidence,'status':'running'},'checks':[],'rows':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        result['metadata']['source_changed_during_run']=(
          digest()!=source_hash or reconstruction_digest()!=reconstruction_hash or parallel_digest()!=parallel_hash
          or hashlib.sha256((ROOT/'test'/'parallel_reconstruction.py').read_bytes()).hexdigest()!=test_hash
          or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=own_hash
          or hashlib.sha256((ROOT/'benchmarks'/'exact_methods.py').read_bytes()).hexdigest()!=helper_hash
          or hashlib.sha256((ROOT/'hpc'/'parallel-reconstruction-benchmark.sh').read_bytes()).hexdigest()!=worker_hash)
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    def emit(row):
        result['rows'].append(row);save();print(json.dumps(row,allow_nan=False),flush=True)
    save()
    device=torch.device('cuda',torch.cuda.current_device())
    for duration in durations:
        target=Renderer(GRID,duration).synthesize(BOX.defaults()).detach().to(device)
        renderer=Renderer(GRID,duration,device=device)
        criterion=criterion_for(target,GRID.sample_rate)
        for checkpoint in checkpoints:
            for batch in batches:
                values=varied_parameters(batch).to(device)
                for group in groups:
                    audio_ref=loss_ref=grad_ref=audio=losses=gradient=None
                    passed=False
                    try:
                        audio_ref,loss_ref,grad_ref=check_loss(values,renderer,criterion,'replay',checkpoint,group)
                        audio,losses,gradient=check_loss(values,renderer,criterion,'parallel',checkpoint,group)
                        errors=dict(waveform_relative_l2=relative_error(audio,audio_ref),
                          loss_relative_l2=relative_error(losses,loss_ref),
                          scaled_gradient_relative_l2=relative_error(values*gradient,values*grad_ref))
                        torch.testing.assert_close(audio,audio_ref,rtol=1e-6,atol=1e-10)
                        torch.testing.assert_close(losses,loss_ref,rtol=5e-4,atol=1e-10)
                        torch.testing.assert_close(values*gradient,values*grad_ref,rtol=5e-4,atol=1e-8)
                        passed=True
                        result['checks'].append(dict(status='passed',batch=batch,duration=duration,checkpoint=checkpoint,group=group,**errors))
                    except (AssertionError,RuntimeError) as error:
                        result['checks'].append(dict(status='failed',batch=batch,duration=duration,checkpoint=checkpoint,group=group,error=str(error)))
                        print('CHECK FAILED',batch,duration,checkpoint,group,str(error),flush=True)
                    finally:
                        del audio_ref,loss_ref,grad_ref,audio,losses,gradient
                        if not passed:torch.cuda.empty_cache()
                    save()
                    for method in (('replay','sequential','parallel') if passed else ('replay','sequential')):
                        pval=values.detach().clone().requires_grad_()
                        probe=torch.linspace(-1,1,renderer.samples,dtype=torch.float64,device=device)
                        for kind in ('synthesis','vjp','loss_backward'):
                            if kind=='synthesis':
                                def fn():
                                    with torch.no_grad():return render(values,renderer,method,checkpoint,group)
                            elif kind=='vjp':
                                def fn():
                                    pval.grad=None
                                    (render(pval,renderer,method,checkpoint,group)*probe).sum().backward()
                            else:
                                def fn():
                                    pval.grad=None
                                    audio=render(pval,renderer,method,checkpoint,group)
                                    torch.stack([criterion(row) for row in audio]).sum().backward()
                            try:
                                timing=measure(fn,device,args.repeats)
                                emit(dict(method=method,kind=kind,batch=batch,duration=duration,checkpoint=checkpoint,
                                          group=group,items_per_second=batch/timing['median_seconds'],**timing))
                            except RuntimeError as error:
                                if isinstance(error,torch.cuda.OutOfMemoryError):torch.cuda.empty_cache()
                                emit(dict(method=method,kind=kind,batch=batch,duration=duration,checkpoint=checkpoint,
                                          group=group,error=str(error)))
    failed=any(v['status']!='passed' for v in result['checks']) or any('error' in v for v in result['rows'])
    result['metadata']['status']='complete_with_failures' if failed else 'complete';save()
    if result['metadata']['source_changed_during_run']:raise RuntimeError('Source changed during benchmark.')
    print('PARALLEL-RECONSTRUCTION BENCHMARK COMPLETE',flush=True)

if __name__=='__main__':main()
