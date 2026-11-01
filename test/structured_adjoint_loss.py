"""Full loss benchmark for structured adjoint, when short timing permits it."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from benchmarks.exact_methods import BOX,GRID,filter_audio,measure,metadata,theta_of,varied_parameters
from benchmarks.reconstruction import relative_error
from experiments.structured_adjoint.autograd import simulate as structured
from src.adjoint import simulate as replay
from src.optimization import criterion_for
from src.synthesis import Renderer


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-seconds',type=float,default=180.);p.add_argument('--repeats',type=int,default=2);a=p.parse_args()
    torch.set_num_threads(1);device=torch.device('cuda');evidence=json.loads(a.evidence.read_text());cases=evidence['cases']
    base_meta=metadata();suite_files=[ROOT/'experiments/structured_adjoint/structured.py',ROOT/'experiments/structured_adjoint/autograd.py',ROOT/'experiments/structured_adjoint/physical.py',ROOT/'experiments/parallel_segments/toy_scan.py',ROOT/'test/structured_adjoint.py',ROOT/'test/structured_adjoint_gpu.py']
    def suite_hash():
        h=hashlib.sha256()
        for file in suite_files:h.update(str(file.relative_to(ROOT)).encode()+file.read_bytes())
        return h.hexdigest()
    if evidence['metadata'].get('status')!='complete' or evidence['metadata'].get('source_changed') or evidence['metadata'].get('source_sha256')!=base_meta['source_sha256'] or evidence['metadata'].get('structured_suite_sha256')!=suite_hash():
        raise RuntimeError('Stale or failed structured short-window validation evidence')
    script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result={'metadata':{**base_meta,'short_evidence':str(a.evidence),'structured_suite_sha256':suite_hash(),'script_sha256':script_hash,'status':'running'},'checks':[],'rows':[]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def save():a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    save()
    for batch in (1,128):
        valid=[x for x in cases if x.get('batch')==batch and x.get('status')=='passed' and x.get('steps',0)<529200]
        if not valid:
            result['checks'].append(dict(batch=batch,status='skipped',reason='no successful short physical timing'));save();continue
        selected=min(valid,key=lambda x:x['estimated_four_second_backward']);estimate=selected['estimated_four_second_backward']
        if estimate>a.max_seconds:
            result['checks'].append(dict(batch=batch,status='skipped',reason='projected per-VJP runtime exceeds guard; full loss was not measured',estimated_seconds=estimate));save();continue
        chunk=selected['chunk'];window=selected['window'];duration=4.0;steps=529200
        target=Renderer(GRID,duration).synthesize(BOX.defaults()).detach().to(device)
        model=Renderer(GRID,duration,device=device);criterion=criterion_for(target,GRID.sample_rate)
        values=varied_parameters(batch).to(device)
        def evaluate(method,count):
            train=values[:,:count].detach().clone().requires_grad_()
            x=torch.cat((train,values[:,count:]),dim=-1);theta=theta_of(x)
            raw=(structured(theta,GRID.intervals,GRID.integration_rate,steps,'multimode',chunk,window)
                 if method=='structured' else replay(theta,GRID.intervals,GRID.integration_rate,steps,'multimode',chunk))
            audio=filter_audio(raw,model);losses=torch.stack([criterion(row) for row in audio])
            grad,=torch.autograd.grad(losses.sum(),train)
            return audio.detach(),losses.detach(),grad.detach()
        for count in (2,6):
            try:
                ref=evaluate('replay',count);candidate=evaluate('structured',count)
                error=dict(audio_relative_l2=relative_error(candidate[0],ref[0]),loss_relative_l2=relative_error(candidate[1],ref[1]),scaled_gradient_relative_l2=relative_error(values[:,:count]*candidate[2],values[:,:count]*ref[2]))
                torch.testing.assert_close(candidate[0],ref[0],rtol=1e-6,atol=1e-10)
                torch.testing.assert_close(candidate[1],ref[1],rtol=5e-4,atol=1e-10)
                torch.testing.assert_close(values[:,:count]*candidate[2],values[:,:count]*ref[2],rtol=5e-4,atol=1e-8)
                del ref,candidate
                result['checks'].append(dict(batch=batch,directions=count,chunk=chunk,window=window,status='passed',**error));save()
                for method in ('replay','structured'):
                    row=dict(batch=batch,directions=count,duration=duration,chunk=chunk,window=window,method=method,**measure(lambda:evaluate(method,count),device,a.repeats))
                    result['rows'].append(row);save();print(json.dumps(row),flush=True)
            except (RuntimeError,AssertionError,ValueError) as exc:
                result['checks'].append(dict(batch=batch,directions=count,chunk=chunk,window=window,status='failed',error=str(exc)));save();print('FAILED',str(exc),flush=True)
                if isinstance(exc,torch.cuda.OutOfMemoryError):torch.cuda.empty_cache()
    result['metadata']['source_changed']=(suite_hash()!=result['metadata']['structured_suite_sha256'] or metadata()['source_sha256']!=base_meta['source_sha256'] or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=script_hash)
    result['metadata']['status']='complete' if all(x['status'] in ('passed','skipped') for x in result['checks']) and not result['metadata']['source_changed'] else 'failed';save()
    if result['metadata']['status']!='complete':raise RuntimeError('Structured full-loss verification failed or source changed')
if __name__=='__main__':main()
