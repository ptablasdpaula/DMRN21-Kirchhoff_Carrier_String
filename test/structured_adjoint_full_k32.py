"""One bounded, directly measured four-second structured CeL VJP vs replay.

Compile/short warmup is excluded. Exactly one full-duration candidate evaluation
is timed (it is not a warmed multi-repetition median). Progress is persisted so
the timeout or scheduler termination cannot be mistaken for completion.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from benchmarks.exact_methods import BOX,GRID,filter_audio,metadata,theta_of,varied_parameters
from benchmarks.reconstruction import relative_error
from experiments.structured_adjoint.autograd import simulate as structured
from src.adjoint import simulate as replay
from src.optimization import criterion_for
from src.synthesis import Renderer


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,choices=(1,128),required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);device=torch.device('cuda');evidence=json.loads(a.evidence.read_text())
    base=metadata(); files=[ROOT/'experiments/structured_adjoint/structured.py',ROOT/'experiments/structured_adjoint/autograd.py',ROOT/'experiments/structured_adjoint/physical.py',ROOT/'experiments/parallel_segments/toy_scan.py',ROOT/'test/structured_adjoint.py',ROOT/'test/structured_adjoint_gpu.py']
    def signature():
        h=hashlib.sha256()
        for f in files:h.update(str(f.relative_to(ROOT)).encode()+f.read_bytes())
        return h.hexdigest()
    suite=signature(); own=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    short=next((x for x in evidence['cases'] if x.get('batch')==a.batch and x.get('chunk')==32 and x.get('steps')==128 and x.get('status')=='passed'),None)
    if short is None or evidence['metadata'].get('status')!='complete' or evidence['metadata'].get('source_changed') or evidence['metadata'].get('source_sha256')!=base['source_sha256'] or evidence['metadata'].get('structured_suite_sha256')!=suite:
        raise RuntimeError('Stale, missing or failed K32 short-window validation evidence')
    result={'metadata':{**base,'kind':'full-loss all-six K32 one sample after short GPU warmup','structured_suite_sha256':suite,'script_sha256':own,'evidence':str(a.evidence),'status':'running','stage':'setup'},'batch':a.batch,'duration':4.,'directions':6,'checkpoint':32,'window':4}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def save():a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    def stage(name):result['metadata']['stage']=name;save();print('STAGE',name,flush=True)
    save()
    try:
        values=varied_parameters(a.batch).to(device);duration=4.;steps=529200
        target=Renderer(GRID,duration).synthesize(BOX.defaults()).detach().to(device)
        model=Renderer(GRID,duration,device=device);criterion=criterion_for(target,GRID.sample_rate)
        stage('short_warmup')
        warm=values[:1].clone().detach().requires_grad_();audio=structured(theta_of(warm),GRID.intervals,GRID.integration_rate,128,'multimode',32,4)
        torch.autograd.grad(audio.square().sum(),warm);del warm,audio;torch.cuda.synchronize()
        def evaluate(method):
            x=values.detach().clone().requires_grad_();theta=theta_of(x)
            raw=(structured(theta,GRID.intervals,GRID.integration_rate,steps,'multimode',32,4) if method=='structured' else replay(theta,GRID.intervals,GRID.integration_rate,steps,'multimode',32))
            audio=filter_audio(raw,model);losses=torch.stack([criterion(row) for row in audio])
            grad,=torch.autograd.grad(losses.sum(),x)
            return audio.detach(),losses.detach(),grad.detach()
        stage('reference_running');t=time.perf_counter();ref=evaluate('replay');torch.cuda.synchronize();result['reference_seconds']=time.perf_counter()-t
        ref=tuple(v.cpu() for v in ref);stage('candidate_running')
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter();candidate=evaluate('structured');torch.cuda.synchronize()
        result['candidate_seconds']=time.perf_counter()-t;result['peak_cuda_bytes']=torch.cuda.max_memory_allocated();candidate=tuple(v.cpu() for v in candidate);stage('validating')
        val=values.detach().cpu()
        result['errors']=dict(audio_relative_l2=relative_error(candidate[0],ref[0]),loss_relative_l2=relative_error(candidate[1],ref[1]),scaled_gradient_relative_l2=relative_error(val*candidate[2],val*ref[2]))
        torch.testing.assert_close(candidate[0],ref[0],rtol=1e-6,atol=1e-10)
        torch.testing.assert_close(candidate[1],ref[1],rtol=5e-4,atol=1e-10)
        torch.testing.assert_close(val*candidate[2],val*ref[2],rtol=5e-4,atol=1e-8)
        result['metadata']['source_changed']=(signature()!=suite or metadata()['source_sha256']!=base['source_sha256'] or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=own)
        if result['metadata']['source_changed']:raise RuntimeError('Source changed while measuring')
        result['metadata']['status']='complete';result['metadata']['stage']='done';save();print(json.dumps(result,allow_nan=False),flush=True)
    except BaseException as exc:
        result['metadata']['status']='failed';result['metadata']['error']=str(exc);save();raise
if __name__=='__main__':main()
