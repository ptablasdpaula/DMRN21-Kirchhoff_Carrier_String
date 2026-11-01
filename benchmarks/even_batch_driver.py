"""Run bounded per-method/batch measurements, stopping each method on CUDA OOM.

Each child is the existing validated benchmark harness. Separate processes free
all GPU allocations between cases and make a large-batch OOM recoverable. Times
are measured after warmup inside each child, not across process startup.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--duration',required=True,choices=['.25','4'])
    p.add_argument('--start',type=int,required=True)
    p.add_argument('--end',type=int,required=True)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--validation-short',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.start<2 or a.end>128 or a.start>a.end or a.start%2 or a.end%2 or a.repeats<1:
        p.error('Choose an even range within 2..128 and positive repeats.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    batches=[1,*range(a.start,a.end+1,2)]
    result={'duration':float(a.duration),'range':[a.start,a.end],
            'job':os.environ.get('SLURM_JOB_ID'),'status':'running','children':[],
            'rows':[],'method_stopped_at_oom':{}}
    def save(): a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    save()
    for method in ('cuda-modal','cuda-seeded','cuda-recompute','cuda-adjoint','cuda-register'):
        kinds='synthesis' if method=='cuda-register' else 'synthesis,loss_backward'
        for batch in batches:
            output=a.output.with_name(a.output.stem+f'-{method}-b{batch}.json')
            cmd=[sys.executable,str(ROOT/'benchmarks/exact_methods.py'),
                 '--methods',method,'--batches',str(batch),'--durations',a.duration,
                 '--kinds',kinds,'--repeats',str(a.repeats),'--validation',str(a.validation),
                 '--validation-short',str(a.validation_short),'--output',str(output)]
            print('RUN',method,'batch',batch,'duration',a.duration,flush=True)
            completed=subprocess.run(cmd,cwd=ROOT)
            child=json.loads(output.read_text()) if output.exists() else None
            result['children'].append(dict(method=method,batch=batch,exit_code=completed.returncode,path=str(output)))
            if child:
                if 'metadata' not in result:result['metadata']=child['metadata']
                result['rows'].extend(child['rows'])
            if completed.returncode:
                result['status']='failed';result['failure']=dict(method=method,batch=batch,exit_code=completed.returncode)
                save();raise SystemExit(completed.returncode)
            oom=child and any('out of memory' in r.get('error','').lower() for r in child['rows'])
            if oom:
                result['method_stopped_at_oom'][method]=batch;save()
                print('Stopping larger batches for',method,'after CUDA OOM at',batch,flush=True)
                break
            save()
    result['status']='complete';save()
    print('EVEN BATCH SWEEP COMPLETE',flush=True)
if __name__=='__main__':main()
