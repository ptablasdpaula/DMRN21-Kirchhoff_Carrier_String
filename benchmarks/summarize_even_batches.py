"""Summarize and plot incremental even-batch sweep JSON without mixing builds.

Pass aggregate sweep files and/or the quick JSON explicitly. Results of child
files are already included in aggregates and are rejected to prevent duplicates.
No data is merged across GPU model, host, or source SHA-256.
"""
import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

EXPECTED={1,*range(2,129,2)}
METHODS=('cuda-modal','cuda-seeded','cuda-recompute','cuda-adjoint','cuda-register')
KINDS={'cuda-register':('synthesis',),'cuda-modal':('synthesis','loss_backward'),
       'cuda-seeded':('synthesis','loss_backward'),'cuda-recompute':('synthesis','loss_backward'),
       'cuda-adjoint':('synthesis','loss_backward')}

def slug(s): return re.sub(r'[^a-zA-Z0-9._-]+','-',str(s)).strip('-')[:60]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs',nargs='+',type=Path,help='Aggregate even-sweep or quick JSON files')
    p.add_argument('--output-dir',required=True,type=Path)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    inputs=[];records=[];rejected=[];metadata_by_key={}
    for path in args.inputs:
        try: data=json.loads(path.read_text())
        except (OSError,json.JSONDecodeError) as e:
            rejected.append({'path':str(path),'reason':str(e)});continue
        metadata=data.get('metadata',{})
        source=metadata.get('source_sha256')
        if not source or not metadata.get('gpu') or not metadata.get('host'):
            rejected.append({'path':str(path),'reason':'missing GPU, host, or source hash'});continue
        if '-cuda-' in path.stem and 'children' not in data:
            rejected.append({'path':str(path),'reason':'child result duplicates aggregate'});continue
        if metadata.get('source_changed_during_run'):
            rejected.append({'path':str(path),'reason':'source changed during run'});continue
        status=data.get('status','complete' if 'source_changed_during_run' in metadata else 'partial')
        # Child benchmark explicitly verifies source on exit; a running aggregate
        # is preserved as provisional until all its children finish.
        stable=('source_changed_during_run' in metadata and not metadata['source_changed_during_run'])
        if status=='running': status='running_provisional'
        elif status=='complete' and not stable: status='complete_source_not_verified'
        meta_key=(metadata['gpu'],metadata['host'],source)
        metadata_by_key[meta_key]=dict(gpu=metadata['gpu'],host=metadata['host'],source_sha256=source)
        inputs.append({'path':str(path),'status':status,**metadata_by_key[meta_key],
                       'method_stopped_at_oom':data.get('method_stopped_at_oom',{}),
                       'failure':data.get('failure')})
        for row in data.get('rows',[]):
            if row.get('method') not in METHODS: continue
            records.append(dict(gpu=metadata['gpu'],host=metadata['host'],source_sha256=source,
                                job=metadata.get('job') or data.get('job'),status=status,path=str(path),**row))
    fields=('gpu','host','source_sha256','job','status','duration','method','kind','batch',
            'median_seconds','items_per_second','peak_cuda_bytes','error','path')
    with (args.output_dir/'timings.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    coverage=defaultdict(lambda:{'success':set(),'oom':set()})
    points=defaultdict(lambda:defaultdict(list))
    for r in records:
        if r.get('kind') not in ('synthesis','loss_backward') or r.get('batch') is None:continue
        key=(r['gpu'],r['host'],r['source_sha256'],r['duration'],r['method'],r['kind'])
        if r.get('error'):
            if 'out of memory' in r['error'].lower():coverage[key]['oom'].add(r['batch'])
        elif r.get('median_seconds') is not None:
            coverage[key]['success'].add(r['batch'])
            points[(r['gpu'],r['host'],r['source_sha256'],r['duration'],r['kind'])][r['method']].append(r)
    summaries=[]
    # Coverage is per GPU and source across hosts (each point retains host in CSV).
    combined=defaultdict(lambda:{'success':set(),'oom':set(),'hosts':set()})
    for (gpu,host,source,duration,method,kind),v in coverage.items():
        key=(gpu,source,duration,method,kind);combined[key]['success'].update(v['success']);combined[key]['oom'].update(v['oom']);combined[key]['hosts'].add(host)
    # Include absent methods so a partial campaign is visibly incomplete.
    groups={(r['gpu'],r['source_sha256'],r['duration']) for r in records if r.get('duration') is not None}
    for gpu,source,duration in groups:
        for method in METHODS:
            for kind in KINDS[method]:
                combined[(gpu,source,duration,method,kind)]
    for (gpu,source,duration,method,kind),v in sorted(combined.items()):
        summaries.append(dict(gpu=gpu,source_sha256=source,duration=duration,method=method,kind=kind,
                              hosts=sorted(v['hosts']),successful_batches=sorted(v['success']),
                              missing_batches=sorted(EXPECTED-v['success']),oom_batches=sorted(v['oom'])))
    report={'inputs':inputs,'rejected':rejected,'coverage':summaries,'record_count':len(records)}
    (args.output_dir/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    # Separate plots for each host and source version to avoid misleading comparisons.
    if points:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for (gpu,host,source,duration,kind),methods in points.items():
            fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
            for method,rows in sorted(methods.items()):
                grouped=defaultdict(list)
                for r in rows:grouped[r['batch']].append(r)
                xs=sorted(grouped)
                latency=[statistics.median(r['median_seconds'] for r in grouped[x]) for x in xs]
                throughput=[statistics.median(r['items_per_second'] for r in grouped[x]) for x in xs]
                axes[0].plot(xs,latency,marker='.',label=method);axes[1].plot(xs,throughput,marker='.',label=method)
            axes[0].set(xlabel='Batch size',ylabel='Elapsed time (seconds)');axes[1].set(xlabel='Batch size',ylabel='Strings per second')
            for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize='small')
            fig.suptitle(f'{gpu} · {host} · {duration}s · {kind}\nsource {source[:12]}')
            filename=f'{slug(gpu)}-{slug(host)}-{source[:12]}-{duration}-{kind}.png'
            fig.savefig(args.output_dir/filename,dpi=160);plt.close(fig)
    print(f'{len(records)} rows; {len(inputs)} accepted inputs; {len(rejected)} rejected; output {args.output_dir}')
if __name__=='__main__':main()
