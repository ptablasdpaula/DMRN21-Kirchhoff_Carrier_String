"""Rebuild the measured tables and figure from checked-in raw observations."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
data=ROOT/'benchmarks/gpu_results'
observations={name:json.loads((data/f'{name.lower()}-matched.json').read_text()) for name in ('A100','H100')}
backends=[('cpu','native'),('cpu','spatial'),('cpu','modal'),('cuda','spatial'),('cuda','modal')]
labels=['Original CPU','Spatial CPU','Modal CPU','Spatial CUDA','Modal CUDA']
def select(run,kind,device,backend,**filters):
    return next(r for r in run['rows'] if r['kind']==kind and r.get('device')==device and r['backend']==backend and all(r.get(k)==v for k,v in filters.items()))
lines=['| Backend | A100 node (s) | H100 node (s) |','|---|---:|---:|']
fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
for label,(device,backend) in zip(labels,backends):
    times=[select(run,'full_evaluation',device,backend)['median_seconds'] for run in observations.values()]
    lines.append(f'| {label} | {times[0]:.3f} | {times[1]:.3f} |')
for ax,(name,run) in zip(axes,observations.items()):
    times=[select(run,'full_evaluation',d,b)['median_seconds'] for d,b in backends]
    ax.barh(labels,times,color=['#91979e','#91979e','#91979e','#4c7ba9','#237d63'])
    ax.invert_yaxis();ax.set_xlabel('Loss + backward, seconds (lower is better)')
    ax.set_title(f'{name} node: four-second single item')
    for i,v in enumerate(times):ax.text(v+.06,i,f'{v:.2f}',va='center',fontsize=9)
    ax.set_xlim(0,max(times)*1.16)
fig.savefig(data/'latency.png',dpi=180)
(data/'latency-table.md').write_text('\n'.join(lines)+'\n')
lines=['| Batch | A100 items/s | H100 items/s |','|---:|---:|---:|']
for batch in (1,2,4,8,16,32,64):
    values=[select(run,'batch','cuda','modal',batch=batch)['items_per_second'] for run in observations.values()]
    lines.append(f'| {batch} | {values[0]:.1f} | {values[1]:.1f} |')
(data/'batch-table.md').write_text('\n'.join(lines)+'\n')
print((data/'latency-table.md').read_text());print((data/'batch-table.md').read_text())
