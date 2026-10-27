"""Plot optimizer trajectories and target/initial/best-fit CeL spectrograms."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from src.kc_string_forward import ForwardString
from cels import STFTPower


def create_plots(data, output_directory, signals=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    output_directory=Path(output_directory)
    output_directory.mkdir(parents=True,exist_ok=True)
    config=data['configuration']
    results=data['results']
    figure,axes=plt.subplots(1,3,figsize=(15,4.2),layout='constrained')
    colors={'adam':'#1976b5','lbfgs':'#ce5a16'}
    for name,result in results.items():
        trace=result['history']
        for axis,key in zip(axes,['iteration','evaluations','elapsed_seconds']):
            x=[r[key] for r in trace]
            axis.semilogy(x,[r['loss'] for r in trace],color=colors[name],alpha=.35,linewidth=1)
            axis.semilogy(x,[r['best_loss'] for r in trace],color=colors[name],linewidth=2,label=name.upper())
            axis.scatter(x[-1],trace[-1]['best_loss'],color=colors[name],s=25,zorder=3)
    for axis,title,label in zip(axes,['Optimizer steps','Loss/gradient evaluations','Wall-clock time'],['Completed optimizer steps','Evaluations (including line search)','Seconds']):
        axis.set(title=title,xlabel=label,ylabel='Default CeL loss')
        axis.grid(alpha=.2,which='both')
        axis.legend()
    figure.suptitle('CeL optimization: faint = current loss; bold = best so far\nStop: less than 10% best-loss improvement over 20 optimizer steps',fontsize=12)
    for extension in ('png','pdf'):
        figure.savefig(output_directory/f'loss_evolution.{extension}',dpi=180)
    plt.close(figure)
    if signals is None:
        model=ForwardString(intervals=config['intervals'],sample_rate=config['sample_rate'])
        theta={'target':config['target_theta'],'initial':config['initial_theta'],
               **{name:result['best_theta'] for name,result in results.items()}}
        with torch.no_grad():
            signals={name:model(torch.tensor(values,dtype=torch.float64),config['duration_seconds']) for name,values in theta.items()}
    transform=STFTPower(sample_rate=config['sample_rate'])
    with torch.no_grad():
        power={name:transform(audio) for name,audio in signals.items()}
    common_peak=max(p.max().item() for p in power.values())
    frequency,time=transform.coordinates(power['target'])
    # Coordinates are displayed at the Hann-window centers (center=False).
    time=time+transform.n_fft/(2*config['sample_rate'])
    for filename,limit in [('spectrograms',1000),('spectrograms_full_band',config['sample_rate']/2)]:
        figure,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,sharey=True,layout='constrained')
        for axis,name in zip(axes.flat,['target','initial','adam','lbfgs']):
            db=10*torch.log10((power[name]/common_peak).clamp_min(1e-8))
            artist=axis.pcolormesh(time.numpy(),frequency.numpy(),db.numpy(),shading='auto',vmin=-80,vmax=0,cmap='magma',rasterized=True)
            title={'target':'Target','initial':'Initial: signed 10% perturbation','adam':'Adam: best fit','lbfgs':'L-BFGS: best fit'}[name]
            if name in results:
                title+=f" | CeL = {results[name]['best_loss']:.3g}"
            axis.set(title=title,ylim=(0,limit),xlim=(0,config['duration_seconds']))
        for axis in axes[-1]: axis.set_xlabel('Time (s)')
        for axis in axes[:,0]: axis.set_ylabel('Frequency (Hz)')
        figure.colorbar(artist,ax=axes.ravel().tolist(),label='Power (dB relative to the common peak)',shrink=.85)
        figure.suptitle(f'Same STFT as the loss: 1024-point periodic Hann, hop 256; shared −80…0 dB scale\nFrequency view: 0–{limit:g} Hz; no per-signal normalization',fontsize=12)
        for extension in ('png','pdf'):
            figure.savefig(output_directory/f'{filename}.{extension}',dpi=180)
        plt.close(figure)
    figure,axis=plt.subplots(figsize=(10,4),layout='constrained')
    labels=['density','tension','radius','E','length','amplitude','sigma0','sigma1']
    for offset,(name,values) in zip([-.25,0,.25],[('Initial',config['initial_theta']),('Adam',results['adam']['best_theta']),('L-BFGS',results['lbfgs']['best_theta'])]):
        axis.bar(np.arange(8)+offset,100*(np.array(values)-1),width=.25,label=name)
    axis.axhline(0,color='black',linewidth=.8)
    axis.set(xticks=np.arange(8),xticklabels=labels,ylabel='Error relative to target (%)',title='Physical parameter recovery (best-loss checkpoints)')
    axis.legend(); axis.grid(axis='y',alpha=.2)
    figure.savefig(output_directory/'parameter_recovery.png',dpi=180)
    plt.close(figure)
    print('Plots saved to',output_directory)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'experiments/optimization_cel_results.json')
    parser.add_argument('--output',type=Path,default=ROOT/'.notebook-build/optimization_cel')
    args=parser.parse_args()
    torch.set_num_threads(1)
    create_plots(json.loads(args.results.read_text()),args.output)
