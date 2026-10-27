"""Generate a comparison table and loss plot from saved full-duration fits."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,default=Path('experiments/results'))
    parser.add_argument('--output',type=Path,default=Path('output/recovery'))
    args=parser.parse_args()
    results={name:json.loads((args.results/f'recovery_{name}.json').read_text()) for name in ('adam','lbfgs')}
    args.output.mkdir(parents=True,exist_ok=True)
    baseline=results['adam']
    for result in results.values():
        if any(result[key] != baseline[key] for key in ('grid','duration','excitation','target','requested_initial','loss')):
            raise ValueError('Results must share the same configuration, target, initial values and loss.')
    lines=['# Full-duration recovery', '',
           f"{baseline['duration']:g}-second experiment: N={baseline['grid']['intervals']}, oversampling ×{baseline['grid']['oversampling']}. "
           'Both methods start independently; loss and duration stay unchanged.', '',
           '| Parameter | Target | Initial | Adam | Error | L-BFGS | Error |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for i,name in enumerate(baseline['parameter_names']):
        target=baseline['target'][i];initial=baseline['requested_initial'][i]
        columns=[name,f'{target:.7g}',f'{initial:.7g}']
        for result in results.values():
            value=result['best_physical'][i]
            error=f'{100*(value/target-1):+.3f}%' if target else 'undefined (zero target)'
            columns.extend([f'{value:.7g}',error])
        lines.append('| '+' | '.join(columns)+' |')
    lines += ['', '| Method | Initial loss | Best loss | Steps | Evaluations | Seconds |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,result in results.items():
        lines.append(f"| {name} | {result['history'][0]['loss']:.7g} | {result['best_loss']:.7g} | {result['iterations']} | {result['evaluations']} | {result['seconds']:.1f} |")
    for name,result in results.items():lines.extend(['',f"{name}: {result['stop_reason']}"])
    lines.extend(['','These are recovery results, not proof of parameter identifiability or nonlinear accuracy. '
                  'Times are observed fitting wall time; concurrent validation may affect them.'])
    report='\n'.join(lines)+'\n'
    (args.output/'summary.md').write_text(report)
    print(report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    for name,result in results.items():
        history=result['history']
        for ax,key in zip(axes,('iteration','seconds')):
            ax.semilogy([row[key] for row in history],[max(row['best_loss'],1e-16) for row in history],label=name.upper())
    for ax,label in zip(axes,('Completed optimizer steps','Fitting wall time (s)')):
        ax.set(xlabel=label,ylabel='Best CeL');ax.grid(alpha=.2);ax.legend()
    fig.savefig(args.output/'loss.png',dpi=160)
    fig.savefig(args.output/'loss.pdf')
    plt.close(fig)

if __name__=='__main__':main()
