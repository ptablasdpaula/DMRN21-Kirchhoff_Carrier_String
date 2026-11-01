"""Plot the live native-rate study without mislabeling rates or mode counts."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from benchmarks import figure_comparison_plot as base


def mode_counts(payload):
    """(primary, all) modal counts; older results without metadata keys used 175, 123 and 8."""
    meta=payload.get('metadata') or {}
    counts=list(meta.get('native_modal_modes') or (175,123,8))
    primary=meta.get('primary_modal_modes') or counts[0]
    return primary,counts


def normalized(payload):
    if (payload.get('metadata') or {}).get('status')=='invalid_source_changed':
        raise ValueError('Cannot render a run with changed source files')
    primary,_=mode_counts(payload)
    rows=[]
    for row in payload['rows']:
        method=row.get('method')
        if method in ('cached3','cached1','fdtd'):
            rows.append(row)
        elif method=='diaz_44100' and row.get('modes')==primary:
            rows.append({**row,'method':'diaz'})
    return base.normalize_rows({'rows':rows})


def render(payload,prefix):
    prefix=Path(prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
    index=normalized(payload)
    meta=payload.get('metadata') or {}
    primary,counts=mode_counts(payload)
    duration=meta.get('duration_seconds') or 4
    excitation=meta.get('excitation') or 'multimode'
    rate=(meta.get('cached_rate') or 132300)/1000
    order=meta.get('spatial_order') or 2
    base.LABELS.update({'fdtd':f'Autodiff FDTD, {rate:g} kHz',
                        'cached3':f'Cached FDTD, {rate:g} kHz',
                        'diaz':f'Díaz JAX, 44.1 kHz, {primary} modes'})
    base.write_tables(index,prefix)
    table=prefix.with_suffix('.md')
    with table.open('a',encoding='utf8') as f:
        f.write(f'\nFDTD uses {rate:g} kHz internally (spatial order {order}) with anti-alias filtering; '
                f'Díaz JAX uses 44.1 kHz without oversampling and {primary} continuous modes. '
                f'Excitation: {excitation}; duration {duration:g} s.\n')
    fig,(ax,ax_mem)=base.create_figure(index,meta)
    ax.set_title(f"{base.gpu_label(index,meta)}\n{duration:g} s · 44.1 kHz output · 10 gradients",loc='right',fontsize=8,color='#52606d',pad=12)
    # A dual-axis plot can put both series at the same height, especially for
    # single-point partial results. Keep the blue interior visible through a
    # slightly larger open red marker and move the legend away from the data.
    for line in ax_mem.lines:
        line.set_markerfacecolor('none');line.set_markersize(line.get_markersize()+2)
    legend=ax.get_legend()
    if legend is not None:
        handles=legend.legend_handles;labels=[t.get_text() for t in legend.get_texts()]
        legend.remove()
        ax.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,-.17),
                  ncol=1,frameon=False,fontsize=8.5)
    for ext in ('.png','.pdf'):fig.savefig(prefix.with_suffix(ext),dpi=300,bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)
    experimental={(r.get('batch'),r.get('modes')):r for r in payload['rows'] if r.get('method')=='diaz_44100'}
    lines=['# Modal-count study','',
           f'44.1 kHz JAX; cells are seconds / sampled GPU GB. Excitation: {excitation}. '
           'Equivalence between mode counts holds only for excitations without content in the omitted modes and no later forcing.','',
           '| Batch | '+' | '.join(f'{m} modes' for m in counts)+' |','|---:|'+'---:|'*len(counts)]
    for batch in base.BATCHES:
        lines.append('| '+str(batch)+' | '+' | '.join(base._display(experimental.get((batch,m))) for m in counts)+' |')
    prefix.with_name(prefix.name+'-modes.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--output-prefix',type=Path,required=True);a=p.parse_args()
    render(json.loads(a.input.read_text()),a.output_prefix)


if __name__=='__main__':main()
