"""Regenerate the compact benchmark figure from a completed native-rate sweep.

Usage: build_benchmark_figure.py [sweep.json] [output-prefix]

The JSON is an untouched native-rate sweep result. Three series are drawn:
Cached adjoint (``cached3``), Autodiff (``fdtd``, direct PyTorch autodiff) and
Modal (``diaz_44100`` at the sweep's primary mode count). Only measured
points are plotted; a cross marks the first batch at which a method failed and
no value is estimated for it. When every failure has the same recorded status
it is named once in the legend, otherwise each cross is labelled.
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path as MarkerPath
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np

folder = Path(__file__).resolve().parent
source = Path(sys.argv[1]) if len(sys.argv) > 1 else folder/'native-rate-sweep-34855605.json'
prefix = Path(sys.argv[2]) if len(sys.argv) > 2 else folder/'benchmark-main'
data = json.loads(source.read_text())
metadata = data['metadata']
if metadata['status'] != 'complete' or metadata.get('source_changed_during_run'):
    raise ValueError('The measurement is incomplete or source hashes changed')
batches = tuple(metadata['batch_sizes'])
primary = metadata.get('primary_modal_modes') or metadata['native_modal_modes'][0]
rows = data['rows']
STATUS = {'oom': 'OOM', 'host_memory_limit': 'host', 'timeout': 'time', 'error': 'error'}


def select(method, modes=None):
    found = {}
    for row in rows:
        if row.get('method') == method and (modes is None or row.get('modes') == modes):
            if row['batch'] in found:
                raise ValueError(f'duplicate result: {method} {modes} {row["batch"]}')
            found[row['batch']] = row
    if set(found) != set(batches):
        raise ValueError(f'missing batches for {method} {modes}: {set(batches)-set(found)}')
    return [found[b] for b in batches]


cached = select('cached3')
naive = select('fdtd')
modal = select('diaz_44100', primary)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':6.2,
    'axes.labelsize':6.5,'xtick.labelsize':6,'ytick.labelsize':6,
    'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.65,
    'xtick.major.width':.6,'ytick.major.width':.6})
fig, ax = plt.subplots(figsize=(3.3,1.22))
right = ax.twinx()
black, purple = '#151515', '#7625a8'
# Unfilled sine-shaped symbol avoids implying a waveform trace in time.
points = np.linspace(-1,1,40)
wave = MarkerPath(np.column_stack((points,.53*np.sin(2*np.pi*points))),
                  [MarkerPath.MOVETO]+[MarkerPath.LINETO]*39)
styles = ((cached, '*', '-', 5.6, True, 'Cached adjoint'),
          (naive, 'o', ':', 3.4, False, 'Autodiff'),
          (modal, wave, '--', 8.0, False, 'Modal'))
times, memories, failures = [], [], []
for seq, marker, linestyle, markersize, filled, label in styles:
    good = [r for r in seq if r['status'] == 'ok']
    x = [r['batch'] for r in good]
    t = [r['median_seconds'] for r in good]
    m = [r['peak_gpu_bytes']/1e9 for r in good]
    times += t; memories += m
    for axis, values, colour in ((ax, t, black), (right, m, purple)):
        axis.plot(x, values, color=colour, marker=marker, ls=linestyle, lw=1.0, ms=markersize, mew=.9,
                  markerfacecolor=colour if filled else 'none', clip_on=True, zorder=3)
    failed = [r for r in seq if r['status'] != 'ok']
    if failed and good:
        first = min(failed, key=lambda r: r['batch'])
        last = max(good, key=lambda r: r['batch'])
        failures.append((first, last))
        for axis, value, colour in ((ax, last['median_seconds'], black), (right, last['peak_gpu_bytes']/1e9, purple)):
            axis.plot([last['batch'], first['batch']], [value, value], color=colour, ls=':', lw=.7, zorder=2)
            axis.plot([first['batch']], [value], color=colour, marker='x', ms=4.2, mew=.9, ls='none', zorder=4)
statuses = {STATUS.get(first['status'], first['status']) for first, _ in failures}
failure_label = 'first failure'
if len(statuses) == 1:
    failure_label += f' ({statuses.pop()})'
else:
    for first, last in failures:
        ax.annotate(STATUS.get(first['status'], first['status']), (first['batch'], last['median_seconds']),
                    xytext=(2.5, 2.5), textcoords='offset points', fontsize=5.2, color=black)

ax.set_xscale('log', base=2)
ax.set_xlim(batches[0]*.84, batches[-1]*1.18)
ax.set_xticks(batches, [str(v) for v in batches])
for axis, values in ((ax, times), (right, memories)):
    axis.set_yscale('log')
    low, high = min(values), max(values)
    axis.set_ylim(low/1.8, high*2.2)
    axis.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    axis.yaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel('Batch size', labelpad=1)
ax.set_ylabel('Time (s)', color=black, labelpad=2)
right.set_ylabel('GPU memory (GB)', color=purple, labelpad=2)
ax.tick_params(axis='y', colors=black, length=2, pad=1, which='both')
right.tick_params(axis='y', colors=purple, length=2, pad=1, which='both')
ax.tick_params(axis='x', length=2, pad=1)
ax.spines['left'].set_color(black)
right.spines['right'].set_color(purple)
ax.spines['top'].set_visible(False)
right.spines['top'].set_visible(False)
ax.grid(axis='y', which='major', color='#dddddd', lw=.4, zorder=0)
ax.set_axisbelow(True)

legend = [Line2D([], [], color=black, marker='*', ls='-', lw=1, ms=5.6, label='Cached adjoint'),
          Line2D([], [], color=black, marker='o', ls=':', lw=1, ms=3.4, markerfacecolor='none', label='Autodiff'),
          Line2D([], [], color=black, marker=wave, ls='--', lw=1, ms=8, label='Modal'),
          Line2D([], [], color=black, marker='x', ls='none', ms=4.2, label=failure_label)]
# One legend row above the axes keeps it off the data at every batch size.
fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, 1.0), ncol=4, frameon=False,
           fontsize=5.8, handlelength=1.6, columnspacing=.7, handletextpad=.3, borderpad=0,
           borderaxespad=0, labelspacing=.2)
fig.subplots_adjust(left=.14, right=.84, bottom=.32, top=.84)
for ext in ('pdf', 'png'):
    fig.savefig(prefix.with_suffix(f'.{ext}'), dpi=400)
plt.close(fig)
print('Wrote', prefix.with_suffix('.pdf'))
