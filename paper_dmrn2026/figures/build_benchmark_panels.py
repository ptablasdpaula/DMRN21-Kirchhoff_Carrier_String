"""Two-panel benchmark figure from a completed native-rate sweep.

Usage: build_benchmark_panels.py [sweep.json] [output-prefix] [width-in] [height-in]

Left panel: median seconds per evaluation on a broken axis. The lower band
is a symmetric-log scale anchored at zero (linear only between 0 and 0.1 s,
logarithmic above) labelled 0 and 1 with marks at 0.1, 0.2, ..., 0.9; the
upper band is logarithmic from 100 to 500 s with marks at 200, 300 and 400.
Both bands use the same physical size per decade. Right panel: peak sampled
GPU memory on a log axis. Both panels run against batch size with one colour
and line style per method (Cached adjoint ``cached3``, Modal ``diaz_44100``
at the sweep's primary mode count, Autodiff ``fdtd``). Only measured points
are drawn: a curve simply ends at the last batch that completed, and no
value is estimated for a failed batch.
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np

folder = Path(__file__).resolve().parent
source = Path(sys.argv[1]) if len(sys.argv) > 1 else folder/'native-rate-sweep-34855605.json'
prefix = Path(sys.argv[2]) if len(sys.argv) > 2 else folder/'benchmark-panels'
WIDTH = float(sys.argv[3]) if len(sys.argv) > 3 else 3.5
HEIGHT = float(sys.argv[4]) if len(sys.argv) > 4 else 1.6
data = json.loads(source.read_text())
metadata = data['metadata']
if metadata['status'] != 'complete' or metadata.get('source_changed_during_run'):
    raise ValueError('The measurement is incomplete or source hashes changed')
batches = tuple(metadata['batch_sizes'])
primary = metadata.get('primary_modal_modes') or metadata['native_modal_modes'][0]
rows = data['rows']


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
autodiff = select('fdtd')
modal = select('diaz_44100', primary)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':6.8,
    'axes.labelsize':7,'xtick.labelsize':6.4,'ytick.labelsize':6.4,
    'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.6,
    'xtick.major.width':.55,'ytick.major.width':.55,'ytick.minor.width':.4})
PURPLE, ORANGE, BLACK = '#7625a8', '#d55e00', '#151515'
styles = ((cached, PURPLE, '-', 'Cached adjoint'),
          (modal, ORANGE, '--', 'Modal'),
          (autodiff, BLACK, ':', 'Autodiff'))
LOWER, UPPER, LINEAR = (0, 1), (100, 500), .1     # time bands; linear part of the lower band
LINSCALE = .3                                     # decades of height given to the 0-0.1 s part
decades_lower = LINSCALE + np.log10(LOWER[1]/LINEAR)
decades_upper = np.log10(UPPER[1]/UPPER[0])

fig = plt.figure(figsize=(WIDTH, HEIGHT))
grid = fig.add_gridspec(2, 2, height_ratios=[decades_upper, decades_lower], hspace=.22, wspace=.4,
                        left=.105, right=.995, bottom=.2, top=.86)
ax_top = fig.add_subplot(grid[0, 0])
ax_bot = fig.add_subplot(grid[1, 0], sharex=ax_top)
ax_m = fig.add_subplot(grid[:, 1])
seconds = lambda r: r['median_seconds']
gigabytes = lambda r: r['peak_gpu_bytes']/1e9


def draw(axis, value):
    for seq, colour, linestyle, name in styles:
        good = [r for r in seq if r['status'] == 'ok']
        axis.plot([r['batch'] for r in good], [value(r) for r in good], color=colour,
                  ls=linestyle, lw=1.3, zorder=3, solid_capstyle='round', dash_capstyle='round')


def style_x(axis, labels=True):
    axis.set_xscale('log', base=2)
    axis.set_xlim(batches[0]*.78, batches[-1]*1.28)
    axis.set_xticks(batches, [str(v) for v in batches] if labels else []*len(batches))
    axis.tick_params(axis='x', length=2 if labels else 0, pad=1.5, labelbottom=labels)


plain = FuncFormatter(lambda v, _: f'{v:g}')
for axis in (ax_top, ax_bot, ax_m):
    draw(axis, seconds if axis is not ax_m else gigabytes)
    axis.spines['top'].set_visible(False); axis.spines['right'].set_visible(False)
    axis.tick_params(axis='y', length=2.4, pad=1.5, which='major')
    axis.tick_params(axis='y', length=1.4, which='minor')
    axis.grid(axis='y', which='major', color='#d4d4d4', lw=.45, zorder=0)
    axis.grid(axis='y', which='minor', color='#ececec', lw=.35, zorder=0)
    axis.set_axisbelow(True)

# Broken time axis, both bands with the same height per decade.
style_x(ax_top, labels=False); style_x(ax_bot); style_x(ax_m)
ax_top.spines['bottom'].set_visible(False)
ax_bot.set_yscale('symlog', base=10, linthresh=LINEAR, linscale=LINSCALE)
ax_bot.set_ylim(*LOWER)
ax_bot.set_yticks([LOWER[0], LOWER[1]])
ax_bot.set_yticks(np.arange(LINEAR, LOWER[1], LINEAR), minor=True)
ax_top.set_yscale('log')
ax_top.set_ylim(*UPPER)
ax_top.set_yticks([UPPER[0], UPPER[1]])
ax_top.set_yticks(np.arange(2*UPPER[0], UPPER[1], UPPER[0]), minor=True)
for axis in (ax_top, ax_bot):          # after set_yticks, which resets the formatters
    axis.yaxis.set_major_formatter(plain)
    axis.yaxis.set_minor_formatter(NullFormatter())
mark = dict(marker=[(-1, -.5), (1, .5)], markersize=5, linestyle='none', color=BLACK,
            mec=BLACK, mew=.7, clip_on=False, zorder=5)
ax_top.plot([0], [0], transform=ax_top.transAxes, **mark)
ax_bot.plot([0], [1], transform=ax_bot.transAxes, **mark)
ax_bot.set_xlabel('Batch size', labelpad=1.5)
top, bottom = ax_top.get_position(), ax_bot.get_position()
fig.text(bottom.x0 - .082, (top.y1 + bottom.y0)/2, 'Evaluation Time (s)',
         rotation=90, ha='center', va='center', fontsize=plt.rcParams['axes.labelsize'])

ax_m.set_yscale('log')
values = [gigabytes(r) for seq in (cached, modal, autodiff) for r in seq if r['status'] == 'ok']
ax_m.set_ylim(min(values)/1.6, max(values)*1.5)
ax_m.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
ax_m.yaxis.set_major_formatter(plain)
ax_m.yaxis.set_minor_formatter(NullFormatter())
ax_m.set_xlabel('Batch size', labelpad=1.5)
ax_m.set_ylabel('Peak Memory (GBs)', labelpad=2)

legend = [Line2D([], [], color=c, ls=ls, lw=1.3, label=n) for _, c, ls, n in styles]
fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, 1.0), ncol=3, frameon=False,
           fontsize=6.6, handlelength=2.2, columnspacing=1.6, handletextpad=.5, borderpad=0,
           borderaxespad=0, labelspacing=.2)
for ext in ('pdf', 'png'):
    fig.savefig(prefix.with_suffix(f'.{ext}'), dpi=400, bbox_inches='tight', pad_inches=0.01)
plt.close(fig)
print('Wrote', prefix.with_suffix('.pdf'))
