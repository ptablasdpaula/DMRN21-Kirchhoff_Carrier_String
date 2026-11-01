"""Single-panel benchmark figure: peak GPU memory against batch size as dots.

Usage: build_benchmark_dots.py size|label [sweep.json] [output-prefix] [decorated]

One dot per measured (method, batch): Ours (``cached3``), jaxdiffmodal
(``diaz_44100`` at the sweep's primary mode count) and Autodiff (``fdtd``),
y = peak sampled GPU memory on a log axis. Mode ``size`` encodes the median
seconds per evaluation in the dot's diameter, linear in log time between the
smallest and largest measured time, with a size key; mode ``label`` draws
small uniform dots joined by thin dashed lines, prints the seconds next to
the first and last dot of each series, names each series directly in its own
colour instead of a legend (in the open space near its line, or, with the
``decorated`` flag, parallel to the line together with a dotted line at the
benchmark GPU's 80 GB limit). Only measured points are
drawn; failed batches are simply absent.
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

MODE = sys.argv[1] if len(sys.argv) > 1 else 'size'
if MODE not in ('size', 'label'):
    raise SystemExit('mode must be size or label')
folder = Path(__file__).resolve().parent
source = Path(sys.argv[2]) if len(sys.argv) > 2 else folder/'native-rate-sweep-34855605.json'
prefix = Path(sys.argv[3]) if len(sys.argv) > 3 else folder/f'benchmark-dots-{MODE}'
WIDTH, HEIGHT = 3.5, 2.02
COLUMN_SCALE = 1.035   # tight-bbox PDF width over the printed column width (240.9 pt measured on the page)
TEXT = 8*COLUMN_SCALE    # every label prints at the abstract's caption size (8 pt) in a Times-compatible face
DECORATED = len(sys.argv) > 4 and sys.argv[4] == 'decorated'
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
    return [found[b] for b in batches if found[b]['status'] == 'ok']


PURPLE, ORANGE, BLACK = '#7625a8', '#d55e00', '#151515'
series = (('Ours', select('cached3'), PURPLE, 'below'),
          ('jaxdiffmodal', select('diaz_44100', primary), ORANGE, 'below'),
          ('Autodiff', select('fdtd'), BLACK, 'above'))
seconds = lambda r: r['median_seconds']
gigabytes = lambda r: r['peak_gpu_bytes']/1e9
times = [seconds(r) for _, seq, _, _ in series for r in seq]
log_lo, log_hi = np.log10(min(times)), np.log10(max(times))
D_MIN, D_MAX = 2.8, 10.0                       # dot diameter in points at the fastest and slowest time
diameter = lambda t: D_MIN + (D_MAX-D_MIN)*(np.log10(t)-log_lo)/(log_hi-log_lo)
time_text = lambda t: f'{t:.2f}' if t < 10 else f'{t:.0f}'

plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':TEXT,
    'axes.labelsize':TEXT,'xtick.labelsize':TEXT,'ytick.labelsize':TEXT,
    'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.6,
    'xtick.major.width':.55,'ytick.major.width':.55,'ytick.minor.width':.4})
fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
# Slowest series first so that smaller dots are never hidden under larger ones.
for name, seq, colour, side in sorted(series, key=lambda item: -max(seconds(r) for r in item[1])):
    x = [r['batch'] for r in seq]; y = [gigabytes(r) for r in seq]
    if MODE == 'size':
        ax.scatter(x, y, s=[diameter(seconds(r))**2 for r in seq], c=colour, alpha=.85,
                   edgecolors='white', linewidths=.6, zorder=3)
    else:
        ax.plot(x, y, color=colour, ls=(0, (2.2, 2.2)), lw=.55, zorder=2)
        ax.scatter(x, y, s=2.6**2, c=colour, edgecolors='none', zorder=3)
        # Each series prints only its first and last time, with the unit.
        for r in (seq[0], seq[-1]):
            text = time_text(seconds(r)) + ' s'
            ax.annotate(text, (r['batch'], gigabytes(r)), xytext=(0, 3.6 if side == 'above' else -3.6),
                        textcoords='offset points', ha='center', va='bottom' if side == 'above' else 'top',
                        fontsize=TEXT, color=colour, zorder=4)
ax.set_xscale('log', base=2)
ax.set_xlim(batches[0]*.72, batches[-1]*1.38)
ax.set_xticks(batches, [str(v) for v in batches])
ax.set_yscale('log')
memories = [gigabytes(r) for _, seq, _, _ in series for r in seq]
ax.set_ylim(min(memories)/2.2, max(memories)*2.4)
ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel('Batch size', labelpad=1.5)
ax.set_ylabel('Peak Memory (GBs)', labelpad=2)
ax.tick_params(axis='both', length=2, pad=1.5, which='major')
ax.tick_params(axis='y', length=1.2, which='minor')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10), numticks=20))
ax.grid(axis='y', which='major', color='#d6d6d6', lw=.45, zorder=0)
ax.grid(axis='y', which='minor', color='#ededed', lw=.3, zorder=0)   # 2, 3, ... 9 within each decade
ax.set_axisbelow(True)

if MODE == 'label':
    # Direct labels in series colour: Ours above its last dot, jaxdiffmodal in the
    # open band between its line and Ours, Autodiff in the corner above its line.
    by_name = {name: seq for name, seq, _, _ in series}
    at = lambda name, batch: next(gigabytes(r) for r in by_name[name] if r['batch'] == batch)
    last = by_name['Ours'][-1]
    ax.annotate('Ours', (last['batch'], gigabytes(last)), xytext=(0, 4.5), textcoords='offset points',
                ha='center', va='bottom', fontsize=TEXT, color=PURPLE)
    fig.subplots_adjust(left=.11, right=.99, bottom=.2, top=.97)
    fig.canvas.draw()                      # finalise transforms before measuring line angles

    def along(name, b0, b1, side, colour, gap=5.5):
        # Name a series parallel to its segment between batches b0 and b1, a few points off the line.
        y0, y1 = at(name, b0), at(name, b1)
        (x0, Y0), (x1, Y1) = ax.transData.transform([(b0, y0), (b1, y1)])
        angle = np.degrees(np.arctan2(Y1-Y0, x1-x0))
        normal = np.array([-(Y1-Y0), x1-x0]); normal = normal/np.linalg.norm(normal)
        ax.annotate(name, (np.sqrt(b0*b1), np.sqrt(y0*y1)), xytext=tuple(normal*(gap if side == 'above' else -gap)),
                    textcoords='offset points', ha='center', va='center', rotation=angle,
                    rotation_mode='anchor', fontsize=TEXT, color=colour, zorder=4)
    if DECORATED:
        along('Autodiff', 2, 4, 'above', BLACK)
        along('jaxdiffmodal', 16, 32, 'below', ORANGE)
        # Device memory limit of the benchmark GPU.
        ax.axhline(80, color='#8c8c8c', ls=(0, (1, 2)), lw=.6, zorder=1)
        ax.annotate('A100 80 GBs limit', (batches[-1]*1.38, 80), xytext=(-2, -1.8), textcoords='offset points',
                    ha='right', va='top', fontsize=TEXT, color='#8c8c8c', zorder=4)
    else:
        def line_value(name, x):
            # Log-log interpolation of a series between the measured batches bracketing x.
            pts = [(r['batch'], gigabytes(r)) for r in by_name[name]]
            for (b0, y0), (b1, y1) in zip(pts, pts[1:]):
                if b0 <= x <= b1:
                    f = (np.log(x)-np.log(b0))/(np.log(b1)-np.log(b0))
                    return np.exp(np.log(y0)+f*(np.log(y1)-np.log(y0)))
            return pts[0][1] if x < pts[0][0] else pts[-1][1]

        def place(name, b0, b1, side, colour, gap=3.0):
            # Horizontal name centred between batches b0 and b1, its nearest edge `gap` points off the line.
            xc = np.sqrt(b0*b1)
            text = ax.text(xc, 1.0, name, ha='center', va='bottom' if side == 'above' else 'top',
                           fontsize=TEXT, color=colour, zorder=4)
            fig.canvas.draw()
            box = text.get_window_extent(fig.canvas.get_renderer())
            edge = box.x1 if side == 'above' else box.x0          # the edge nearest a line rising to the right
            inverse = ax.transData.inverted()
            x_edge = inverse.transform((edge, 0))[0]
            y_line = ax.transData.transform((x_edge, line_value(name, x_edge)))[1]
            y_text = y_line + (gap if side == 'above' else -gap)*fig.dpi/72
            text.set_position((xc, inverse.transform((edge, y_text))[1]))
        place('Autodiff', 2, 4, 'above', BLACK)
        place('jaxdiffmodal', 32, 64, 'below', ORANGE)
else:
    handles = [Line2D([], [], marker='o', ls='none', color=c, ms=5, label=n) for n, _, c, _ in series]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, 1.0), ncol=3, frameon=False,
               fontsize=6.6, handletextpad=.3, columnspacing=1.6, borderpad=0, borderaxespad=0)
    fig.subplots_adjust(left=.11, right=.99, bottom=.2, top=.87)
if MODE == 'size':
    key = [Line2D([], [], marker='o', ls='none', color='#8a8a8a', ms=diameter(t), label=f'{t:g} s')
           for t in (0.3, 3, 30, 300)]
    ax.legend(handles=key, loc='upper left', ncol=4, frameon=False, fontsize=5.8, handletextpad=.6,
              columnspacing=1.2, borderpad=.1, borderaxespad=.2, handlelength=1.8, title='evaluation time',
              title_fontsize=5.8)
for ext in ('pdf', 'png'):
    fig.savefig(prefix.with_suffix(f'.{ext}'), dpi=400, bbox_inches='tight', pad_inches=0.01)
plt.close(fig)
print('Wrote', prefix.with_suffix('.pdf'))
