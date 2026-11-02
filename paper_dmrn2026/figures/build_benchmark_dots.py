"""Single-panel benchmark figure: peak GPU memory against batch size as dots.

Usage: build_benchmark_dots.py size|label [sweep.json] [output-prefix] [decorated]

One dot per measured (method, batch), y = peak sampled GPU memory on a log
axis: Ours (``cached3``), jaxdiffmodal (``diaz_44100`` at the sweep's primary
mode count) and Autodiff (``fdtd``), plus, when the sweep contains them, the
rematerialised baselines ``diaz_44100_ckpt`` and ``fdtd_ckpt`` as hollow dots
in their method's colour (colour = method, fill = checkpointing). Mode
``label`` draws small dots joined by thin dashed (solid series) or dotted
(hollow series) lines, prints the seconds per evaluation next to the first
and last measured dot of each series, names each series directly in its own
colour instead of a legend, and marks the first batch that ran out of memory
with an x on the benchmark GPU's 80 GB line. Mode ``size`` is the older
variant that encodes time in the dot diameter for the three solid series.
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
source = Path(sys.argv[2]) if len(sys.argv) > 2 else folder/'native-rate-sweep-merged.json'
prefix = Path(sys.argv[3]) if len(sys.argv) > 3 else folder/f'benchmark-dots-{MODE}'
WIDTH, HEIGHT = 3.5, 1.62
COLUMN_SCALE = 1.035   # tight-bbox PDF width over the printed column width (240.9 pt measured on the page)
TEXT = 8*COLUMN_SCALE    # every label prints at the abstract's caption size (8 pt) in a Times-compatible face
DECORATED = len(sys.argv) > 4 and sys.argv[4] == 'decorated'
GPU_LIMIT_GB = 80
data = json.loads(source.read_text())
metadata = data['metadata']
if metadata['status'] != 'complete' or metadata.get('source_changed_during_run'):
    raise ValueError('The measurement is incomplete or source hashes changed')
batches = tuple(metadata['batch_sizes'])
primary = metadata.get('primary_modal_modes') or metadata['native_modal_modes'][0]
rows = data['rows']


def select(method, modes=None):
    """(measured rows in batch order, batches that failed) for one method."""
    found = {}
    for row in rows:
        if row.get('method') == method and (modes is None or row.get('modes') == modes):
            if row['batch'] in found:
                raise ValueError(f'duplicate result: {method} {modes} {row["batch"]}')
            found[row['batch']] = row
    if set(found) != set(batches):
        raise ValueError(f'missing batches for {method} {modes}: {set(batches)-set(found)}')
    return ([found[b] for b in batches if found[b]['status'] == 'ok'],
            [b for b in batches if found[b]['status'] != 'ok'])


PURPLE, ORANGE, BLACK, GREY = '#7625a8', '#d55e00', '#151515', '#8c8c8c'
# key, method, modes, printed name, colour, hollow
SPECS = (('ours', 'cached3', None, 'Ours', PURPLE, False),
         ('modal', 'diaz_44100', primary, 'jaxdiffmodal', ORANGE, False),
         ('autodiff', 'fdtd', None, 'Autodiff', BLACK, False),
         ('modal_ckpt', 'diaz_44100_ckptro', primary, 'jaxdiffmodal, checkpointed', ORANGE, True),
         ('modal_ckpt_traj', 'diaz_44100_ckpt', primary, 'jaxdiffmodal, checkpointed', ORANGE, True),
         ('autodiff_ckpt', 'fdtd_ckpt', None, 'Autodiff, checkpointed', BLACK, True))
present = [spec for spec in SPECS if spec[1] in metadata['methods'] and (MODE == 'label' or not spec[5])]
# The checkpointed modal series is the readout-in-scan variant when the sweep has it;
# the variant that returns the full modal trajectory stands in otherwise.
if any(spec[0] == 'modal_ckpt' for spec in present):
    present = [spec for spec in present if spec[0] != 'modal_ckpt_traj']
else:
    present = [(('modal_ckpt',) + spec[1:]) if spec[0] == 'modal_ckpt_traj' else spec for spec in present]
series = {key: dict(name=name, colour=colour, hollow=hollow, rows=select(method, modes)[0], failed=select(method, modes)[1])
          for key, method, modes, name, colour, hollow in present}
# Where each series prints its first and last time: above/below the dot, left/right of it, or not at all
# (the checkpointed series are flat in time, so their last value says it all).
TIME_SIDES = {'ours': ('below', 'below'), 'modal': ('right', 'right'), 'autodiff': ('above', 'right'),
              'modal_ckpt': (None, 'right'), 'autodiff_ckpt': (None, 'right')}
seconds = lambda r: r['median_seconds']
gigabytes = lambda r: r['peak_gpu_bytes']/1e9
times = [seconds(r) for s in series.values() for r in s['rows']]
log_lo, log_hi = np.log10(min(times)), np.log10(max(times))
D_MIN, D_MAX = 2.8, 10.0                       # dot diameter in points at the fastest and slowest time
diameter = lambda t: D_MIN + (D_MAX-D_MIN)*(np.log10(t)-log_lo)/(log_hi-log_lo)
time_text = lambda t: f'{t:.2f}' if t < 10 else f'{t:.0f}'

plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':TEXT,
    'axes.labelsize':TEXT,'xtick.labelsize':TEXT,'ytick.labelsize':TEXT,
    'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.6,
    'xtick.major.width':.55,'ytick.major.width':.55,'ytick.minor.width':.4})
fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))


def annotate_time(r, text, side, colour):
    offset = {'above': (0, 3.6), 'below': (0, -3.6), 'left': (-4.0, 0), 'right': (4.0, 0)}[side]
    ha = {'above': 'center', 'below': 'center', 'left': 'right', 'right': 'left'}[side]
    va = {'above': 'bottom', 'below': 'top', 'left': 'center', 'right': 'center'}[side]
    ax.annotate(text, (r['batch'], gigabytes(r)), xytext=offset, textcoords='offset points',
                ha=ha, va=va, fontsize=TEXT, color=colour, zorder=4)


# Slowest series first so that smaller dots are never hidden under larger ones.
for key, s in sorted(series.items(), key=lambda item: -max(seconds(r) for r in item[1]['rows'])):
    seq, colour, hollow = s['rows'], s['colour'], s['hollow']
    x = [r['batch'] for r in seq]; y = [gigabytes(r) for r in seq]
    if MODE == 'size':
        ax.scatter(x, y, s=[diameter(seconds(r))**2 for r in seq], c=colour, alpha=.85,
                   edgecolors='white', linewidths=.6, zorder=3)
        continue
    if hollow:
        ax.plot(x, y, color=colour, ls=(0, (1, 1.8)), lw=.55, zorder=2)
        ax.scatter(x, y, s=3.4**2, facecolors='white', edgecolors=colour, linewidths=.6, zorder=3)
    else:
        ax.plot(x, y, color=colour, ls=(0, (2.2, 2.2)), lw=.55, zorder=2)
        ax.scatter(x, y, s=2.6**2, c=colour, edgecolors='none', zorder=3)
    # Each series prints only its first and last time, with the unit.
    for r, side in zip((seq[0], seq[-1]), TIME_SIDES[key]):
        if side is not None:
            annotate_time(r, time_text(seconds(r)) + ' s', side, colour)
ax.set_xscale('log', base=2)
# Room on the right for the labels printed beside the last dots.
ax.set_xlim(batches[0]*.72, batches[-1]*(2.0 if MODE == 'label' else 1.38))
ax.set_xticks(batches, [str(v) for v in batches])
ax.set_yscale('log')
memories = [gigabytes(r) for s in series.values() for r in s['rows']]
ax.set_ylim(min(memories)/2.2, max(max(memories)*2.4, GPU_LIMIT_GB*1.6))
ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel('Batch size', labelpad=1.5)
ax.set_ylabel('Peak memory (GB)', labelpad=2)
ax.tick_params(axis='both', length=2, pad=1.5, which='major')
ax.tick_params(axis='y', length=1.2, which='minor')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10), numticks=20))
ax.grid(axis='y', which='major', color='#d6d6d6', lw=.45, zorder=0)
ax.grid(axis='y', which='minor', color='#ededed', lw=.3, zorder=0)   # 2, 3, ... 9 within each decade
ax.set_axisbelow(True)

if MODE == 'label':
    # Out-of-memory failures: an x in the series colour on the GPU's memory limit.
    if any(s['failed'] for s in series.values()):
        ax.axhline(GPU_LIMIT_GB, color=GREY, ls=(0, (1, 2)), lw=.6, zorder=1)
        ax.annotate(f'{GPU_LIMIT_GB} GB', (batches[0]*.72, GPU_LIMIT_GB), xytext=(1.5, 1.2), textcoords='offset points',
                    ha='left', va='bottom', fontsize=TEXT-.5, color=GREY, zorder=4)
        for s in series.values():
            if s['failed']:
                ax.scatter([s['failed'][0]], [GPU_LIMIT_GB], marker='x', s=14, c=s['colour'], linewidths=.8, zorder=4)
    # Direct labels in series colour.
    by_name = {key: s['rows'] for key, s in series.items()}
    at = lambda key, batch: next(gigabytes(r) for r in by_name[key] if r['batch'] == batch)
    fig.subplots_adjust(left=.11, right=.99, bottom=.2, top=.97)
    fig.canvas.draw()                      # finalise transforms before measuring line angles

    def along(key, b0, b1, side, gap=5.5):
        # Name a series parallel to its segment between batches b0 and b1, a few points off the line.
        y0, y1 = at(key, b0), at(key, b1)
        (x0, Y0), (x1, Y1) = ax.transData.transform([(b0, y0), (b1, y1)])
        angle = np.degrees(np.arctan2(Y1-Y0, x1-x0))
        normal = np.array([-(Y1-Y0), x1-x0]); normal = normal/np.linalg.norm(normal)
        ax.annotate(series[key]['name'], (np.sqrt(b0*b1), np.sqrt(y0*y1)), xytext=tuple(normal*(gap if side == 'above' else -gap)),
                    textcoords='offset points', ha='center', va='center', rotation=angle,
                    rotation_mode='anchor', fontsize=TEXT, color=series[key]['colour'], zorder=4)

    def line_value(key, x):
        # Log-log interpolation of a series between the measured batches bracketing x.
        pts = [(r['batch'], gigabytes(r)) for r in by_name[key]]
        for (b0, y0), (b1, y1) in zip(pts, pts[1:]):
            if b0 <= x <= b1:
                f = (np.log(x)-np.log(b0))/(np.log(b1)-np.log(b0))
                return np.exp(np.log(y0)+f*(np.log(y1)-np.log(y0)))
        return pts[0][1] if x < pts[0][0] else pts[-1][1]

    def place(key, b0, b1, side, gap=3.0):
        # Horizontal name centred between batches b0 and b1, its nearest edge `gap` points off the line.
        xc = np.sqrt(b0*b1)
        text = ax.text(xc, 1.0, series[key]['name'], ha='center', va='bottom' if side == 'above' else 'top',
                       fontsize=TEXT, color=series[key]['colour'], zorder=4)
        fig.canvas.draw()
        box = text.get_window_extent(fig.canvas.get_renderer())
        inverse = ax.transData.inverted()
        # The edge of the text nearest the line decides the clearance: for a line
        # rising to the right the right edge when above and the left edge when
        # below; for a flat line the centre is as good as any.
        x_left, x_right = inverse.transform((box.x0, 0))[0], inverse.transform((box.x1, 0))[0]
        probes = [line_value(key, v) for v in (x_left, xc, x_right)]
        y_line = ax.transData.transform((xc, max(probes) if side == 'above' else min(probes)))[1]
        y_text = y_line + (gap if side == 'above' else -gap)*fig.dpi/72
        text.set_position((xc, inverse.transform((0, y_text))[1]))

    # (series, from batch, to batch, side): each name sits beside its own line.
    NAMES = {'ours': (2, 8, 'below'), 'autodiff': (2, 4, 'above'), 'modal': (32, 64, 'below'),
             'autodiff_ckpt': (4, 16, 'above'), 'modal_ckpt': (16, 64, 'above')}
    if DECORATED:
        along('autodiff', 2, 4, 'above')
        along('modal', 16, 32, 'below')
        place('ours', 2, 8, 'below')
    else:
        for key, (b0, b1, side) in NAMES.items():
            if key in series:
                place(key, b0, b1, side)
else:
    handles = [Line2D([], [], marker='o', ls='none', color=s['colour'], ms=5, label=s['name']) for s in series.values()]
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
