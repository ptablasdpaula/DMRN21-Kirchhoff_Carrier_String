"""Build the paper's memory and time table: five implementations of one gradient, each adding one ingredient.

Usage: build_benchmark_summary_table.py [draft] [two-frameworks]

Rows, from the merged sweep data:
    Autodiff        fdtd                 eager PyTorch autodiff of the scheme on the grid
    + compiler      fdtd_jax             the same arithmetic compiled by JAX (jax.jit), every step kept
    + checkpoints   fdtd_jax_ckpt        the same with a checkpoint every 256 steps
    + modes         fdtd_jax_sine_ckpt   the same stepped in the discrete sine basis (mode by mode)
    Hand-written    replay               the hand-derived adjoint, same checkpoints and modes, in one CUDA kernel
The three middle rows are prefixed with the paper's \\step arrow: each builds on the row above.

The preferred source is native-rate-one-host-merged.json (every row measured on one host type, because
the eager and the host-driven compiled loops are bound by the host CPU); until it exists the script falls
back to native-rate-sweep-merged.json. With ``draft``, a row that has not been measured yet is filled
with question marks so that the page layout can be checked; without it a missing row is an error.

With ``two-frameworks`` the script writes benchmark-table-two-frameworks.tex instead, a test layout that
keeps each framework's rows together. A narrow first column names the framework beside its rows, set
sideways (the table needs \\usepackage{multirow}; a variant with the frameworks' logos was tried and dropped
as less clear):
    PyTorch   Auto              fdtd                      eager autodiff
              + compiler        fdtd_compiled             torch.compile blocks, every step kept
              + checkpoints     fdtd_ckpt_compiled        the same with a checkpoint every 256 steps
              + modes           fdtd_sine_ckpt_compiled   the same stepped in the discrete sine basis
    JAX       Auto (compiled)   fdtd_jax                  jax.jit (JAX always compiles), every step kept
              + checkpoints     fdtd_jax_ckpt
              + modes           fdtd_jax_sine_ckpt
    Hand-written                replay
Its source is native-rate-two-frameworks-merged.json; with ``draft`` and no such file it falls back to
the one-host data and question marks. With ``draft``, further arguments ending in .json are sweep files
whose rows are added as they are (a sweep that is still running, say), and a batch that has not been
measured is a question mark.

The five-model version of this table (Auto, Modal and the cached adjoint) is in commit 2d5372d.
"""
import json
from pathlib import Path
import sys

DRAFT = 'draft' in sys.argv[1:]
TWO = 'two-frameworks' in sys.argv[1:]
folder = Path(__file__).resolve().parent
one_host = folder / 'native-rate-one-host-merged.json'
source = one_host if one_host.exists() else folder / 'native-rate-sweep-merged.json'
two_frameworks = folder / 'native-rate-two-frameworks-merged.json'
if TWO and two_frameworks.exists():
    source = two_frameworks
elif TWO:
    assert DRAFT, 'native-rate-two-frameworks-merged.json is missing; pass draft to lay the table out without it'
data = json.loads(source.read_text())
extra = [Path(arg) for arg in sys.argv[1:] if arg.endswith('.json')]
assert DRAFT or not extra, 'extra sweep files are for draft tables only'
for path in extra:
    data['rows'] += json.loads(path.read_text())['rows']
meta = data['metadata']
assert meta['status'] == 'complete' and not meta.get('source_changed_during_run')
batches = meta['batch_sizes']
# (group label, rows): in the two-framework layout the label spans its rows in a narrow first column.
PYTORCH = r'\rotatebox[origin=c]{90}{\scriptsize PyTorch}'
JAX = r'\rotatebox[origin=c]{90}{\scriptsize JAX}'
if TWO:
    groups = [(PYTORCH, [('Auto', 'fdtd'),
                         (r'\step\,+ compiler', 'fdtd_compiled'),
                         (r'\step\,+ checkpoints', 'fdtd_ckpt_compiled'),
                         (r'\step\,+ modes', 'fdtd_sine_ckpt_compiled')]),
              (JAX, [('Auto (compiled)', 'fdtd_jax'),
                     (r'\step\,+ checkpoints', 'fdtd_jax_ckpt'),
                     (r'\step\,+ modes', 'fdtd_jax_sine_ckpt')]),
              (None, [('Hand-written', 'replay')])]
else:
    groups = [(None, [('Autodiff', 'fdtd'),
                      (r'\step\,+ compiler', 'fdtd_jax'),
                      (r'\step\,+ checkpoints', 'fdtd_jax_ckpt'),
                      (r'\step\,+ modes', 'fdtd_jax_sine_ckpt'),
                      ('Hand-written', 'replay')])]
models = [(group, len(rows), index == 0, name, method)
          for group, rows in groups for index, (name, method) in enumerate(rows)]
lead = r'\multicolumn{2}{@{}l}{} & ' if TWO else ' & '
lines = [r'\begin{tabular}{@{}c@{\hspace{3pt}}l*{8}{c}}' if TWO else r'\begin{tabular}{@{}l*{8}{c}}',
         lead + r'\multicolumn{8}{c}{Batch Size}\\',
         (r'\multicolumn{2}{@{}l}{Method} & ' if TWO else 'Method & ') + ' & '.join(map(str, batches)) + r'\\', r'\hline']
for label, field, factor in [('Peak memory (GB)', 'peak_gpu_bytes', 1e-9),
                              ('Iteration time (s)', 'median_seconds', 1)]:
    lines += [r'\noalign{\vskip 2pt}',
              lead + r'\multicolumn{8}{c}{' + label + r'}\\[2pt]']
    block = []
    for group, size, first, name, method in models:
        rows = [r for r in data['rows'] if r['method'] == method]
        if not rows and DRAFT:
            block.append((group, size, first, name, ['?'] * len(batches)))
            continue
        assert DRAFT or (len(rows) == len(batches) and {r['batch'] for r in rows} == set(batches)), method
        cells = []
        for batch in batches:
            row = next((r for r in rows if r['batch'] == batch), None)
            if row is None:
                cells.append('?')
            elif row['status'] != 'ok':
                assert row['status'] == 'oom', (method, batch, row['status'])
                cells.append('-')
            else:
                value = row[field] * factor
                cells.append(f'{value:.1f}' if factor != 1 else
                             (f'{value:.2f}' if value < 10 else f'{value:.1f}' if value < 100 else f'{value:.0f}'))
        block.append((group, size, first, name, cells))
    measured = lambda cell: cell not in ('-', '?')
    # Compare models within each batch, retaining ties at displayed precision.
    minima = [min(float(row[-1][i]) for row in block if measured(row[-1][i]))
              for i in range(len(batches))]
    # Equal colour steps between distinct displayed values, separately per metric.
    ranked = sorted({float(cell) for row in block for cell in row[-1] if measured(cell)})
    shades = {value: 100 * rank / (len(ranked) - 1) if len(ranked) > 1 else 0
              for rank, value in enumerate(ranked)}
    for position, (group, size, first, name, cells) in enumerate(block):
        formatted = []
        for i, cell in enumerate(cells):
            if cell == '-':
                formatted.append(r'\cellcolor{heatpurple}\textemdash')
                continue
            if cell == '?':
                formatted.append('?')
                continue
            value = float(cell)
            body = r'\textbf{' + cell + '}' if value == minima[i] else cell
            shade = shades[value]
            colour = (r'\cellcolor{heatpurple!' + f'{shade:.2f}' + '}'
                      if shade > 0 else '')
            formatted.append(colour + body)
        if not TWO:
            head = name
        elif group is None:
            head = r'\multicolumn{2}{@{}l}{' + name + '}'
        else:
            head = (r'\multirow{' + str(size) + '}{*}{' + group + '}' if first else '') + ' & ' + name
        if TWO and first and position:
            lines.append(r'\noalign{\vskip 1.5pt}')
        lines.append(head + ' & ' + ' & '.join(formatted) + r'\\')
    lines.append((r'\noalign{\vskip 2pt}\cline{' + ('3-10' if TWO else '2-9') + '}') if factor != 1 else r'\hline')
lines.append(r'\end{tabular}')
(folder / ('benchmark-table-two-frameworks.tex' if TWO else 'benchmark-table.tex')).write_text('% Generated by build_benchmark_summary_table.py from ' + source.name
                                            + (' (DRAFT: unmeasured cells are question marks'
                                               + ''.join(f'; with {p.name}' for p in extra) + ')' if DRAFT else '') + '\n'
                                            + '\n'.join(lines) + '\n')
print('Wrote', folder / ('benchmark-table-two-frameworks.tex' if TWO else 'benchmark-table.tex'), 'from', source.name,
      '(draft)' if DRAFT else '')
