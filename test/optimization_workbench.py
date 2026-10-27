"""Execute the interactive notebook and exercise its state/selection contracts."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
notebook=nbformat.read(ROOT/'notebooks/optimization_workbench.ipynb',as_version=4)
notebook.cells.append(nbformat.v4.new_code_cell('''
workbench.duration.value = .08
next(b for b in workbench.buttons if b.description == 'Generate target').click()
assert workbench.target is not None
next(b for b in workbench.buttons if b.description == 'Generate initial signal').click()
assert workbench.initial is not None
original_initial = workbench.initial['physical'].clone()
for i, box in enumerate(workbench.optimize_boxes):
    box.value = (i == 5)  # Optimise amplitude only.
for name in ('adam', 'lbfgs'):
    workbench.max_steps[name].value = 2
    next(b for b in workbench.buttons if b.description == f'Run {name.upper()} from initial signal').click()
    assert name in workbench.results, name
    result = workbench.results[name]
    frozen = torch.tensor([i != 5 for i in range(8)])
    assert torch.equal(result['best_physical'][frozen], original_initial[frozen])
    assert result['iterations'] == 2
    assert result['best_loss'] <= workbench.initial['loss']
assert torch.equal(workbench.initial['physical'], original_initial)
for box in workbench.optimize_boxes: box.value = False
workbench.run('adam')
assert workbench.results['adam']['iterations'] == 0
assert torch.equal(workbench.results['adam']['best_physical'], original_initial)
workbench.initial_fields[0].value *= 1.01
assert workbench.initial is None and workbench.target is not None and not workbench.results
workbench.sample_rate.value = 48000
assert workbench.target is None and workbench.initial is None
workbench.oversampling.value = 2
workbench.generate_target()
workbench.generate_initial()
assert workbench.target is not None and workbench.initial is not None
assert workbench.target['audio'].numel() == round(.08 * 48000)
x = (workbench.initial['physical'] / __import__('src.kc_string_forward', fromlist=['nominal_parameters']).nominal_parameters()).requires_grad_()
workbench.renderer(x).square().mean().backward()
assert torch.isfinite(x.grad).all()
# Zero damping is valid, including when selected for optimization.
workbench.target_fields[6].value = 0.
workbench.initial_fields[6].value = 0.
workbench.generate_target()
workbench.generate_initial()
workbench.optimize_boxes[6].value = True
workbench.run('adam')
assert 'adam' in workbench.results
assert workbench.results['adam']['best_physical'][6] >= 0
from src.optimization_workbench import fit_selected
for name in ('adam', 'lbfgs'):
    plateau = fit_selected(workbench.renderer, lambda audio: audio.sum()*0+1,
                           workbench.initial['physical'], [False]*5+[True,False,False],
                           name, .01, 30)
    assert plateau['iterations'] == 20
    assert 'Less than 10%' in plateau['stop_reason']
# Invalid grids are reported rather than run or silently altered.
workbench.intervals.value = 4095
workbench.generate_target()
assert workbench.target is None
assert workbench.intervals.value == 4095
print('Notebook workflow, frozen parameters, independent starts, invalidation, zero selection, oversampling, zero damping and invalid-grid checks passed.')
'''))
NotebookClient(notebook,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
path=ROOT/'.notebook-build/optimization_workbench.tested.ipynb'
nbformat.write(notebook,path)
print('Passed; executed notebook:',path)
