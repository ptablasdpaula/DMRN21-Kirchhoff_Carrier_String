"""Execute both entry notebooks and exercise controls in a real kernel."""
from pathlib import Path
import base64
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
checks='''
assert workbench.coordinates.value=='material'
workbench.coordinates.value='perceptual'
workbench.duration.value=.04
workbench.prepare()
assert workbench.grid is not None
fixed_grid=workbench.grid
assert len(workbench.target_fields)==6
workbench.generate_target()
assert workbench.target is not None
workbench.fill_initial(True)
workbench.generate_initial()
assert workbench.initial is not None
for name in ('adam','lbfgs'):
    workbench.max_steps[name].value=2
    workbench.run(name)
    assert name in workbench.results
    assert workbench.results[name]['iterations']==2
    assert workbench.grid is fixed_grid
initial=workbench.initial['physical'].clone()
for result in workbench.results.values():
    torch.testing.assert_close(result['requested_initial'],initial)
workbench.target_fields[0].value=80.
assert workbench.target is None and workbench.grid is fixed_grid
assert not workbench.results
workbench.generate_target()
assert workbench.target is not None
# Range edits invalidate every dependent object immediately.
workbench.range_fields[0][1].value=110.
assert workbench.grid is None and workbench.renderer is None
assert workbench.target is None and workbench.initial is None
assert all(field.disabled for field in workbench.target_fields)
workbench.prepare()
assert workbench.grid is not None and workbench.grid is not fixed_grid
# Bad ranges do not leave a stale configuration usable.
workbench.range_fields[0][0].value=120.
workbench.prepare()
assert workbench.grid is None
workbench.range_fields[0][0].value=60.
workbench.prepare();workbench.generate_target();workbench.generate_initial()
workbench.max_steps['adam'].value=1
workbench.run('adam')
assert 'adam' in workbench.results
# Switch to materials: old results disappear and the direct controls replace them.
workbench.coordinates.value='material'
assert workbench.grid is None and not workbench.results
assert len(workbench.range_fields)==8
assert workbench.density.layout.display=='none'
workbench.prepare()
assert workbench.grid is not None and len(workbench.target_fields)==8
material_grid=workbench.grid
workbench.generate_target();workbench.fill_initial(True);workbench.generate_initial()
assert workbench.target is not None and workbench.initial is not None
saved_initial=workbench.initial['physical'].clone()
for checkbox in workbench.optimize_boxes:checkbox.value=False
workbench.optimize_boxes[1].value=True
workbench.optimize_boxes[5].value=True
for name in ('adam','lbfgs'):
    workbench.max_steps[name].value=2
    workbench.run(name)
    assert name in workbench.results
    result=workbench.results[name]
    torch.testing.assert_close(result['requested_initial'],saved_initial)
    frozen=torch.tensor([True,False,True,True,True,False,True,True])
    assert torch.equal(result['best_physical'][frozen],saved_initial[frozen])
    assert workbench.grid is material_grid
# Fitted length changes physical h continuously but never changes N or the rate.
workbench.target_fields[4].value=1.1
assert workbench.grid is material_grid and workbench.target is None
workbench.generate_target()
assert workbench.target is not None
workbench.coordinates.value='perceptual'
assert workbench.grid is None and len(workbench.range_fields)==6
workbench.prepare();workbench.generate_target();workbench.generate_initial()
assert workbench.target is not None
print('Both coordinate systems: ranges, native synthesis, independent fits, freezing, switching and plots passed.')
'''
for name in ('synthesis','optimization'):
    notebook=nbformat.read(ROOT/f'notebooks/{name}.ipynb',as_version=4)
    notebook.cells.append(nbformat.v4.new_code_cell(checks if name=='optimization' else '''
workbench.duration.value=.04
workbench.prepare();workbench.generate_target()
assert workbench.target is not None and workbench.grid is not None
'''))
    NotebookClient(notebook,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    output=ROOT/f'.notebook-build/{name}.tested.ipynb';output.parent.mkdir(exist_ok=True)
    nbformat.write(notebook,output)
    images=[]
    def walk(value):
        if isinstance(value,dict):
            if 'image/png' in value:images.append(value['image/png'])
            for item in value.values():walk(item)
        elif isinstance(value,list):
            for item in value:walk(item)
    walk(notebook)
    assert images,'Missing rendered figure'
    (ROOT/f'.notebook-build/{name}_preview.png').write_bytes(base64.b64decode(images[-1]))
    print('Passed:',output)
