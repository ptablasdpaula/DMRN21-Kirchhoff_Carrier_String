"""Check fixed-grid rate selection, synthesis parity, and notebook controls."""
from pathlib import Path
import base64
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
notebook = nbformat.read(ROOT / 'notebooks/inharmonicity_fixed_grid.ipynb', as_version=4)
notebook.cells.append(nbformat.v4.new_code_cell('''
# Independently verify the selected rate against the original spacing bound.
for f1, B in [(reference_f1, reference_B), (100., 0.), (100., .001), (1000., .01), (2000., .01)]:
    plan = sampling_plan(f1, B)
    p = physical_parameters(f1, B, reference.material_density, reference.radius, reference.length)
    c = math.sqrt(p['tension']/p['mass_per_length'])
    K = math.sqrt(p['young_modulus']*p['inertia']/p['mass_per_length'])
    h = p['length']/128
    minimum = reference.minimum_grid_spacing(1/plan['integration_rate'], c, K, reference.sigma_1)*1.01
    assert minimum <= h*(1+1e-14)
    assert plan['intervals'] == 128
    assert plan['stability_score'] <= 1+1e-14
    if plan['oversampling'] > 1:
        lower = reference.minimum_grid_spacing(1/(44100*(plan['oversampling']-1)), c, K, reference.sigma_1)*1.01
        assert lower > h
    print(f"f1={f1:.2f}, B={B:.6g}: ×{plan['oversampling']}, {plan['integration_rate']} Hz, N=128")

# Same fixed grid and rate must reproduce the original notebook's renderer.
import nbformat
original = nbformat.read(root/'notebooks/inharmonicity_synthesis.ipynb', as_version=4)
namespace = dict(globals())
exec(original.cells[2].source, namespace)
for f1, B in [(reference_f1, reference_B), (100., .001), (100., 0.)]:
    result = synthesise(f1, B, duration=.04, excitation='first_mode')
    expected = namespace['synthesise'](f1, B, duration=.04, excitation='first_mode',
                    oversampling=result['oversampling'], max_intervals=128)
    assert expected['intervals'] == result['intervals'] == 128
    assert result['audio'].numel() == 1764
    torch.testing.assert_close(result['audio'], expected['audio'], rtol=1e-12, atol=1e-14)
result = synthesise(1000., .01, duration=.01, amplitude=1e-6)
assert result['intervals'] == 128 and result['oversampling'] > 1
assert result['audio'].numel() == 441 and torch.isfinite(result['audio']).all()
for settings in [dict(intervals=127), dict(intervals=128.5), dict(sample_rate=0), dict(sigma_1=-1.)]:
    try: sampling_plan(100., .001, **settings)
    except ValueError: pass
    else: raise AssertionError(f'Accepted invalid settings: {settings}')
try: synthesise(1000., .01, max_oversampling=1, duration=.01)
except ValueError as error: assert 'Requires ×' in str(error)
else: raise AssertionError('Oversampling cap ignored')
try: synthesise(1000., .01, duration=60.)
except ValueError as error: assert '10 million' in str(error)
else: raise AssertionError('Step budget ignored')
controls['duration'].value=.04
controls['fundamental_hz'].value=100.
controls['inharmonicity'].value=.001
synthesise_button.click()
assert last_synthesis is not None and last_synthesis['intervals'] == 128
controls['fundamental_hz'].value=110.
assert last_synthesis is None
synthesise_button.click()
assert last_synthesis is not None and last_synthesis['oversampling'] > 1
assert not synthesise_button.disabled
print('Fixed grid, minimum factor, waveform parity, B=0, high-rate synthesis, resource limits and widgets passed.')
'''))
NotebookClient(notebook, timeout=180, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
output = ROOT / '.notebook-build/inharmonicity_fixed_grid.tested.ipynb'
output.parent.mkdir(exist_ok=True)
nbformat.write(notebook, output)
images = []
def walk(value):
    if isinstance(value, dict):
        if 'image/png' in value: images.append(value['image/png'])
        for item in value.values(): walk(item)
    elif isinstance(value, list):
        for item in value: walk(item)
walk(notebook)
assert images, 'Missing synthesis figure'
(ROOT / '.notebook-build/inharmonicity_fixed_grid_preview.png').write_bytes(base64.b64decode(images[-1]))
for cell in notebook.cells:
    for item in cell.get('outputs', []):
        if item.get('output_type') == 'stream': print(item['text'])
print('Passed:', output)
