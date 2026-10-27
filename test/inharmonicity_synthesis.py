"""Execute the synthesis-only notebook and check its physical reparameterisation."""
from pathlib import Path
import base64
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
notebook=nbformat.read(ROOT/'notebooks/inharmonicity_synthesis.ipynb',as_version=4)
notebook.cells.append(nbformat.v4.new_code_cell('''
from src.kc_string_forward import ForwardString
p = physical_parameters(reference_f1, reference_B, reference.material_density, reference.radius, reference.length)
assert math.isclose(p['tension'], reference.tension, rel_tol=1e-12)
assert math.isclose(p['young_modulus'], reference.young_modulus, rel_tol=1e-12)
for f1, B in [(76., 0.), (220., .001), (110., .01)]:
    p = physical_parameters(f1, B, reference.material_density, reference.radius, reference.length)
    recovered_B = math.pi**2*p['young_modulus']*p['inertia']/(p['tension']*p['length']**2)
    recovered_f1 = math.sqrt(p['tension']/p['mass_per_length'])/(2*p['length'])*math.sqrt(1+recovered_B)
    assert math.isclose(recovered_f1, f1, rel_tol=1e-12)
    assert math.isclose(recovered_B, B, rel_tol=1e-12, abs_tol=1e-15)
    assert partial_frequencies(f1, B)[0].item() == f1
    sound = synthesise(f1, B, duration=.08, excitation='multimode')
    assert torch.isfinite(sound['audio']).all()
    assert sound['audio'].numel() == 3528
    assert sound['intervals'] <= sound['maximum_stable_intervals']
    assert sound['intervals'] % 2 == 0
# At reference parameters and excitation this is the existing model's waveform.
sound = synthesise(reference_f1, reference_B, duration=.08, excitation='first_mode')
with torch.no_grad():
    expected = ForwardString(intervals=sound['intervals'])(torch.ones(8,dtype=torch.float64),.08)[:3528]
torch.testing.assert_close(sound['audio'],expected,rtol=1e-9,atol=1e-12)
resampled = synthesise(reference_f1,reference_B,duration=.08,oversampling=2)
assert resampled['audio'].numel() == 3528
assert torch.isfinite(resampled['audio']).all()
for f1, B in [(0., .001), (100., -.1)]:
    try: physical_parameters(f1,B,reference.material_density,reference.radius,reference.length)
    except ValueError: pass
    else: raise AssertionError('Invalid pitch or B accepted')
controls['duration'].value=.08
synthesise_button.click()
assert last_synthesis is not None
controls['fundamental_hz'].value=100.
assert last_synthesis is None
synthesise_button.click()
assert last_synthesis is not None
print('Mapping, B=0, original waveform, multimode synthesis, resampling and control checks passed.')
'''))
NotebookClient(notebook,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
output=ROOT/'.notebook-build/inharmonicity_synthesis.tested.ipynb'
nbformat.write(notebook,output)
images=[]
def walk(value):
    if isinstance(value,dict):
        if 'image/png' in value: images.append(value['image/png'])
        for item in value.values():walk(item)
    elif isinstance(value,list):
        for item in value:walk(item)
walk(notebook)
assert images,'Missing rendered synthesis figure'
(ROOT/'.notebook-build/inharmonicity_synthesis_preview.png').write_bytes(base64.b64decode(images[-1]))
print('Passed:',output)
