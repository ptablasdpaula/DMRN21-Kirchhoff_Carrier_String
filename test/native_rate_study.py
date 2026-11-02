"""Native-rate upstream adapter keeps the same physical excitation."""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('JAX_ENABLE_X64','true')
import jax
import jax.numpy as jnp
import numpy as np
from benchmarks.native_rate_compare import compare_artifacts, comparison_pairs
from experiments.figure_comparison.diaz import initial_shape, renderer as old_renderer
from experiments.figure_comparison.physical import physical_values
from experiments.native_rate_study.renderer import renderer
from src.kc_string_forward import excitation_shape


def main():
    x=jnp.asarray(physical_values(1).numpy())
    n=80;rate=44100
    original=old_renderer(x,n-2,176,rate)
    full=renderer(x,n,rate,175,176)
    np.testing.assert_allclose(full,original,rtol=1e-11,atol=1e-13)
    weights=jnp.sin(jnp.arange(n,dtype=jnp.float64)/7.)
    baseline=jax.value_and_grad(lambda y:jnp.sum(renderer(y,n,rate,175,176)*weights))(x)
    scales=np.asarray(x)
    for modes in (8,123):
        observed=jax.value_and_grad(lambda y:jnp.sum(renderer(y,n,rate,modes,176)*weights))(x)
        np.testing.assert_allclose(observed[0],baseline[0],rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(np.asarray(observed[1])*scales,np.asarray(baseline[1])*scales,rtol=1e-8,atol=1e-11)
    # The JAX initial profile is the FDTD solver's profile, for every kind.
    for excitation in ('first_mode','multimode','raised_cosine','raised_cosine:0.2'):
        np.testing.assert_allclose(np.asarray(initial_shape(176,excitation)),
                                   excitation_shape(176,excitation).numpy(),rtol=0,atol=1e-15)
    # A raised cosine excites every mode: 100 and 123 modes differ, both finite, and the
    # old and new adapters agree at the full grid count.
    raised=old_renderer(x,n-2,176,rate,'raised_cosine')
    np.testing.assert_allclose(renderer(x,n,rate,175,176,'raised_cosine'),raised,rtol=1e-11,atol=1e-13)
    outputs={}
    for modes in (100,123):
        value,grad=jax.value_and_grad(lambda y:jnp.sum(renderer(y,n,rate,modes,176,'raised_cosine')*weights))(x)
        assert np.isfinite(float(value)) and np.isfinite(np.asarray(grad)).all()
        outputs[modes]=np.asarray(renderer(x,n,rate,modes,176,'raised_cosine'))
    assert not np.allclose(outputs[100],outputs[123],rtol=1e-9,atol=0)
    # The checkpointed recurrence is the upstream recurrence: same values and
    # gradients for full segments only, full segments plus a remainder, and a
    # remainder alone (segment longer than the run).
    n=300
    plain=renderer(x,n,rate,100,176,'raised_cosine')
    objective=lambda y,ck:jnp.sum(renderer(y,n,rate,100,176,'raised_cosine',ck)*jnp.sin(jnp.arange(n,dtype=jnp.float64)/7.))
    base_value,base_grad=jax.value_and_grad(lambda y:objective(y,None))(x)
    for segment in (13,64,256,600):
        chunked=renderer(x,n,rate,100,176,'raised_cosine',segment)
        np.testing.assert_allclose(chunked,plain,rtol=1e-13,atol=1e-15)
        value,grad=jax.value_and_grad(lambda y:objective(y,segment))(x)
        np.testing.assert_allclose(float(value),float(base_value),rtol=1e-12,atol=0)
        np.testing.assert_allclose(np.asarray(grad)*scales,np.asarray(base_grad)*scales,rtol=1e-10,atol=1e-13)
    for bad in (0,-1,2.0,True):
        try: renderer(x,n,rate,100,176,'raised_cosine',bad)
        except ValueError: pass
        else: raise AssertionError(f'checkpoint_steps={bad!r} accepted')
    assert [p[2] for p in comparison_pairs([100,123],88200)]==['123 vs 100 modes at 44.1 kHz','44.1 vs 88.2 kHz, 100 modes']
    assert comparison_pairs([100],88200)[0][1]=='diaz-88200-m100-b1.npz'
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmp:
        checks=compare_artifacts(Path(tmp))
        assert len(checks)==3 and all(c['status']=='unavailable' for c in checks)
        checks=compare_artifacts(Path(tmp),[100,123],88200)
        assert len(checks)==2 and all(c['status']=='unavailable' for c in checks)
    print('PASS: native 175 matches upstream adapter; 8/123 modes preserve the fixed eight-mode excitation; '
          'raised-cosine profiles match the FDTD solver and run at 100/123 modes; '
          'the checkpointed recurrence reproduces upstream values and gradients')


if __name__=='__main__':main()
