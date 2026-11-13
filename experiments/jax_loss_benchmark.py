"""JAX reference for the project's default anti-alias filter and bound CeL.

This deliberately covers the fixed default loss only: 1024-point uncentered
periodic-Hann STFT, hop 256, four diagonal cumulative surfaces, no logarithmic
weighting or decay, and epsilon 1e-12. The target features are computed once,
as in ``STFTCumulativeEnergyLoss.bind``. Inputs may have leading batch axes;
``jax_loss`` returns the mean loss across those axes.
"""

import jax
import jax.numpy as jnp
from jax import lax


def anti_alias(wave, kernel, output_samples, oversampling=3):
    """Apply Renderer._filter_audio's edge padding and correlation in FP64."""
    if oversampling == 1:
        return wave[..., :output_samples]
    if kernel.ndim != 1 or kernel.shape[0] % 2 != 1:
        raise ValueError('Expected an odd, one-dimensional filter kernel')
    half_width = (kernel.shape[0] - 1) // 2
    leading = wave.shape[:-1]
    flat = wave.reshape((-1, wave.shape[-1]))
    padded = jnp.pad(flat, ((0, 0), (half_width, half_width)), mode='edge')
    result = lax.conv_general_dilated(
        padded[..., None], kernel[:, None, None], (oversampling,), 'VALID',
        dimension_numbers=('NWC', 'WIO', 'NWC'))[..., 0]
    return result.reshape((*leading, result.shape[-1]))[..., :output_samples]


def _power(audio):
    n_fft, hop = 1024, 256
    if audio.shape[-1] < n_fft:
        raise ValueError('Default CeL requires at least 1024 samples')
    count = 1 + (audio.shape[-1] - n_fft) // hop
    indices = jnp.arange(count)[:, None] * hop + jnp.arange(n_fft)[None, :]
    frames = audio[..., indices]
    index = jnp.arange(n_fft, dtype=audio.dtype)
    window = .5 - .5 * jnp.cos(2 * jnp.pi * index / n_fft)
    spectrum = jnp.fft.rfft(frames * window, axis=-1)
    power = jnp.real(spectrum) ** 2 + jnp.imag(spectrum) ** 2
    return jnp.swapaxes(power, -1, -2)


def _surface(energy, reverse_frequency, reverse_time):
    result = jnp.flip(energy, axis=-1) if reverse_time else energy
    result = jnp.cumsum(result, axis=-1)
    result = jnp.flip(result, axis=-1) if reverse_time else result
    result = jnp.flip(result, axis=-2) if reverse_frequency else result
    result = jnp.cumsum(result, axis=-2)
    return jnp.flip(result, axis=-2) if reverse_frequency else result


def _surfaces(energy):
    return jnp.stack([_surface(energy, rf, rt) for rf, rt in
                      ((False, False), (True, False),
                       (False, True), (True, True))], axis=-3)


@jax.custom_jvp
def _safe_norm(x):
    # torch.linalg.vector_norm returns a zero gradient at an all-zero input.
    return jnp.sqrt(jnp.sum(x * x, axis=(-2, -1)))


@_safe_norm.defjvp
def _safe_norm_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    norm = _safe_norm(x)
    denominator = jnp.where(norm > 0, norm, 1.)
    derivative = jnp.sum(x * dx, axis=(-2, -1)) / denominator
    return norm, derivative


def prepare_target(target):
    """Cache target surfaces and total mass, preserving output sample shape."""
    power = _power(target)
    mass = jnp.sum(power, axis=(-2, -1), keepdims=True)
    reference = jnp.sqrt(jnp.maximum(_surfaces(power) / mass[..., None, :, :], 1e-12))
    return reference, mass, target


def cel(audio, prepared):
    """Bound default CeL (mean across directions and batch items)."""
    reference, mass, _ = prepared
    power = _power(audio)
    feature = jnp.sqrt(jnp.maximum(_surfaces(power) / mass[..., None, :, :], 1e-12))
    error = feature - reference
    return jnp.mean(_safe_norm(error) / jnp.sqrt(error.shape[-2] * error.shape[-1]))


def prepare_mae_target(target):
    """The waveform MAE needs no cached features: the target itself is the preparation."""
    return target


def mae(audio, target):
    """Mean over items of the per-item mean |audio - target|, the JAX twin of ``WaveformMAE``."""
    return jnp.mean(jnp.mean(jnp.abs(audio - target), axis=-1))


def jax_loss(wave, prepared, filter_kernel, oversampling=3):
    """Filter internal-rate audio and calculate cached-target CeL."""
    output_samples = prepared[2].shape[-1]
    return cel(anti_alias(wave, filter_kernel, output_samples, oversampling), prepared)
