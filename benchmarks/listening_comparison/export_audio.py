"""Export the recorded batch-one benchmark waveforms for listening."""
from pathlib import Path
import wave
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/data/scratch/acw794/kc-string-gpu/native-rate-pilot-34555111-logs/diaz-44100-m175-b1.npz')


def write_pcm24(path, samples, channels=1):
    # Encode little-endian signed PCM24 using a single common gain.
    q = np.rint(np.clip(samples, -1, 1)*(2**23-1)).astype(np.int32).reshape(-1)
    b = np.empty((q.size, 3), dtype=np.uint8)
    b[:, 0] = q & 255
    b[:, 1] = (q >> 8) & 255
    b[:, 2] = (q >> 16) & 255
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(channels)
        w.setsampwidth(3)
        w.setframerate(44100)
        w.writeframes(b.tobytes())


def main():
    with np.load(SOURCE) as data:
        fdtd = data['reference_audio'].copy()
        jax = data['audio'].copy()
        rate, modes = int(data['rate']), int(data['modes'])
    assert rate == 44100 and modes == 175 and fdtd.shape == jax.shape == (4*rate,)
    assert np.isfinite(fdtd).all() and np.isfinite(jax).all()
    peak = max(float(np.max(np.abs(fdtd))), float(np.max(np.abs(jax))))
    gain = .9/peak
    fdtd_scaled, jax_scaled = gain*fdtd, gain*jax
    write_pcm24(ROOT/'01-fdtd.wav', fdtd_scaled)
    write_pcm24(ROOT/'02-jax-175.wav', jax_scaled)
    sequential = np.concatenate((fdtd_scaled, np.zeros(rate), jax_scaled))
    write_pcm24(ROOT/'fdtd-then-jax.wav', np.repeat(sequential[:,None], 2, axis=1), 2)
    print('common gain:', gain)
    print('original peaks FDTD/JAX:', np.max(np.abs(fdtd)), np.max(np.abs(jax)))
    print('original RMS FDTD/JAX:', np.sqrt(np.mean(fdtd**2)), np.sqrt(np.mean(jax**2)))
    print('source waveform relative L2:', np.linalg.norm(fdtd-jax)/np.linalg.norm(fdtd))


if __name__ == '__main__':
    main()
