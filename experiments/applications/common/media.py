"""Audio files, spectrograms, frame logging and animated GIFs for the application pages."""
import json
import math
from pathlib import Path
import wave
import numpy as np
import torch

CMAP = 'magma'   # one perceptual sequential ramp for magnitude; no rainbow maps


def read_wave(path):
    """16-bit PCM WAV -> (float64 array in [-1, 1) with shape (samples,) or (channels, samples), rate)."""
    with wave.open(str(path), 'rb') as handle:
        channels, width, rate, frames = handle.getnchannels(), handle.getsampwidth(), handle.getframerate(), handle.getnframes()
        data = handle.readframes(frames)
    if width != 2:
        raise ValueError(f'{path}: expected 16-bit PCM, got {8*width}-bit.')
    audio = np.frombuffer(data, dtype='<i2').astype(np.float64)/32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).T
    return audio, rate


def write_wave(path, audio, sample_rate):
    """Write a mono float array in [-1, 1] as 16-bit PCM (values are clipped)."""
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(audio)):
        raise ValueError('audio must be finite')
    pcm = np.clip(np.round(audio*32767.0), -32768, 32767).astype('<i2')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())


def spectrogram_db(audio, n_fft=1024, hop=256, floor_db=-80.0, reference=None):
    """Log-magnitude STFT in dB relative to ``reference`` (default: the signal's own peak), clipped at ``floor_db``.

    ``audio`` is ``(samples,)`` or ``(batch, samples)``; the result is ``(..., frequency, frames)``.
    """
    audio = torch.as_tensor(audio, dtype=torch.float64)
    window = torch.hann_window(n_fft, periodic=True, dtype=torch.float64, device=audio.device)
    spectrum = torch.stft(audio, n_fft, hop_length=hop, win_length=n_fft, window=window, center=True,
                          pad_mode='reflect', return_complex=True)
    magnitude = spectrum.abs()
    peak = magnitude.amax(dim=(-2, -1), keepdim=True) if reference is None else torch.as_tensor(reference, dtype=torch.float64)
    db = 20*torch.log10(magnitude.clamp_min(1e-12)/peak.clamp_min(1e-12))
    return db.clamp_min(floor_db)


def _axes_extent(spec, sample_rate, hop):
    frames = spec.shape[-1]
    return (0.0, frames*hop/sample_rate, 0.0, sample_rate/2)


def save_spectrogram_png(path, panels, sample_rate, hop=256, fmax=None, vmin=-80.0, vmax=0.0, dpi=110, width=3.6, height=2.8):
    """Side-by-side spectrogram panels ``[(title, spec_db), ...]`` with a shared dB scale."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(panels), figsize=(width*len(panels), height), squeeze=False, constrained_layout=True)
    for ax, (title, spec) in zip(axes[0], panels):
        spec = np.asarray(spec.detach().cpu() if torch.is_tensor(spec) else spec, dtype=np.float64)
        image = ax.imshow(spec, origin='lower', aspect='auto', cmap=CMAP, vmin=vmin, vmax=vmax,
                          extent=_axes_extent(spec, sample_rate, hop), interpolation='nearest')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=9); ax.set_ylabel('Frequency (Hz)', fontsize=9)
        if fmax is not None:
            ax.set_ylim(0, fmax)
        ax.tick_params(labelsize=8, length=2)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    fig.colorbar(image, ax=axes[0].tolist(), label='dB', shrink=0.9, pad=0.01)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


class FrameLogger:
    """Collect spectrogram frames of the current best candidate every ``every`` iterations.

    Frames are kept on the host as float16 so a GIF can be built at the end of
    the run, or regenerated later from the saved ``frames.npz`` + ``frames.jsonl``.
    """
    def __init__(self, every=5, target=None):
        self.every = max(1, int(every))
        self.target = None if target is None else np.asarray(target.detach().cpu() if torch.is_tensor(target) else target, dtype=np.float16)
        self.frames, self.meta = [], []

    def due(self, iteration):
        return iteration % self.every == 0

    def log(self, iteration, spec, title, footer='', force=False, **extra):
        if not (force or self.due(iteration)):
            return False
        spec = spec.detach().to('cpu', torch.float16) if torch.is_tensor(spec) else torch.as_tensor(np.asarray(spec, dtype=np.float16))
        self.frames.append(spec.numpy())
        self.meta.append({'iteration': int(iteration), 'title': str(title), 'footer': str(footer), **extra})
        return True

    def save(self, npz_path, jsonl_path):
        Path(npz_path).parent.mkdir(parents=True, exist_ok=True)
        arrays = {'frames': np.stack(self.frames) if self.frames else np.zeros((0, 1, 1), np.float16),
                  'iterations': np.array([m['iteration'] for m in self.meta], dtype=np.int64)}
        if self.target is not None:
            arrays['target'] = self.target
        np.savez_compressed(npz_path, **arrays)
        with open(jsonl_path, 'w') as handle:
            for meta in self.meta:
                handle.write(json.dumps(meta)+'\n')


def animate(frames_npz, frames_jsonl, out_gif, *, sample_rate, hop=256, fps=12, dpi=72, fmax=None,
            vmin=-80.0, vmax=0.0, target_title='target', frame_title='current best', width=7.0, height=3.2):
    """Write a GIF from saved frames: target panel (if any) on the left, the logged best candidate on the right."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    data = np.load(frames_npz)
    frames = data['frames'].astype(np.float32)
    target = data['target'].astype(np.float32) if 'target' in data.files else None
    with open(frames_jsonl) as handle:
        meta = [json.loads(line) for line in handle if line.strip()]
    if len(frames) == 0 or len(meta) != len(frames):
        raise ValueError('frames and metadata do not match or are empty')
    panels = 2 if target is not None else 1
    fig, axes = plt.subplots(1, panels, figsize=(width, height), squeeze=False, constrained_layout=True)
    axes = axes[0]
    extent = _axes_extent(frames[0], sample_rate, hop)
    if target is not None:
        axes[0].imshow(target, origin='lower', aspect='auto', cmap=CMAP, vmin=vmin, vmax=vmax, extent=extent, interpolation='nearest')
        axes[0].set_title(target_title, fontsize=10)
    image = axes[-1].imshow(frames[0], origin='lower', aspect='auto', cmap=CMAP, vmin=vmin, vmax=vmax, extent=extent, interpolation='nearest')
    axes[-1].set_title(frame_title, fontsize=10)
    for ax in axes:
        ax.set_xlabel('Time (s)', fontsize=9); ax.tick_params(labelsize=8, length=2)
        if fmax is not None:
            ax.set_ylim(0, fmax)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    axes[0].set_ylabel('Frequency (Hz)', fontsize=9)
    title = fig.suptitle(meta[0]['title'], fontsize=10)
    footer = fig.text(0.5, 0.01, meta[0].get('footer', ''), ha='center', va='bottom', fontsize=8)

    def update(index):
        image.set_data(frames[index])
        title.set_text(meta[index]['title'])
        footer.set_text(meta[index].get('footer', ''))
        return image, title, footer

    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    Path(out_gif).parent.mkdir(parents=True, exist_ok=True)
    animation.save(str(out_gif), writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return Path(out_gif)
