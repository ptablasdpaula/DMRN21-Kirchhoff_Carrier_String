"""CPU checks for experiments/applications/common: box transforms, decimation, the plateau optimiser and media."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiments.applications.common.boxes import (decode_batch, encode_rows, sample_logits,
                                                   largest_stable_intervals, f0_range)
from experiments.applications.common.optim import Schedule, fit_batch
from experiments.applications.common.render import decimate_batch, tail_fade, theta_from_physical
from experiments.applications.common.media import (read_wave, write_wave, spectrogram_db, FrameLogger,
                                                   animate, save_spectrogram_png)
from src.grid import GridConfig
from src.kc_string_forward import nominal_parameters
from src.parameters import MaterialParameterBox, BenchmarkMaterialBox
from src.synthesis import decimate, antialias_kernel

torch.manual_seed(0)


def check_boxes():
    box = MaterialParameterBox()
    logits = sample_logits(box, 64, seed=3)
    values = decode_batch(box, logits)
    low, high = box.bounds()
    assert values.shape == (64, 10) and bool(((values >= low) & (values <= high)).all())
    rows = torch.stack([box.decode(z) for z in logits])
    torch.testing.assert_close(values, rows, rtol=1e-12, atol=0)
    torch.testing.assert_close(decode_batch(box, encode_rows(box, values)), values, rtol=1e-9, atol=0)
    single = encode_rows(box, box.defaults())
    torch.testing.assert_close(decode_batch(box, single), box.defaults(), rtol=1e-9, atol=0)
    theta = theta_from_physical(values)
    torch.testing.assert_close(theta*nominal_parameters(), values, rtol=1e-12, atol=0)
    f_low, f_high = f0_range(box)
    nominal_f0 = float(torch.sqrt(torch.tensor(75/(8050*torch.pi*3.556e-4**2)))/2)
    assert f_low < nominal_f0 < f_high, (f_low, nominal_f0, f_high)
    bench = BenchmarkMaterialBox()
    n = largest_stable_intervals(bench, 2, sample_rate=44100)
    assert n is not None and n >= 176 and n % 2 == 0, n
    GridConfig(bench, oversampling=2, intervals=n)
    try:
        GridConfig(bench, oversampling=2, intervals=n+2)
    except ValueError:
        pass
    else:
        raise AssertionError('largest_stable_intervals did not return the largest stable count')
    print('boxes ok', {'f0_range_hz': (round(f_low, 1), round(f_high, 1)), 'benchmark_box_max_intervals_2x': n})


def check_decimation():
    raw = torch.randn(3, 5000, dtype=torch.float64)
    kernel = antialias_kernel(6)
    batched = decimate_batch(raw, 6, 800, kernel)
    for i in range(3):
        torch.testing.assert_close(batched[i], decimate(raw[i], 6, 800, kernel), rtol=0, atol=1e-13)
    window = tail_fade(1000, 100)
    assert window[0] == 1 and abs(float(window[-1])) < 1e-12 and window.shape == (1000,)
    print('decimation ok')


def check_optimiser():
    # Items 1-3: quadratic bowls with distinct minima; item 0: constant loss (zero gradient) to exercise the plateau logic.
    targets = torch.tensor([[0., 0.], [1., -1.], [2., 0.5], [-1.5, 1.]], dtype=torch.float64)
    logits = torch.zeros(4, 2, dtype=torch.float64)
    calls = []

    def objective(z, indices):
        calls.append(indices.tolist())
        loss = ((z-targets[indices])**2).sum(-1)
        constant = torch.ones_like(loss)
        is_zero = indices == 0
        return torch.where(is_zero, constant + 0*loss, loss), z.detach()

    schedule = Schedule(initial_lr=0.1, plateau_patience=3, stop_patience=10, maximum_updates=60, threshold=1e-3)
    seen = []
    result = fit_batch(logits, objective, schedule=schedule, callback=lambda s: seen.append(s['evaluation']))
    assert result.evaluations == len(calls) == len(seen)
    # Item 0: lr halves when the plateau counter exceeds 3 (evaluations 5 and 9), stops when the stop counter reaches 10 (evaluation 11).
    assert int(result.reductions[0]) == 2 and int(result.updates[0]) == 10, (result.reductions, result.updates)
    assert [e['update'] for e in result.events[0]] == [4, 8], result.events[0]
    assert abs(result.events[0][1]['new_lr']-0.025) < 1e-12
    assert bool((result.best_loss[1:] < 0.25*result.initial_loss[1:]).all()), result.best_loss   # bowls improve, then plateau-stop
    assert bool((result.best_loss[1:] < result.initial_loss[1:]).all())
    assert bool((result.best_loss <= result.final_loss + 1e-15).all())
    assert torch.allclose(result.best_logits[0], logits[0])     # constant loss: the initial point is retained
    improvement = ((result.best_logits[1:]-targets[1:]).norm(dim=1) < (logits[1:]-targets[1:]).norm(dim=1))
    assert bool(improvement.all())
    assert result.history[-1]['active'] == 0 and result.history[0]['evaluation'] == 1
    print('optimiser ok', {'evaluations': result.evaluations, 'updates': result.updates.tolist(),
                           'reductions': result.reductions.tolist()})


def check_media():
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        tone = 0.5*np.sin(2*np.pi*440*np.arange(16000)/16000)
        write_wave(folder/'tone.wav', tone, 16000)
        back, rate = read_wave(folder/'tone.wav')
        assert rate == 16000 and back.shape == tone.shape and np.abs(back-tone).max() < 1/32000
        spec = spectrogram_db(torch.tensor(tone), n_fft=512, hop=128)
        assert spec.shape[0] == 257 and float(spec.max()) == 0.0 and float(spec.min()) >= -80
        batch = spectrogram_db(torch.tensor(np.stack([tone, 0.1*tone])), n_fft=512, hop=128)
        assert batch.shape[0] == 2 and float(batch[1].max()) == 0.0   # each row relative to its own peak
        save_spectrogram_png(folder/'spec.png', [('a', spec), ('b', spec)], 16000, hop=128, fmax=4000)
        assert (folder/'spec.png').stat().st_size > 1000
        logger = FrameLogger(every=2, target=spec)
        for it in range(5):
            logger.log(it, spec*(1-it/10), f'iteration {it}', footer=f'f0 {440+it} Hz', loss=float(it))
        assert len(logger.frames) == 3
        logger.save(folder/'frames.npz', folder/'frames.jsonl')
        meta = [json.loads(l) for l in open(folder/'frames.jsonl')]
        assert [m['iteration'] for m in meta] == [0, 2, 4] and meta[1]['loss'] == 2.0
        gif = animate(folder/'frames.npz', folder/'frames.jsonl', folder/'anim.gif', sample_rate=16000, hop=128, fps=5, fmax=4000)
        from PIL import Image
        with Image.open(gif) as image:
            assert getattr(image, 'n_frames', 1) == 3, image.n_frames
    print('media ok')


if __name__ == '__main__':
    check_boxes(); check_decimation(); check_optimiser(); check_media()
    print('applications_common: all checks passed')
