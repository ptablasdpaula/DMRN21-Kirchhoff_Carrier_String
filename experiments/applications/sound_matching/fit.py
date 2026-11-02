"""Fit recorded NSynth guitar notes with the cached-adjoint FDTD string.

Stage A: many quasi-random starts over a whole string family on a coarse
grid (the family's highest f0 limits the stable interval count). Stage B: the
best distinct candidates are refined inside a narrow box around the winner on
a grid with as many intervals as that box allows. Nothing about the note's
pitch label enters either stage; it is only reported afterwards.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiments.applications.common.boxes import decode_batch, encode_rows, sample_logits
from experiments.applications.common.optim import Schedule, fit_batch
from experiments.applications.common.render import BatchRenderer
from experiments.applications.common.media import read_wave, write_wave, spectrogram_db, FrameLogger, animate, save_spectrogram_png
from experiments.applications.sound_matching.boxes import (family_box, refinement_box, stage_a_grid, choose_grid,
                                                           EXCITATION, SAMPLE_RATE, FAMILIES)
from experiments.applications.sound_matching.loss import make_criterion, per_item_loss, unit_rms, log_spectral_distance

STAGE_A = Schedule()
STAGE_B = Schedule(initial_lr=0.02, plateau_patience=10, stop_patience=40, maximum_updates=150)
TOP_K, DISTINCT_CENTS = 8, 10.0
SPEC_FFT, SPEC_HOP = 1024, 256


def derived(physical):
    """f0 (tension-only), inharmonicity B, f1, mass per length for ``(..., 10)`` physical values."""
    rho, tension, radius, modulus, length = physical[..., :5].unbind(-1)
    mu = rho*math.pi*radius**2
    f0 = torch.sqrt(tension/mu)/(2*length)
    B = math.pi**2*modulus*(math.pi*radius**4/4)/(tension*length**2)
    return {'f0_hz': f0, 'B': B, 'f1_hz': f0*torch.sqrt(1+B), 'mu_kg_per_m': mu}


def cents(f, reference):
    return 1200*math.log2(f/reference)


def footer_text(physical):
    rho, T, r, E, L, a, s0, s1, pluck, pickup = [float(v) for v in physical]
    d = derived(physical)
    return (f"T {T:.1f} N · L {L*100:.1f} cm · r {r*1e3:.3f} mm · ρ {rho:.0f} · E {E/1e9:.1f} GPa · "
            f"B {float(d['B']):.2e} · a {a*1e3:.2f} mm · σ₀ {s0:.2f} · σ₁ {s1:.1e} · pluck {pluck:.2f} · pickup {pickup:.2f}")


def run_stage(name, box, grid, target, logits0, schedule, *, duration, excitation, device, frames, frame_every,
              target_spec, log_handle, label_f0=None):
    renderer = BatchRenderer(grid, duration, excitation, device=device)
    criterion = make_criterion(grid.sample_rate)
    target_t = torch.as_tensor(target, dtype=torch.float64, device=device)
    peak_ref = spectrogram_db(unit_rms(target_t), SPEC_FFT, SPEC_HOP)  # only to share the reference; spectrogram_db re-normalises
    state = {'evaluation': 0}

    def objective(z, indices):
        physical = decode_batch(box, z)
        audio = renderer(physical)
        losses = per_item_loss(criterion, audio, target_t)
        return losses, (audio.detach(), physical.detach())

    def callback(info):
        audio, physical = info['aux']
        losses = info['losses']
        k = int(losses.argmin())
        best_physical = physical[k]
        d = derived(best_physical)
        row = {'stage': name, 'evaluation': info['evaluation'], 'loss_min': float(losses.min()),
               'loss_median': float(losses.median()), 'best_overall': float(info['strict_best'].min()),
               'active': int(info['active'].sum()), 'lr_best': float(info['lr'][info['indices'][k]]),
               'f0_hz': float(d['f0_hz']), 'B': float(d['B']), 'seconds': time.perf_counter()-start}
        log_handle.write(json.dumps(row)+'\n'); log_handle.flush()
        iteration = frames_offset + info['evaluation'] - 1
        if frames is not None and (frames.due(iteration) or info['evaluation'] == 1):
            spec = spectrogram_db(unit_rms(audio[k:k+1]), SPEC_FFT, SPEC_HOP)[0]
            title = f"stage {name} ({grid.intervals} intervals, {grid.integration_rate/1000:.0f} kHz) · iteration {iteration} · CeL {float(losses[k]):.4f} · f₀ {float(d['f0_hz']):.1f} Hz"
            if label_f0:
                title += f" (label {label_f0:.1f} Hz)"
            frames.log(iteration, spec, title, footer_text(best_physical), force=True, loss=float(losses[k]),
                       f0_hz=float(d['f0_hz']), stage=name)

    frames_offset = frames.meta[-1]['iteration']+1 if (frames is not None and frames.meta) else 0
    start = time.perf_counter()
    result = fit_batch(logits0, objective, schedule=schedule, callback=callback)
    physical = decode_batch(box, result.best_logits)
    return {'result': result, 'physical': physical, 'renderer': renderer, 'grid': grid, 'box': box, 'criterion': criterion}


def distinct_top(physical, losses, k=TOP_K, min_cents=DISTINCT_CENTS):
    """Indices of the k lowest-loss items whose f0 differ pairwise by more than ``min_cents``."""
    f0 = derived(physical)['f0_hz']
    order = torch.argsort(losses)
    chosen = []
    for i in order.tolist():
        if all(abs(cents(float(f0[i]), float(f0[j]))) > min_cents for j in chosen):
            chosen.append(i)
        if len(chosen) == k:
            break
    return chosen


def fit_note(target, family, *, starts=128, seed=0, device='cuda', duration=3.0, excitation=EXCITATION,
             schedule_a=STAGE_A, schedule_b=STAGE_B, output_dir=None, label_f0=None, frames_per_stage=90):
    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    log_handle = open(output_dir/'history.jsonl', 'w') if output_dir is not None else open('/dev/null', 'w')
    target_t = torch.as_tensor(target, dtype=torch.float64)
    target_spec = spectrogram_db(unit_rms(target_t), SPEC_FFT, SPEC_HOP)
    frames = FrameLogger(every=max(1, math.ceil(schedule_a.maximum_updates/frames_per_stage)), target=target_spec)
    # Stage A: the whole family on its coarse grid.
    box_a, grid_a = family_box(family), stage_a_grid(family)
    logits_a = sample_logits(box_a, starts, seed=seed, device=device)
    stage_a = run_stage('A', box_a, grid_a, target, logits_a, schedule_a, duration=duration, excitation=excitation,
                        device=device, frames=frames, frame_every=frames.every, target_spec=target_spec,
                        log_handle=log_handle, label_f0=label_f0)
    top = distinct_top(stage_a['physical'], stage_a['result'].best_loss)
    center = stage_a['physical'][top[0]]
    # Stage B: a narrow box around the winner, many more intervals.
    box_b = refinement_box(center, family)
    grid_b = choose_grid(box_b)
    frames.every = max(1, math.ceil(schedule_b.maximum_updates/50))
    logits_b = encode_rows(box_b, stage_a['physical'][top].to(device))
    stage_b = run_stage('B', box_b, grid_b, target, logits_b, schedule_b, duration=duration, excitation=excitation,
                        device=device, frames=frames, frame_every=frames.every, target_spec=target_spec,
                        log_handle=log_handle, label_f0=label_f0)
    log_handle.close()
    best = int(stage_b['result'].best_loss.argmin())
    physical = stage_b['physical'][best]
    with torch.no_grad():
        audio = stage_b['renderer'](physical[None])[0]
        final_loss = float(per_item_loss(stage_b['criterion'], audio[None], torch.as_tensor(target, device=device))[0])
    audio_cpu = unit_rms(audio.cpu()[None])[0]
    target_unit = unit_rms(target_t[None])[0]
    d = derived(physical)
    report = {
        'family': family, 'duration_s': duration, 'excitation': excitation, 'starts': starts, 'seed': seed,
        'stages': {
            'A': {'grid': stage_a['renderer'].describe(), 'box': stage_a['box'].ranges, 'schedule': schedule_a.__dict__,
                  'evaluations': stage_a['result'].evaluations, 'seconds': stage_a['result'].seconds,
                  'best_loss': float(stage_a['result'].best_loss.min()), 'top_items': top,
                  'top_f0_hz': derived(stage_a['physical'][top])['f0_hz'].tolist(),
                  'top_loss': stage_a['result'].best_loss[top].tolist()},
            'B': {'grid': stage_b['renderer'].describe(), 'box': stage_b['box'].ranges, 'schedule': schedule_b.__dict__,
                  'evaluations': stage_b['result'].evaluations, 'seconds': stage_b['result'].seconds,
                  'best_loss': float(stage_b['result'].best_loss.min()), 'all_loss': stage_b['result'].best_loss.tolist(),
                  'all_f0_hz': derived(stage_b['physical'])['f0_hz'].tolist(),
                  'lr_events': stage_b['result'].events}},
        'physical': {name: float(v) for name, v in zip(box_b.names, physical.tolist())},
        'derived': {k: float(v) for k, v in d.items()},
        'loss': final_loss,
        'log_spectral_distance_db': log_spectral_distance(audio_cpu, target_unit),
        'label_f0_hz': label_f0,
        'f1_cents_vs_label': cents(float(d['f1_hz']), label_f0) if label_f0 else None,
    }
    if output_dir is not None:
        scale = 0.9/max(float(audio_cpu.abs().max()), float(target_unit.abs().max()))
        write_wave(output_dir/'fit.wav', (audio_cpu*scale).numpy(), SAMPLE_RATE)
        write_wave(output_dir/'target_matched_level.wav', (target_unit*scale).numpy(), SAMPLE_RATE)
        fit_spec = spectrogram_db(audio_cpu, SPEC_FFT, SPEC_HOP)
        save_spectrogram_png(output_dir/'spectrograms.png', [('target', target_spec), (f'fit ({family})', fit_spec)],
                             SAMPLE_RATE, SPEC_HOP, fmax=SAMPLE_RATE/2)
        frames.save(output_dir/'frames.npz', output_dir/'frames.jsonl')
        try:
            animate(output_dir/'frames.npz', output_dir/'frames.jsonl', output_dir/'fit.gif', sample_rate=SAMPLE_RATE,
                    hop=SPEC_HOP, fps=12, fmax=SAMPLE_RATE/2, frame_title=f'current best ({family})')
        except Exception as error:   # the GIF is a convenience; never lose the fit over it
            report['gif_error'] = repr(error)
        with open(output_dir/'fit.json', 'w') as handle:
            json.dump(report, handle, indent=1)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--note', action='append', default=None, help='note_str to fit (repeatable); default: all')
    parser.add_argument('--families', default='steel,nylon')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--starts', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--duration', type=float, default=3.0)
    parser.add_argument('--excitation', default=EXCITATION)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true', help='0.5 s, 8 starts, a handful of iterations per stage')
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    with open(args.manifest) as handle:
        manifest = json.load(handle)
    notes = [n for n in manifest['notes'] if args.note is None or n['note_str'] in args.note]
    if not notes:
        raise SystemExit('no matching notes in the manifest')
    schedule_a, schedule_b, starts, duration = STAGE_A, STAGE_B, args.starts, args.duration
    if args.smoke:
        schedule_a = Schedule(initial_lr=0.05, plateau_patience=2, stop_patience=3, maximum_updates=4)
        schedule_b = Schedule(initial_lr=0.02, plateau_patience=2, stop_patience=3, maximum_updates=3)
        starts, duration = min(starts, 8), min(duration, 0.5)
    args.output.mkdir(parents=True, exist_ok=True)
    for note in notes:
        target, rate = read_wave(note['target_wav'])
        assert rate == SAMPLE_RATE
        target = target[:round(duration*rate)]
        folder = args.output/note['note_str']
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy(note['target_wav'], folder/'target.wav')
        summary = {'note': note, 'families': {}}
        for family in args.families.split(','):
            print(f"== {note['note_str']} / {family} ({time.strftime('%H:%M:%S')})", flush=True)
            report = fit_note(target, family, starts=starts, seed=args.seed, device=args.device, duration=duration,
                              excitation=args.excitation, schedule_a=schedule_a, schedule_b=schedule_b,
                              output_dir=folder/family, label_f0=note['label_f0_hz'])
            summary['families'][family] = {k: report[k] for k in ('loss', 'log_spectral_distance_db', 'derived', 'physical', 'f1_cents_vs_label')}
            print(json.dumps({'family': family, 'loss': report['loss'], 'f0_hz': report['derived']['f0_hz'],
                              'cents_vs_label': report['f1_cents_vs_label'],
                              'stage_A': (report['stages']['A']['evaluations'], round(report['stages']['A']['seconds'])),
                              'stage_B': (report['stages']['B']['evaluations'], round(report['stages']['B']['seconds']))}), flush=True)
        winner = min(summary['families'], key=lambda f: summary['families'][f]['loss'])
        summary['winner'] = winner
        with open(folder/'summary.json', 'w') as handle:
            json.dump(summary, handle, indent=1)
        print(f"winner for {note['note_str']}: {winner}", flush=True)


if __name__ == '__main__':
    main()
