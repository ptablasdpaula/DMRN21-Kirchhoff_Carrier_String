"""CLAP-guided semantic shaping of one string: Text2FX's directional objective on a wide material box.

For a prompt pair (A, B) the batch holds two directions, A-with-contrast-B and
B-with-contrast-A, each from the same quasi-random starts (start 0 is the
paper's nominal string). Each item is optimised independently with Adam and
plateau decay; the retained result is the item's lowest-objective iterate.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from experiments.applications.common.boxes import decode_batch, encode_rows, sample_logits
from experiments.applications.common.optim import Schedule, fit_batch
from experiments.applications.common.render import BatchRenderer
from experiments.applications.common.media import FrameLogger, animate, write_wave, save_spectrogram_png, spectrogram_db
from experiments.applications.semantic.box import (WideStringBox, production_grid, pilot_grid, perceptual,
                                                   SAMPLE_RATE, DURATION, EXCITATION, FADE_SECONDS)
from experiments.applications.semantic.clap import ClapScorer, directional_loss, margin_loss, barrier
from experiments.applications.semantic.prompts import PAIRS, directions

SCHEDULE = Schedule(initial_lr=0.05, factor=0.5, plateau_patience=15, threshold=1e-3, stop_patience=60,
                    maximum_updates=300, relative=False)
SPEC_FFT, SPEC_HOP = 2048, 480
MATERIAL_NAMES = ('rho', 'T', 'r', 'E', 'L', 'a', 'sigma_0', 'sigma_1', 'pluck', 'pickup')


def git_revision():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def footer(material):
    p = perceptual(material)
    rho, T, r, E, L, a, s0, s1, pluck, pickup = [float(v) for v in material]
    return (f"f₀ {float(p['f0_hz']):.1f} Hz · B {float(p['B']):.1e} · T60₁ {float(p['t60_mode1_s']):.2f} s · "
            f"T60(1 kHz) {float(p['t60_1khz_s']):.2f} s · ρ {rho:.0f} · E {E/1e9:.1f} GPa · T {T:.1f} N · "
            f"L {L*100:.0f} cm · r {r*1e3:.2f} mm · pluck {pluck:.2f}")


OBJECTIVES = {'margin': margin_loss, 'directional': directional_loss}


def run_pair(pair, *, starts=32, grid='production', seed=0, device='cuda', schedule=SCHEDULE, group=16,
             frames_per_run=100, output_dir=None, top=3, objective_name='margin'):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    box = WideStringBox()
    grid_config = production_grid(box) if grid == 'production' else pilot_grid(box)
    renderer = BatchRenderer(grid_config, DURATION, EXCITATION, device=device, fade_seconds=FADE_SECONDS)
    scorer = ClapScorer(device)
    dirs = directions(pair)                                  # [(target, contrast), (contrast, target)]
    prompts = [d[0] for d in dirs]
    text = scorer.embed_text(prompts)                        # (2, 512)
    direction = F.normalize(text[0]-text[1], dim=-1)
    d_hat = torch.stack((direction, -direction))             # per direction
    starts_z = sample_logits(box, starts, seed=seed, device=device)
    starts_z[0] = encode_rows(box, box.defaults()).to(device)
    logits0 = torch.cat([starts_z, starts_z.clone()])        # (2*starts, 9)
    which = torch.arange(2*starts, device=device)//starts    # direction of each item
    with torch.no_grad():
        silence = torch.zeros(1, renderer.samples, dtype=torch.float64, device=device)
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(1, renderer.samples, dtype=torch.float64, device=device, generator=generator)
        baseline = {'silence': (scorer.embed_audio(silence, normalise=False)@text.T)[0].tolist(),
                    'noise': (scorer.embed_audio(noise)@text.T)[0].tolist()}
    state = {'e0': None}
    history = open(output_dir/'history.jsonl', 'w')
    trajectory = []
    frames = [FrameLogger(every=max(1, math.ceil(schedule.maximum_updates/frames_per_run))) for _ in dirs]
    start_time = time.perf_counter()

    def clap_losses(audio_chunk, idx):
        e = scorer.embed_audio(audio_chunk)                  # (k, 512) float32
        s_plus = (e*text[which[idx]]).sum(-1)
        s_minus = (e*text[1-which[idx]]).sum(-1)
        # Iteration 0: the displacement is zero, so the objective is its reference value (1 for both objectives).
        e0 = e.detach() if state['e0'] is None else state['e0'][idx]
        loss = OBJECTIVES[objective_name](e, e0, d_hat[which[idx]])
        return loss.to(torch.float64), e.detach(), s_plus.detach(), s_minus.detach()

    def objective(z, indices):
        material = box.to_material(decode_batch(box, z))
        audio = renderer(material)
        losses, embeddings, s_plus, s_minus = [], [], [], []
        for start in range(0, len(indices), group):
            idx = indices[start:start+group]
            chunk = audio[start:start+group]
            loss, e, sp, sm = checkpoint(clap_losses, chunk, idx, use_reentrant=False)
            losses.append(loss); embeddings.append(e); s_plus.append(sp); s_minus.append(sm)
        loss = torch.cat(losses) + barrier(z)
        aux = {'audio': audio.detach(), 'material': material.detach(), 'embeddings': torch.cat(embeddings),
               's_plus': torch.cat(s_plus), 's_minus': torch.cat(s_minus)}
        if state['e0'] is None:
            state['e0'] = torch.zeros(len(logits0), text.shape[1], dtype=aux['embeddings'].dtype, device=device)
            state['e0'][indices] = aux['embeddings']
        return loss, aux

    def callback(info):
        aux, idx = info['aux'], info['indices']
        trajectory.append(info['best_logits'].detach().cpu().numpy().copy())
        row = {'evaluation': info['evaluation'], 'seconds': time.perf_counter()-start_time,
               'indices': idx.tolist(), 'loss': info['losses'].tolist(), 's_plus': aux['s_plus'].tolist(),
               's_minus': aux['s_minus'].tolist(), 'lr': info['lr'][idx].tolist(), 'active': int(info['active'].sum()),
               'peak_m': aux['audio'].abs().amax(-1).tolist()}
        history.write(json.dumps(row)+'\n'); history.flush()
        iteration = info['evaluation']-1
        for d, logger in enumerate(frames):
            mask = which[idx] == d
            if not bool(mask.any()) or not (logger.due(iteration) or iteration == 0):
                continue
            local = int(torch.nonzero(mask).flatten()[info['losses'][mask].argmin()])
            spec = spectrogram_db(aux['audio'][local:local+1], SPEC_FFT, SPEC_HOP)[0]
            title = (f"'{prompts[d]}' · iteration {iteration} · objective {float(info['losses'][local]):.3f} · "
                     f"s⁺ {float(aux['s_plus'][local]):.3f} · s⁻ {float(aux['s_minus'][local]):.3f}")
            logger.log(iteration, spec, title, footer(aux['material'][local]), force=True,
                       loss=float(info['losses'][local]), s_plus=float(aux['s_plus'][local]), item=int(idx[local]))

    result = fit_batch(logits0, objective, schedule=schedule, callback=callback)
    history.close()
    best_material = box.to_material(decode_batch(box, result.best_logits))
    start_material = box.to_material(decode_batch(box, logits0))
    with torch.no_grad():
        audio_end = renderer(best_material)
        audio_start = renderer(start_material)
        e_end, e_start = scorer.embed_audio(audio_end), scorer.embed_audio(audio_start)
        sp_end, sm_end = (e_end*text[which]).sum(-1), (e_end*text[1-which]).sum(-1)
        sp_start, sm_start = (e_start*text[which]).sum(-1), (e_start*text[1-which]).sum(-1)
    gain = sp_end-sp_start
    np.savez_compressed(output_dir/'trajectory.npz', logits=np.stack(trajectory), start=logits0.cpu().numpy(),
                        best=result.best_logits.cpu().numpy(), which=which.cpu().numpy())
    perc_end = {k: v.cpu() for k, v in perceptual(best_material).items()}
    perc_start = {k: v.cpu() for k, v in perceptual(start_material).items()}
    items = []
    for i in range(len(logits0)):
        items.append({'item': i, 'direction': int(which[i]), 'prompt': prompts[int(which[i])], 'start_index': i % starts,
                      'objective_start': float(result.initial_loss[i]), 'objective_best': float(result.best_loss[i]),
                      's_plus_start': float(sp_start[i]), 's_plus_end': float(sp_end[i]), 's_minus_start': float(sm_start[i]),
                      's_minus_end': float(sm_end[i]), 'gain': float(gain[i]), 'margin_end': float(sp_end[i]-sm_end[i]),
                      'updates': int(result.updates[i]), 'lr_reductions': int(result.reductions[i]),
                      'peak_m_end': float(audio_end[i].abs().max()),
                      'material_end': {n: float(v) for n, v in zip(MATERIAL_NAMES, best_material[i].tolist())},
                      'material_start': {n: float(v) for n, v in zip(MATERIAL_NAMES, start_material[i].tolist())},
                      'perceptual_end': {k: float(v[i]) for k, v in perc_end.items()},
                      'perceptual_start': {k: float(v[i]) for k, v in perc_start.items()}})
    selected = []
    for d in range(len(dirs)):
        ranked = sorted((it for it in items if it['direction'] == d), key=lambda it: -it['gain'])[:top]
        for rank, it in enumerate(ranked):
            tag = f"dir{d}_rank{rank}"
            for stage, audio in (('start', audio_start), ('end', audio_end)):
                row = audio[it['item']].cpu()
                write_wave(output_dir/'audio'/f'{tag}_{stage}.wav', (row/row.abs().max()*10**(-1/20)).numpy(), SAMPLE_RATE)
            save_spectrogram_png(output_dir/'img'/f'{tag}.png',
                                 [('start', spectrogram_db(audio_start[it['item']].cpu(), SPEC_FFT, SPEC_HOP)),
                                  ('end', spectrogram_db(audio_end[it['item']].cpu(), SPEC_FFT, SPEC_HOP))],
                                 SAMPLE_RATE, SPEC_HOP, fmax=14000)
            selected.append(dict(it, tag=tag, rank=rank))
    gifs = []
    for d, logger in enumerate(frames):
        if logger.frames:
            logger.save(output_dir/f'frames_dir{d}.npz', output_dir/f'frames_dir{d}.jsonl')
            try:
                gifs.append(str(animate(output_dir/f'frames_dir{d}.npz', output_dir/f'frames_dir{d}.jsonl', output_dir/f'dir{d}.gif',
                                        sample_rate=SAMPLE_RATE, hop=SPEC_HOP, fps=10, fmax=14000, frame_title='current best')))
            except Exception as error:
                gifs.append(f'error: {error!r}')
    summary = {'pair': pair, 'objective': objective_name, 'prompts': prompts, 'directions': [{'target': t, 'contrast': c} for t, c in dirs],
               'box': {'names': box.names, 'ranges': box.ranges}, 'grid': renderer.describe(), 'schedule': schedule.__dict__,
               'starts': starts, 'seed': seed, 'group': group, 'git': git_revision(), 'torch': torch.__version__,
               'gpu': torch.cuda.get_device_name() if device != 'cpu' and torch.cuda.is_available() else None,
               'evaluations': result.evaluations, 'seconds': result.seconds, 'baseline_similarity': baseline,
               'items': items, 'selected': selected, 'gifs': gifs,
               'spread_end': {k: float(v.std()) for k, v in perc_end.items()}}
    with open(output_dir/'summary.json', 'w') as handle:
        json.dump(summary, handle, indent=1)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', required=True, choices=sorted(PAIRS))
    parser.add_argument('--starts', type=int, default=32)
    parser.add_argument('--grid', default='production', choices=('production', 'pilot'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--group', type=int, default=16)
    parser.add_argument('--max-iter', type=int, default=None)
    parser.add_argument('--objective', default='margin', choices=sorted(OBJECTIVES))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    schedule = SCHEDULE
    if args.max_iter is not None:
        schedule = Schedule(**{**SCHEDULE.__dict__, 'maximum_updates': args.max_iter,
                               'stop_patience': min(SCHEDULE.stop_patience, max(1, args.max_iter))})
    summary = run_pair(args.pair, starts=args.starts, grid=args.grid, seed=args.seed, device=args.device,
                       schedule=schedule, group=args.group, output_dir=args.output, objective_name=args.objective)
    best = max(summary['items'], key=lambda it: it['gain'])
    print(json.dumps({'pair': args.pair, 'evaluations': summary['evaluations'], 'seconds': round(summary['seconds']),
                      'best_gain': round(best['gain'], 4), 'best_prompt': best['prompt'],
                      'best_f0_hz': round(best['perceptual_end']['f0_hz'], 1)}), flush=True)


if __name__ == '__main__':
    main()
