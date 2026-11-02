"""NSynth acoustic-guitar notes: download, filter, select and prepare fitting targets.

Run on the login node (it needs the network); the extracted archive is treated
as untrusted data and lives in its own directory on scratch.
"""
import argparse
import json
from pathlib import Path
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.applications.common.media import read_wave, write_wave

URL = 'http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-{split}.jsonwav.tar.gz'
DEFAULT_ROOT = Path('/data/scratch/acw794/data/nsynth')
# NSynth quality order; 'reverb' is index 8.
QUALITIES = ('bright', 'dark', 'distortion', 'fast_decay', 'long_release', 'multiphonic',
             'nonlinear_env', 'percussive', 'reverb', 'tempo-synced')
EXCLUDED = ('reverb', 'distortion', 'multiphonic', 'tempo-synced')
DEFAULT_PITCHES = (40, 50, 59, 67, 76)          # E2, D3, B3, G4, E5
DEFAULT_VELOCITIES = (25, 50, 75, 100, 127)
SAMPLE_RATE = 16000


def download(split, root=DEFAULT_ROOT):
    """Fetch and extract one jsonwav split; returns the directory holding examples.json and audio/."""
    folder = root/split
    archive = folder/'download'/f'nsynth-{split}.jsonwav.tar.gz'
    extracted = folder/'extracted'
    inner = extracted/f'nsynth-{split}'
    if (inner/'examples.json').exists():
        return inner
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        print(f'downloading {URL.format(split=split)} -> {archive}', flush=True)
        urllib.request.urlretrieve(URL.format(split=split), archive)
    extracted.mkdir(parents=True, exist_ok=True)
    print(f'extracting {archive} -> {extracted}', flush=True)
    with tarfile.open(archive, 'r:gz') as handle:
        handle.extractall(extracted, filter='data')
    if not (inner/'examples.json').exists():
        raise FileNotFoundError(f'{inner}/examples.json not found after extraction')
    return inner


def eligible(entry):
    """Acoustic guitar without reverb, distortion, multiphonic or tempo-synced qualities."""
    if entry['instrument_family_str'] != 'guitar' or entry['instrument_source_str'] != 'acoustic':
        return False
    qualities = set(entry['qualities_str'])
    flags = entry['qualities']
    for index, name in enumerate(QUALITIES):
        if bool(flags[index]) != (name in qualities):
            raise ValueError(f"{entry['note_str']}: qualities vector and qualities_str disagree on {name}")
    return not (qualities & set(EXCLUDED))


def pool(examples_json):
    """Eligible entries sorted by note name, plus the filter statistics."""
    with open(examples_json) as handle:
        examples = json.load(handle)
    entries = list(examples.values())
    guitar = [e for e in entries if e['instrument_family_str'] == 'guitar' and e['instrument_source_str'] == 'acoustic']
    removed = {name: sum(name in e['qualities_str'] for e in guitar) for name in EXCLUDED}
    kept = sorted((e for e in guitar if eligible(e)), key=lambda e: e['note_str'])
    stats = {'total': len(entries), 'acoustic_guitar': len(guitar), 'removed_by_quality': removed, 'pool': len(kept),
             'pitch_min': min((e['pitch'] for e in kept), default=None), 'pitch_max': max((e['pitch'] for e in kept), default=None)}
    return kept, stats


def select_notes(candidates, pitches=DEFAULT_PITCHES, velocities=DEFAULT_VELOCITIES, max_offset=2):
    """One note per requested pitch (exact, else within ±max_offset), spreading the velocities; deterministic."""
    chosen, used_velocities = [], []
    for pitch in pitches:
        picked = None
        for offset in range(max_offset+1):
            near = [e for e in candidates if abs(e['pitch']-pitch) <= offset and e not in chosen]
            if near:
                unused = [e for e in near if e['velocity'] not in used_velocities]
                order = sorted(unused or near, key=lambda e: (abs(e['pitch']-pitch), e['note_str']))
                picked = dict(order[0], pitch_offset=order[0]['pitch']-pitch, requested_pitch=pitch)
                break
        if picked is None:
            raise LookupError(f'No eligible acoustic-guitar note within ±{max_offset} semitones of MIDI pitch {pitch}; '
                              'widen --pitches or add another split.')
        chosen.append(picked); used_velocities.append(picked['velocity'])
    return chosen


def prepare_target(audio, sample_rate=SAMPLE_RATE, duration=3.0, onset_threshold=0.02):
    """Trim leading silence to the first sample above ``onset_threshold``·peak, remove the mean, keep ``duration`` seconds."""
    audio = np.asarray(audio, dtype=np.float64)
    peak = float(np.abs(audio).max())
    if peak <= 0:
        raise ValueError('silent recording')
    onset = int(np.argmax(np.abs(audio) >= onset_threshold*peak))
    samples = round(duration*sample_rate)
    segment = audio[onset:onset+samples]
    if len(segment) < samples:
        segment = np.pad(segment, (0, samples-len(segment)))
    segment = segment - segment.mean()
    return segment, onset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--splits', default='test', help='comma-separated NSynth splits to pool (test, valid)')
    parser.add_argument('--out', type=Path, default=None, help='manifest path (default <root>/selected/manifest.json)')
    parser.add_argument('--pitches', default=','.join(map(str, DEFAULT_PITCHES)))
    parser.add_argument('--velocities', default=','.join(map(str, DEFAULT_VELOCITIES)))
    parser.add_argument('--duration', type=float, default=3.0)
    parser.add_argument('--max-offset', type=int, default=2)
    args = parser.parse_args(argv)
    pitches = tuple(int(p) for p in args.pitches.split(','))
    velocities = tuple(int(v) for v in args.velocities.split(','))
    out = args.out or args.root/'selected'/'manifest.json'
    candidates, statistics = [], {}
    for split in args.splits.split(','):
        folder = download(split, args.root)
        kept, stats = pool(folder/'examples.json')
        for entry in kept:
            entry['split'] = split; entry['wav'] = str(folder/'audio'/f"{entry['note_str']}.wav")
        candidates += kept; statistics[split] = stats
        print(f"{split}: {stats['total']} notes, {stats['acoustic_guitar']} acoustic guitar, removed "
              + ', '.join(f"{k} {v}" for k, v in stats['removed_by_quality'].items())
              + f", pool {stats['pool']} (pitch {stats['pitch_min']}-{stats['pitch_max']})", flush=True)
    selected = select_notes(candidates, pitches, velocities, args.max_offset)
    out.parent.mkdir(parents=True, exist_ok=True)
    notes = []
    for entry in selected:
        audio, rate = read_wave(entry['wav'])
        if rate != SAMPLE_RATE or audio.ndim != 1:
            raise ValueError(f"{entry['wav']}: expected mono {SAMPLE_RATE} Hz")
        target, onset = prepare_target(audio, rate, args.duration)
        folder = out.parent/entry['note_str']
        write_wave(folder/'target.wav', target, rate)
        notes.append({'note_str': entry['note_str'], 'split': entry['split'], 'instrument_str': entry['instrument_str'],
                      'pitch': entry['pitch'], 'requested_pitch': entry['requested_pitch'], 'pitch_offset': entry['pitch_offset'],
                      'velocity': entry['velocity'], 'qualities_str': entry['qualities_str'], 'source_wav': entry['wav'],
                      'target_wav': str(folder/'target.wav'), 'onset_sample': onset, 'duration_s': args.duration,
                      'sample_rate': rate, 'rms': float(np.sqrt(np.mean(target**2))), 'peak': float(np.abs(target).max()),
                      'label_f0_hz': 440.0*2**((entry['pitch']-69)/12)})
        print(f"selected {entry['note_str']}: pitch {entry['pitch']} (requested {entry['requested_pitch']}), "
              f"velocity {entry['velocity']}, onset {onset} samples, qualities {entry['qualities_str']}", flush=True)
    manifest = {'filter': {'family': 'guitar', 'source': 'acoustic', 'excluded_qualities': list(EXCLUDED)},
                'statistics': statistics, 'pitches': list(pitches), 'velocities': list(velocities), 'notes': notes}
    with open(out, 'w') as handle:
        json.dump(manifest, handle, indent=1)
    print(f'manifest -> {out}')
    return manifest


if __name__ == '__main__':
    main()
