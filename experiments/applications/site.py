"""Build the online supplement (docs/index.html) from the application results.

Inputs: a sound-matching ``results.json`` (from ``sound_matching/evaluate.py``,
assets copied under ``docs/supplement/sound_matching/``) and one or more CLAP
``summary.json`` runs (assets copied under ``docs/supplement/semantic/<pair>/``).
Static HTML and CSS only, so GitHub Pages can serve the ``docs`` folder as is.
"""
import argparse
import html
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
TITLE = 'Efficient Backpropagation for Finite-Difference Nonlinear String Synthesis: online supplement'
REPO = 'https://github.com/ptablasdpaula/DMRN21-Kirchhoff_Carrier_String'

CSS = """
:root{--bg:#fbfbf8;--fg:#1b1d22;--muted:#636872;--rule:#d9dbd4;--panel:#f1f2ec;--accent:#1e5a78;--mono:ui-monospace,Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root{--bg:#15171b;--fg:#e6e6e0;--muted:#9ea3ad;--rule:#2e3138;--panel:#1c1f25;--accent:#7cbbd9;color-scheme:dark}}
body{margin:0;background:var(--bg);color:var(--fg);font:17px/1.5 Georgia,'Times New Roman',serif;padding:0 max(16px,env(safe-area-inset-left)) 48px max(16px,env(safe-area-inset-right))}
.wrap{max-width:1100px;margin:0 auto}
h1,h2,h3,h4{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;line-height:1.2;text-wrap:balance}
h1{font-size:1.9rem;margin:2rem 0 .3rem}h2{font-size:1.35rem;margin-top:2.8rem;padding-top:.6rem;border-top:1px solid var(--rule)}
h3{font-size:1.1rem;margin:1.8rem 0 .4rem}h4{font-size:.95rem;margin:1rem 0 .3rem;color:var(--muted)}
p{max-width:72ch}.muted{color:var(--muted)}.small{font-size:.9rem}
.note{background:var(--panel);border-left:3px solid var(--accent);padding:.8rem 1rem;border-radius:0 6px 6px 0;margin:1rem 0;max-width:80ch}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1rem 1.6rem;align-items:start}.grid>*{min-width:0}
img{max-width:100%;height:auto;display:block;border:1px solid var(--rule);background:#fff}
audio{width:100%;max-width:420px;display:block;margin:.3rem 0}
table{border-collapse:collapse;font-size:.92rem;margin:.5rem 0;width:100%}th,td{text-align:left;vertical-align:top;padding:.3rem .6rem;border-bottom:1px solid var(--rule)}
th{font-family:system-ui,sans-serif;font-size:.78rem;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);font-weight:500}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}.tablewrap{overflow-x:auto}
code,pre{font-family:var(--mono);font-size:.85em}pre{background:var(--panel);padding:.8rem;border-radius:6px;overflow-x:auto}
.tag{font-family:system-ui,sans-serif;font-size:.78rem;background:var(--panel);border-radius:4px;padding:.1rem .5rem;color:var(--accent)}
a{color:var(--accent)}
"""


def e(x):
    return html.escape(str(x))


def fmt(x, digits=3):
    if x is None:
        return '–'
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return f'{x}'
    ax = abs(x)
    if ax != 0 and (ax < 1e-2 or ax >= 1e5):
        return f'{x:.{digits-1}e}'
    return f'{x:.{digits}g}'


def table(headers, rows, numeric=()):
    head = ''.join(f'<th class="{"num" if i in numeric else ""}">{e(h)}</th>' for i, h in enumerate(headers))
    body = ''.join('<tr>'+''.join(f'<td class="{"num" if i in numeric else ""}">{c}</td>' for i, c in enumerate(row))+'</tr>' for row in rows)
    return f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def sound_matching_section(results, base):
    stats = results['manifest']['statistics']
    split_lines = '; '.join(f"{split}: {s['acoustic_guitar']} acoustic-guitar notes of {s['total']}, "
                            + ', '.join(f"{k} −{v}" for k, v in s['removed_by_quality'].items()) + f", pool {s['pool']}"
                            for split, s in stats.items())
    out = [f'<h2 id="sound-matching">1. Sound matching on recorded guitar notes</h2>',
           f'<p>Target notes come from the NSynth dataset\'s acoustic-guitar subset, excluding notes labelled with the '
           f'<code>reverb</code>, <code>distortion</code>, <code>multiphonic</code> or <code>tempo-synced</code> qualities '
           f'({e(split_lines)}). Each note is trimmed to its first 3 s. All ten string parameters are fitted blind: density, tension, '
           f'radius, Young\'s modulus, length, pluck amplitude, the two damping coefficients, and the pluck and pickup positions. '
           f'The pitch label is never used in the fit; it is only reported afterwards. The loss is the cumulative energy loss '
           f'(<code>cels-audio</code>, log-weighted, time-forward directions) on unit-RMS waveforms at 16 kHz, so level is not fitted '
           f'and the pluck amplitude acts only through the string\'s nonlinearity. No body model is used.</p>',
           '<p>Stage A runs 128 quasi-random starts over a whole string family (steel-core or nylon-core homogeneous equivalents) on a '
           'coarse grid whose interval count is set by the family\'s highest fundamental; stage B refines the eight best distinct '
           'candidates in a narrow box around the winner on a grid with as many intervals as that box allows. Both stages use Adam on '
           'box logits with per-candidate plateau decay and keep each candidate\'s lowest-loss iterate.</p>',
           '<div class="note small">Identifiability: the waveform determines f₀ = √(T/μ)/(2L), the inharmonicity B = π²EI/(TL²), the two decays, '
           'the pluck and pickup fractions and, weakly through the pitch glide, the nonlinearity EA/T. Density, radius, modulus, tension and '
           'length are not individually identifiable; the catalogue lookup below (D\'Addario unit weights, Woodhouse et al. 2012 bending '
           'stiffness per construction class) is what turns a fit into a physical string, scale length and fret.</div>']
    for entry in results['notes']:
        note = entry['note']
        out.append(f'<h3 id="{e(note["note_str"])}">{e(note["note_str"])} <span class="tag">MIDI {note["pitch"]} · {fmt(note["label_f0_hz"], 4)} Hz · velocity {note["velocity"]}</span></h3>')
        for family, rec in entry['families'].items():
            win = ' <span class="tag">best family</span>' if family == entry['winner'] else ''
            assets = rec['assets']
            out.append(f'<h4>{e(family)} family{win}</h4><div class="grid"><div>')
            if 'target_matched_level.wav' in assets:
                out.append(f'<p class="small muted">target (level-matched)</p><audio controls preload="none" src="{base}/{e(assets["target_matched_level.wav"])}"></audio>')
            if 'fit.wav' in assets:
                out.append(f'<p class="small muted">fit</p><audio controls preload="none" src="{base}/{e(assets["fit.wav"])}"></audio>')
            d, f = rec['derived'], rec['fitted']
            out.append(table(['metric', 'value'], [
                ['CeL (lower is better)', fmt(rec['loss'], 4)], ['log-spectral distance', f"{fmt(rec['log_spectral_distance_db'], 3)} dB"],
                ['fitted f₁', f"{fmt(d['f1_hz'], 4)} Hz ({rec['f1_cents_vs_label']:+.0f} cents vs label)"],
                ['inharmonicity B', fmt(d['B'])], ['mass per length μ', f"{fmt(f['mu_kg_per_m']*1e3)} g/m"],
                ['stage A / B evaluations', f"{rec['stages']['A']['evaluations']} / {rec['stages']['B']['evaluations']}"],
                ['stage A / B grid', f"{rec['stages']['A']['grid']['intervals']} intervals at {rec['stages']['A']['grid']['integration_rate']//1000} kHz / "
                                     f"{rec['stages']['B']['grid']['intervals']} at {rec['stages']['B']['grid']['integration_rate']//1000} kHz"]], numeric=(1,)))
            p = rec['physical']
            if rec.get('at_bounds'):
                out.append(f'<p class="small muted">On a family-box bound: {e(", ".join(rec["at_bounds"]))}. The waveform does not pin these down individually; see the caveat above.</p>')
            out.append(table(['parameter', 'fitted'], [
                ['density ρ', f"{fmt(p['material_density'], 4)} kg/m³"], ['tension T', f"{fmt(p['tension'], 4)} N"],
                ['radius r', f"{fmt(p['radius']*1e3)} mm"], ['Young\'s modulus E', f"{fmt(p['young_modulus']/1e9)} GPa"],
                ['length L', f"{fmt(p['length']*100, 4)} cm"], ['pluck amplitude', f"{fmt(p['amplitude']*1e3)} mm"],
                ['σ₀', f"{fmt(p['sigma_0'])} s⁻¹"], ['σ₁', f"{fmt(p['sigma_1'])} m²/s"],
                ['pluck position', fmt(p['pluck_position'])], ['pickup position', fmt(p['pickup_position'])]], numeric=(1,)))
            out.append('</div><div>')
            if 'spectrograms.png' in assets:
                out.append(f'<img src="{base}/{e(assets["spectrograms.png"])}" alt="target and fit spectrograms">')
            if 'fit.gif' in assets:
                out.append(f'<p class="small muted">optimisation, iteration 0 to the end (current best candidate)</p><img src="{base}/{e(assets["fit.gif"])}" alt="optimisation animation" loading="lazy">')
            out.append('</div></div>')
            if rec['catalogue']:
                out.append('<p class="small muted">Nearest real strings (D\'Addario) by (μ, T, B) in log space, over 25.5 in, 24.75 in and 650 mm scales and frets 0–12:</p>')
                out.append(table(['string', 'construction', 'gauge (in)', 'scale', 'fret', 'predicted T', 'predicted B', 'distance'],
                                 [[e(m['code']), e(m['material'] or m['class']), f"{m['diameter_in']:.3f}", e(m['scale']), m['fret'],
                                   f"{m['tension_n']:.0f} N", fmt(m['B']), fmt(m['distance'])] for m in rec['catalogue']], numeric=(2, 4, 5, 6, 7)))
    return '\n'.join(out)


def semantic_section(runs, base):
    out = ['<h2 id="semantic">2. Text-guided sound design with CLAP</h2>',
           '<p>Following Text2FX, the string\'s parameters are optimised so that the CLAP audio embedding of the rendered sound moves along '
           'the direction from a contrasting prompt to a target prompt (the directional objective). The string box is wide: stress 50 MPa–1 GPa, '
           'radius 0.15–0.6 mm, length 0.5–1 m, a material axis from nylon to steel, dampings over three decades, and free pluck and pickup positions. '
           'Each 2.5 s clip is peak-normalised before CLAP (so loudness is not a lever) and tiled four times to CLAP\'s 10 s window, exactly as '
           'the model\'s own preprocessing does. Every prompt pair is run in both directions from the same 32 starts (start 0 is the paper\'s string).</p>']
    for run in runs:
        s = run['summary']
        out.append(f'<h3 id="{e(s["pair"])}">{e(s["pair"].replace("_", " vs "))}</h3>')
        grid = s['grid']
        out.append(f'<p class="small muted">{grid["intervals"]} intervals at {grid["integration_rate"]//1000} kHz internal rate, 48 kHz output, '
                   f'{s["evaluations"]} iterations, {s["seconds"]/60:.0f} min on {e(s.get("gpu") or "GPU")}; baseline similarities to the two prompts: '
                   f'silence {", ".join(f"{v:.3f}" for v in s["baseline_similarity"]["silence"])}, white noise {", ".join(f"{v:.3f}" for v in s["baseline_similarity"]["noise"])}.</p>')
        for d, direction in enumerate(s['directions']):
            out.append(f'<h4>towards “{e(direction["target"])}” (away from “{e(direction["contrast"])}”)</h4>')
            gif = run['assets'].get(f'dir{d}.gif')
            out.append('<div class="grid"><div>')
            for it in [x for x in s['selected'] if x['direction'] == d]:
                tag = it['tag']
                out.append(f'<p class="small muted">start {it["start_index"]}: similarity to target {it["s_plus_start"]:.3f} → {it["s_plus_end"]:.3f} '
                           f'(gain {it["gain"]:+.3f}); f₀ {it["perceptual_start"]["f0_hz"]:.0f} → {it["perceptual_end"]["f0_hz"]:.0f} Hz</p>')
                for stage in ('start', 'end'):
                    key = f'{tag}_{stage}.wav'
                    if key in run['assets']:
                        out.append(f'<audio controls preload="none" src="{base}/{e(run["assets"][key])}"></audio>')
                if f'{tag}.png' in run['assets']:
                    out.append(f'<img src="{base}/{e(run["assets"][f"{tag}.png"])}" alt="start and end spectrograms" loading="lazy">')
            out.append('</div><div>')
            if gif:
                out.append(f'<p class="small muted">optimisation of the best start, iteration 0 to the end</p><img src="{base}/{e(gif)}" alt="optimisation animation" loading="lazy">')
            items = [x for x in s['items'] if x['direction'] == d]
            items.sort(key=lambda x: -x['gain'])
            rows = [[x['start_index'], f"{x['s_plus_start']:.3f}", f"{x['s_plus_end']:.3f}", f"{x['gain']:+.3f}",
                     f"{x['perceptual_end']['f0_hz']:.0f}", fmt(x['perceptual_end']['B']), f"{x['perceptual_end']['t60_mode1_s']:.2f}",
                     f"{x['material_end']['rho']:.0f}", f"{x['material_end']['E']/1e9:.1f}", f"{x['material_end']['T']:.0f}"] for x in items[:8]]
            out.append(table(['start', 's⁺ start', 's⁺ end', 'gain', 'f₀ (Hz)', 'B', 'T60₁ (s)', 'ρ', 'E (GPa)', 'T (N)'], rows, numeric=range(10)))
            out.append('</div></div>')
    return '\n'.join(out)


def verification_section():
    """Where the abstract's gradients are checked, with the numbers from the runs behind it."""
    blob = REPO + '/blob/main/'
    link = lambda path, text=None: f'<a href="{blob}{path}"><code>{e(text or path)}</code></a>'
    return '\n'.join([
        '<h2 id="verification">3. Verification of the gradients</h2>',
        '<p>The abstract keeps its correctness claims out of the page and in the repository. '
        f'The checkpointed adjoint ({link("src/adjoint.py")}: 256-step checkpoints, each segment re-run on the reverse pass) is checked against independent references; '
        'the numbers are from the runs behind the abstract.</p>',
        '<ul>',
        f'<li><strong>Forward-mode Jacobian.</strong> The hand-derived reverse pass against an independent forward-mode Jacobian of the same scheme ({link("test/adjoint.py")}).</li>',
        '<li><strong>Float64 PyTorch autodiff of the same scheme.</strong> In the benchmark sweep (job 34855605, 1 s, batches 1 to 32) '
        'every evaluation is validated against the checkpointed adjoint outside the timed region: autodiff\'s ten scaled gradients agree with it to '
        '1.5&times;10<sup>&minus;9</sup> relative &#8467;<sub>2</sub> '
        f'({link("benchmarks/native_rate_results/a100-sweep-1s-mae-si10-34855605.json")}, field <code>validation.scaled_gradient_relative_l2</code>).</li>',
        '<li><strong>Compiled autodiff of the same scheme.</strong> The abstract\'s three compiled rows, the scheme under <code>jax.jit</code> on the grid without and with '
        f'256-step checkpoints and in the discrete sine basis with them ({link("experiments/figure_comparison/fdtd_jax.py")}), are checked in every benchmark case against the project\'s reference FDTD solver '
        '(first item of each batch, jobs 35747154, 35747153 and 35746438): waveform to 3.9&times;10<sup>&minus;11</sup> and ten scaled gradients to 1.5&times;10<sup>&minus;10</sup> relative &#8467;<sub>2</sub> '
        f'({link("benchmarks/native_rate_results/a100-jit-jax-sine-ckpt-rdg-35746438-1s-mae-si10.json")}); on short runs they reproduce the PyTorch baseline to '
        f'10<sup>&minus;10</sup> in waveform and 5&times;10<sup>&minus;9</sup> in gradients ({link("test/figure_fdtd_jax.py")}).</li>',
        '<li><strong>Scalar-cached variant, used for the applications above.</strong> The sound matching and CLAP optimisations ran on a variant of the kernel that also caches '
        'three scalars per time step and regenerates the states backwards from each checkpoint instead of re-running the segment. In the same sweep its ten scaled gradients '
        'agree with the checkpointed adjoint to 2.7&times;10<sup>&minus;11</sup> relative &#8467;<sub>2</sub>, and the two are compared on 4 s batches in '
        f'{link("test/scalar_cache_smoke.py")}; the one-step algebra of its reverse pass is checked against autograd in 96 float64 cases to 10<sup>&minus;11</sup> '
        f'({link("test/scalar_cache_algebra.py")}).</li>',
        '<li><strong>Central finite differences of the reference Octave code</strong> (the non-SAV scheme of Russo et al. with the five-point stencil substituted, '
        'off-centre pluck and pickup), run on the scalar-cached variant: all ten gradients pass; the largest relative error is 4.3&times;10<sup>&minus;6</sup> on &sigma;<sub>1</sub>, whose gradient is '
        '2&times;10<sup>&minus;6</sup> in magnitude, and the pluck and pickup position gradients agree to 1.3&times;10<sup>&minus;7</sup> and 8.7&times;10<sup>&minus;8</sup> '
        f'({link("test/reference_paper_exact.py")} with <code>--order 4</code>; jobs 34846668 to 34846671).</li>',
        f'<li><strong>Unit checks.</strong> The spatial PyTorch baseline against a dual-number reference, positions included ({link("test/figure_fdtd.py")}, {link("test/positions.py")}).</li>',
        '</ul>',
        f'<p class="small muted">The derivation of the scalar-cached variant and the full validation tables are in the <a href="{blob}paper_dmrn2026/technical_note.pdf">technical note</a>.</p>'])


def build(results_path, semantic_runs, docs_dir, base='supplement'):
    docs_dir = Path(docs_dir)
    parts = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             f'<title>{e(TITLE)}</title><style>{CSS}</style></head><body><div class="wrap">',
             f'<h1>{e(TITLE)}</h1>',
             '<p class="muted">Companion to the DMRN+21 abstract by Pablo Tablas de Paula, Rodrigo Diaz, Emmanouil Benetos and Joshua D. Reiss (Centre for Digital Music, Queen Mary University of London; Rodrigo Diaz also Fraunhofer HHI, Berlin). '
             f'Code: <a href="{REPO}">{REPO.replace("https://", "")}</a>.</p>',
             '<p>The abstract shows that a nonlinear FDTD Kirchhoff–Carrier string can be backpropagated quickly and in flat memory with the adjoint method and '
             'checkpointing, compiled into a CUDA kernel. '
             'This page shows two things those gradients are for: fitting the string to recorded guitar notes, and steering it with text through a CLAP embedding. '
             'The optimisations below ran on a variant of that kernel which also caches three scalars per time step; its gradients agree with the '
             'checkpointed adjoint to 2.7&times;10<sup>&minus;11</sup> (<a href="#verification">Section 3</a>). '
             'Every optimisation below keeps the string\'s physical parameters inside a box by a sigmoid reparameterisation, uses Adam with per-candidate plateau decay, '
             'and runs many candidates as one GPU batch.</p>']
    if results_path is not None:
        with open(results_path) as handle:
            parts.append(sound_matching_section(json.load(handle), base+'/sound_matching'))
    if semantic_runs:
        parts.append(semantic_section(semantic_runs, base+'/semantic'))
    parts.append(verification_section())
    parts.append('<h2 id="reproduce">Reproduction</h2><pre>'
                 'python -I experiments/applications/sound_matching/data.py --splits test\n'
                 'python experiments/applications/sound_matching/fit.py --manifest .../selected/manifest.json --families steel,nylon --output RUN\n'
                 'python experiments/applications/sound_matching/evaluate.py --run RUN --assets docs/supplement/sound_matching --output docs/supplement/sound_matching/results.json\n'
                 'python experiments/applications/semantic/optimise.py --pair metallic_dull --output RUN_CLAP\n'
                 'python experiments/applications/site.py --sound docs/supplement/sound_matching/results.json --semantic RUN_CLAP ...</pre>'
                 '<p class="small muted">Sources: NSynth (Engel et al. 2017); cels-audio (Tablas de Paula et al., ICASSP 2027 submission); CLAP laion/clap-htsat-unfused (Wu et al. 2023); '
                 'Text2FX (Chu et al., ICASSP 2025); D\'Addario string tension specifications; Woodhouse et al., Acta Acustica 98 (2012) Table I.</p>')
    parts.append('</div></body></html>')
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir/'index.html').write_text('\n'.join(parts))
    (docs_dir/'.nojekyll').touch()
    return docs_dir/'index.html'


def collect_semantic(run_dirs, assets_root):
    """Copy a CLAP run's audio, images and GIFs under assets_root/<pair>/ and return the run records."""
    runs = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        with open(run_dir/'summary.json') as handle:
            summary = json.load(handle)
        target = Path(assets_root)/summary['pair']
        target.mkdir(parents=True, exist_ok=True)
        assets = {}
        for sub in ('audio', 'img'):
            for path in sorted((run_dir/sub).glob('*')) if (run_dir/sub).exists() else []:
                shutil.copy(path, target/path.name); assets[path.name] = f"{summary['pair']}/{path.name}"
        for path in sorted(run_dir.glob('dir*.gif')):
            shutil.copy(path, target/path.name); assets[path.name] = f"{summary['pair']}/{path.name}"
        runs.append({'summary': summary, 'assets': assets})
    return runs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sound', type=Path, default=None, help='sound-matching results.json (assets already under docs/supplement/sound_matching)')
    parser.add_argument('--semantic', type=Path, nargs='*', default=[], help='CLAP run directories (assets are copied)')
    parser.add_argument('--docs', type=Path, default=ROOT/'docs')
    args = parser.parse_args(argv)
    runs = collect_semantic(args.semantic, args.docs/'supplement'/'semantic') if args.semantic else []
    path = build(args.sound, runs, args.docs)
    print(f'wrote {path} ({path.stat().st_size//1024} kB)')


if __name__ == '__main__':
    main()
