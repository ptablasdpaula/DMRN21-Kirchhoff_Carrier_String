# Companion applications: sound matching and semantic conditioning

Two applications of the differentiable string, built for the DMRN+21 online
supplement (`docs/index.html`, published with GitHub Pages from the `docs`
folder). Both reuse the batched cached adjoint through
`experiments/applications/common/render.py:BatchRenderer`, which renders a
`(B, 10)` batch of physical parameters at any output rate (the internal rate is
`sample_rate × oversampling`), and the per-candidate Adam optimiser with plateau
decay and lowest-loss retention in `common/optim.py:fit_batch`, a port of the
ICASSP27-Phrase optimiser. Candidates live in a bounded box through a sigmoid
reparameterisation (`common/boxes.py`), so the stability bound checked at the
box corner by `GridConfig` holds for every iterate.

## Sound matching on NSynth acoustic-guitar notes (`experiments/applications/sound_matching/`)

* `data.py` downloads the NSynth test split, keeps acoustic-guitar notes
  without the `reverb`, `distortion`, `multiphonic` or `tempo-synced`
  qualities (327 of 481 in the test split), selects one note per requested
  pitch with distinct velocities, trims each to its first 3 s and writes
  `selected/manifest.json`.
* `boxes.py` defines two string families (homogeneous equivalents of
  steel-core and nylon-core strings) and the two-stage grids at 16 kHz output:
  stage A on the whole family at 10× (68 intervals for steel, 60 for nylon,
  set by the family's highest fundamental), stage B inside a narrow box around
  the stage-A winner at the smallest oversampling (6× upwards) that gives
  about 1.5 × the partials below Nyquist (172 intervals for E2 down to 56 for
  E5).
* `loss.py` is the cumulative energy loss (`cels-audio`, log-weighted,
  time-forward directions) on unit-RMS waveforms, so level is not a parameter
  and the pluck amplitude acts only through the nonlinearity.
* `fit.py` runs 128 quasi-random starts in stage A, keeps the eight best
  candidates with distinct fundamentals, refines them in stage B, and writes
  `fit.json`, `fit.wav`, spectrograms and an animated GIF of the optimisation
  per (note, family). The pitch label is never used in the fit.
* `catalogue.py` interprets a fit: the waveform fixes `f0 = sqrt(T/mu)/(2L)`,
  `B = pi^2 EI/(T L^2)`, the decays and the positions, not the five material
  parameters individually. For each catalogue string (D'Addario unit weights
  from the archived tension chart; bending stiffness interpolated within the
  construction class from Woodhouse et al. 2012, Table I) and each scale
  length and fret, the tension at the fitted `f0` and the inharmonicity are
  predicted and ranked by log-distance to the fitted `(mu, T, B)`.
* `evaluate.py` aggregates the fits into `results.json` and copies the page
  assets.

Jobs: `hpc/sound-matching-smoke.{sh,json}` (GPU test plus a short fit) and
`hpc/sound-matching.{sh,json}` (all notes, both families).

## Semantic conditioning with CLAP (`experiments/applications/semantic/`)

* `box.py` is a nine-coordinate wide box: stress `T/(pi r^2)` 50 MPa–1 GPa,
  radius 0.15–0.6 mm, length 0.5–1 m, amplitude, two dampings, a material axis
  moving (density, modulus) log-linearly from nylon to steel, and the pluck
  and pickup positions. Production grid 48 kHz × 12 with 184 intervals, pilot
  48 kHz × 8 with 148.
* `clap.py` reproduces the HuggingFace CLAP feature extractor in torch
  (repeat-pad to 10 s, Hann/1024/480 power spectrogram, Slaney mel filters,
  10·log10) so gradients reach the waveform, loads
  `laion/clap-htsat-unfused` offline, and implements the Text2FX directional
  objective. `test/clap_frontend.py` checks the torch front end against the
  HF extractor.
* `optimise.py` runs a prompt pair in both directions from the same 32 starts
  (start 0 is the paper's string), 2.5 s clips tiled four times into CLAP's
  window, peak-normalised so loudness is not a lever, with gradient
  checkpointing over CLAP chunks; it writes audio and spectrograms for the top
  three per direction, GIFs, `history.jsonl`, `trajectory.npz` and
  `summary.json`.

Jobs: `hpc/clap-smoke.{sh,json}` and `hpc/clap-guidance.{sh,json}` (one task
per prompt pair).

## Page

`experiments/applications/site.py` renders `docs/index.html` from the
sound-matching `results.json` and the CLAP `summary.json` files, copying the
audio, images and GIFs under `docs/supplement/`.
