# DMRN+21 2026 draft

`main.pdf` is the one-page A4 submission draft; its source is `main.tex`.
`technical_note.pdf` is a separate, longer technical explanation and is **not**
part of the one-page submission. Both use `references.bib`. The author order is
Pablo Tablas de Paula, Emmanouil D. Benetos and Joshua D. Reiss.

The official 2026 call requires a one-page A4 abstract and gives a submission
deadline of 14 November 2026: <https://www.qmul.ac.uk/dmrn/dmrn21/>. The
unchanged official archive is retained in `template-source/`. Its style still
contains 2025 event text and its sample PDF is Letter-sized. The working copy
`dmrn2026.sty` updates the event header/footer for 11 December 2026 and fixes
an extra space between the header's minipages; `main.tex` selects A4 and the
template's intended Times font. Margins, columns, font sizes and line spacing
follow the supplied template.

Build from this directory with Tectonic and an available TeX bundle:

```sh
tectonic --keep-logs main.tex
tectonic --keep-logs technical_note.tex
```

For the current cluster environment, replace `tectonic` with
`PIXI_CACHE_DIR=/data/scratch/acw794/dmrn-tex/pixi-cache XDG_CACHE_HOME=/data/scratch/acw794/dmrn-tex/xdg-cache pixi exec --spec tectonic tectonic`.

The figure and its reproducible source/data are in `figures/`. The measured
workload is one sound-matching step: a batch of strings, each of the ten
parameters (eight material parameters plus the pluck and pickup positions as
fractions of L) moved by up to ±3 % from the nominal string, is simulated for
one second with the reference code's raised-cosine pluck, the mean absolute
error of each waveform against the nominal string's is computed, and it is
backpropagated to all ten parameters (`benchmarks/exact_methods.py:varied_material`,
`--loss mae`). The excitation is the same as in the Octave validation; `figures/build_benchmark_dots.py label` draws peak GPU memory against
batch size from `figures/native-rate-sweep-merged.json` (the original sweep,
job 34855605, merged by `figures/merge_sweeps.py` with the checkpointed
baselines of jobs 35062741 and 35105080): Ours (the cached adjoint),
jaxdiffmodal and Autodiff (direct PyTorch autodiff) as small filled dots
joined by dashed lines, the two checkpointed baselines (`torch.utils.checkpoint`
over 256-step chunks; the upstream jaxdiffmodal recurrence under
`jax.checkpoint` with the pickup read inside the scan) as hollow dots joined
by dotted lines in their method's colour, the first and last evaluation time
of the filled series and the last of the hollow ones printed, the series
named in their colours, and an x on the 80 GB line at the first batch that
ran out of memory. The alternatives explored on the way (dual-axis figure,
two-panel figure, rank/log-shaded heat-map table, dot-size figure) remain as
scripts in `figures/` with their outputs in `figures/variants/`; the page
variants built around them are kept in `variants/` for the record and no
longer build in place.
FDTD uses the fourth-order (five-point) spatial stencil on 176 intervals at
2× oversampling (88.2 kHz), a grid that is stable over the benchmark's narrow
`BenchmarkBox`; Modal runs 100 continuous modes at 44.1 kHz.
`benchmarks/accuracy_study.py` (results in `benchmarks/accuracy_study_results/`,
discussion in `docs/accuracy_convergence_study.md`) refines the FDTD grid
along Russo et al.'s stability path and shows it converges to the modal
solution; with the reference's three-point stencil the benchmark grid was
11.7% away in waveform relative L2, with the five-point stencil about 1% at
2×. The memory curves are NVML sampled per-process memory, not guaranteed
instantaneous peaks.

Layout: the body is set at 9 pt with the style's own `\ninept` option, and
`main.tex` carries a few layout overrides after the DMRN+15 auraloss
one-pager (tighter title block, one-line 9 pt affiliation with the contact
e-mail, 7 pt footer close to the page edge, `\textheight` 266 mm); the style
file itself is unchanged. The paper no longer states the
Octave/finite-difference validation; it lives in the test suite and
`docs/paper_reference_validation.md` (whose "Ten parameters" section covers
the pluck and pickup positions).

The figure's data is the completed one-second sweep, job 34855605
(`figures/native-rate-sweep-34855605.json`, the script's default input; a copy
sits in `benchmarks/native_rate_results/` as
`a100-sweep-1s-mae-si10-34855605.json`; the setup and results are written up
in `docs/native_rate_study.md`). The earlier sweeps, job 34614181 with the
six perceptual controls perturbed and the STFT cumulative-energy loss and the
four-second eight-sine-mode campaign 34557214 with its hand-typed table, are
superseded; their data stays in `figures/` for the record.
