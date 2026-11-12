# DMRN+21 2026 draft

`main.pdf` is the one-page A4 submission draft; its source is `main.tex`.
`technical_note.pdf` is a separate, longer technical explanation and is **not**
part of the one-page submission; it uses `references.bib` and documents the
scalar-cached variant of the adjoint, which the abstract no longer presents.
`main.tex` carries a hand-written bibliography (titles kept, venues abbreviated,
e.g. `Front. Sig. Proc., v. 3, 1284100, 2024`). The author order is
Pablo Tablas de Paula, Rodrigo Diaz, Emmanouil Benetos and Joshua D. Reiss;
Rodrigo Diaz carries a second affiliation, Fraunhofer HHI, Berlin.

`main-two-frameworks.tex` (with its PDF) is a test layout of 12 November 2026, not the submission: the table
shows eight rows grouped by framework (PyTorch: Auto, + compiled, + checkpoints, + modes; JAX: Auto (compiled),
+ checkpoints, + modes; Hand-written) and the text is rewritten around it. The left column ends with the
Backpropagation paragraph and the acknowledgement (UKRI, Apocrita, Eloi Moliner); the four numbered footnotes
(supplement, PyTorch, JAX, CUDA) are placed at the foot of the right column, the last three on one line, so
that both footnote rules sit at the same height. The gap under the table is 6 pt and the title gaps are
1.2, 1 and 1 em. Its table comes from
`python figures/build_benchmark_summary_table.py two-frameworks`
(`figures/benchmark-table-two-frameworks.tex`, data in `figures/native-rate-two-frameworks-merged.json`); the
measurements are in `docs/native_rate_study.md`.

The abstract's subject (since 11 November 2026) is the established recipe
rather than a bespoke technique: the adjoint method with checkpointing,
compiled. Checkpointing fixes memory, compilation fixes time, and the
hand-written CUDA kernel is measured against compiled autodiff as well as
eager autodiff. The previous source is `variants/main-before-checkpointed-adjoint.tex`.

The official 2026 call requires a one-page A4 abstract and gives a submission
deadline of 14 November 2026: <https://www.qmul.ac.uk/dmrn/dmrn21/>. The
unchanged official archive is retained in `template-source/`. Its style still
contains 2025 event text and its sample PDF is Letter-sized. The working copy
`dmrn2026.sty` updates the event header/footer for 11 December 2026 and fixes
an extra space between the header's minipages; `main.tex` selects A4 and the
template's intended Times font. Side and top margins, columns, the title block
gaps and the footer distance are the template's. Deliberate departures, all set
in the preamble of `main.tex`: the text block is 266 mm tall against the
template's 250 mm, the body uses the style's optional `\ninept`, and the
paragraph, display, caption, table-row and reference spacings are tightened.
The fully tightened layout (title block and footer included) is
`variants/main-layout-tight.tex`; `variants/main-layout-text-height-255.pdf`
is the alternative that spends the spare room on a 255 mm text block instead.

Build from this directory with Tectonic and an available TeX bundle:

```sh
tectonic --keep-logs main.tex
tectonic --keep-logs technical_note.tex
```

For the current cluster environment, replace `tectonic` with
`PIXI_CACHE_DIR=/data/scratch/acw794/dmrn-tex/pixi-cache XDG_CACHE_HOME=/data/scratch/acw794/dmrn-tex/xdg-cache pixi exec --spec tectonic tectonic`.

The table and its reproducible source/data are in `figures/`. The measured
workload is one sound matching step: a batch of strings, each of the ten
parameters (eight material parameters plus the pluck and pickup positions as
fractions of L) moved by up to ±3 % from the nominal string, is simulated for
one second with the reference code's raised-cosine pluck, the mean absolute
error of each waveform against the nominal string's is computed, and it is
backpropagated to all ten parameters (`benchmarks/exact_methods.py:varied_material`,
`--loss mae`). The excitation is the same as in the Octave validation.

`figures/build_benchmark_summary_table.py` writes `figures/benchmark-table.tex`,
the abstract's Table 1: peak GPU memory and time of one gradient step against
batch size for five implementations of the same gradient, each adding one
ingredient to the one before (the rows marked with the step arrow):

| Row | Code name | What it is |
|---|---|---|
| Autodiff | `fdtd` | eager PyTorch autodiff of the scheme on the grid (`experiments/figure_comparison/fdtd.py`) |
| + compiler | `fdtd_jax` | the same arithmetic, operation for operation, as a `jax.jit` scan that keeps every step (`experiments/figure_comparison/fdtd_jax.py`, `basis='grid'`) |
| + checkpoints | `fdtd_jax_ckpt` | the same with `jax.checkpoint` on 256-step segments |
| + modes | `fdtd_jax_sine_ckpt` | the same stepped in the discrete sine basis (`basis='sine'`), in which the stencils are diagonal |
| Hand-written | `replay` | `src/adjoint.py` and `src/kc_adjoint.cu`: the hand-derived adjoint with the same checkpoints and modes in one CUDA kernel |

The script reads `figures/native-rate-one-host-merged.json`, which
`figures/merge_sweeps.py` builds from job 34855605 (Autodiff), job 35747154
(+ compiler), job 35747153 (+ checkpoints), job 35746438 (+ modes) and job
35746439 (Hand-written). All five ran on the cluster's `rdg` hosts (Xeon Gold
6442Y, A100 80 GB PCIe): eager PyTorch is bound by the host CPU and is about
1.6 times slower on the `sbg` hosts, so every row shares one host type
(`docs/native_rate_study.md`, "The eager checkpointed baseline" and "The
ladder of the abstract"). The compiled rows and the kernel are not sensitive
to the host: their earlier `sbg` measurements, in
`figures/native-rate-sweep-merged.json` with every other row, agree with
these to 2 %. Without the one-host file the script falls back to that one.

Rows that are no longer in the abstract stay in
`figures/native-rate-sweep-merged.json` and are written up in
`docs/native_rate_study.md`: eager PyTorch with checkpoints
(`fdtd_ckpt`, `fdtd_ckpt_reentrant`), `torch.compile`
(`fdtd_ckpt_compiled`), the sine basis without checkpoints
(`fdtd_jax_sine`), the cached adjoint (`cached3`, three scalars per step,
formerly "Ours": a constant factor of 1.5 over the checkpointed adjoint) and
the jaxdiffmodal rows (the modal solver appears only as a citation). The
previous page, with a four-series figure of memory against batch size
(`figures/build_benchmark_dots.py label`), is
`variants/main-four-series-figure.tex`; the alternatives explored earlier
(dual-axis figure, two-panel figure, dot-size figure, and the five-model
version of this table in commit 2d5372d) remain as scripts in `figures/`
with their outputs in `figures/variants/`, and the page variants built around
them are kept in `variants/` for the record and no longer build in place.
FDTD uses the fourth-order (five-point) spatial stencil on 176 intervals at
2× oversampling (88.2 kHz), a grid that is stable over the benchmark's narrow
`BenchmarkBox`.
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
