# DMRN+21 paper

`main.tex` (with `dmrn2026.sty`, the DMRN+21 style, and the logo) builds the one-page `main.pdf`:

```sh
tectonic main.tex
```

Table 1 is `figures/benchmark-table.tex`, written by `python figures/build_benchmark_summary_table.py` from the
four benchmark result files beside it, one per row (A100 80 GB, the cluster's rdg hosts).
