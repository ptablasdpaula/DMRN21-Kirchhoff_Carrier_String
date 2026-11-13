# Efficient Backpropagation for Finite-Difference Nonlinear String Synthesis

Code accompanying the DMRN+21 (2026) paper by Pablo Tablas de Paula, Rodrigo Diaz, Emmanouil Benetos and
Joshua D. Reiss. Online supplement: <https://ptablasdpaula.github.io/DMRN21-Kirchhoff_Carrier_String>.

| Where | What |
| --- | --- |
| `src/kc_string/` | The library: the Kirchhoff–Carrier scheme (`scheme.py`), the hand-written adjoint as a CUDA kernel (`adjoint.py`, `adjoint.cu`), its JAX autodiff baselines (`jax_scheme.py`) and the paper's workload (`workload.py`) |
| `tests/` | Agreement with Russo et al.'s code (`nLinStringsConv-SAV/`, a submodule) and gradient checks of every implementation |
| `benchmark/` | Time and memory of one gradient step (Table 1) |
| `applications/` | Sound matching of recorded guitar notes and text-guided sound design, shown in the supplement |
| `web/` | Builds the supplement page in `docs/`, which GitHub Pages serves |
| `paper_dmrn2026/` | The paper, and Table 1 with its data |
| `hpc/` | Environment and job scripts for the cluster |

```sh
git submodule update --init
pixi run test     # CPU and Octave tests; on a GPU node the CUDA tests also run
pixi run lint
```
