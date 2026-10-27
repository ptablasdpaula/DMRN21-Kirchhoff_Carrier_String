Accompanying code for the DMRN+21 submission "Differentiable Kirchhoff Carrier Stiff String" to the [Digital Music Research Network One-day Workshop 2026](https://www.qmul.ac.uk/dmrn/dmrn21/), scheduled for Friday, 11 December 2026. The implementation is based on [Russo, Ducceschi & Bilbao (2026)](https://github.com/Nemus-Project/nLinStringsConv-SAV/blob/main/KCstring/KCStringSAV_noSplit.m)'s `KCStringSAV_nosplit.m` with `SAV = false`.

## Installation

```sh
# macOS / Linux
curl -fsSL https://pixi.sh/install.sh | sh
export PATH="$HOME/.pixi/bin:$PATH"
git submodule update --init --recursive
pixi install

# Windows PowerShell alternative (run separately):
# powershell -ExecutionPolicy Bypass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

## Equations and implementation

Open [the learning notebook](notebooks/kc_string.ipynb) for a standalone PyTorch
walkthrough that builds from basic algebra and Python. It introduces units, mass,
forces, derivatives, integrals, waves, damping, and the numerical update as they
are needed, with rendered equations, worked examples, and plots. No SymPy is used.

All simulation parameters and functions are defined directly in the notebook.
It compares its own output against the MATLAB reference using Octave, preserving
the reference's numerical conventions. The separate implementation in
[`src/kc_string.py`](src/kc_string.py) remains independently usable; edits are not
automatically synchronized between the two.

Run `pixi install` after updating the repository. In VS Code, install the Python
and Jupyter extensions, open the notebook, then choose **Select Kernel → Python
Environments** and select this project's `.pixi/envs/default/bin/python`
(`.pixi/envs/default/python.exe` on Windows). Use **Run All** to execute in order.
After editing the notebook's parameters, restart the kernel and run all cells again.

The notebook uses a short demonstration and a short comparison at the reference
resolution. For the full 0.06-second comparison, run
`compare_with_reference(duration_seconds=0.06)` in a separate cell; this takes
substantially longer. The supplied comparison assumes the default physical parameters.
The Octave timeout is 900 seconds; pass `timeout_seconds=3600` to allow a longer run.

```sh
pixi run test             # Test the separate Python module against Octave
pixi run notebook-check   # Run the standalone notebook, including its Octave check
```

The executed notebook is written to `.notebook-build/kc_string.executed.ipynb`.
Octave is included in Pixi on macOS arm64 and Linux; the waveform test on Windows
requires Octave to be installed separately and available on PATH. The notebook's
simulation and lessons can run without Octave, but its final reference check requires it.

## Interactive parameter ranges

- [Fixed-rate explorer](notebooks/stability_ranges.ipynb): 44,100 Hz with no oversampling.
  Change parameters and see the linear stability score, conditional parameter limits,
  maximum interval count, and a joint feasible-region plot.
- [Automatic-oversampling explorer](notebooks/oversampling_ranges.ipynb): the same base
  rate, with the smallest sufficient power-of-two oversampling factor selected after
  each edit. Its plot shows the factor required across a pair of parameters.

Both are standalone notebooks and start with 128 spatial intervals. Run `pixi install`,
select the Pixi kernel in VS Code, and **Run All**. Edit a numeric control and press
Enter or leave the field to refresh the live view. The kernel must remain running;
if widgets appear as text or placeholders, check that the VS Code Jupyter extension
is enabled and that the selected environment includes `ipywidgets`. A saved initial
snapshot remains readable without live widgets.

Inputs are never clipped. Conditional table rows hold all other values and the displayed
working rate fixed; they cannot be combined into independent optimisation bounds.
Plot limits change only the viewing window. These notebooks assess the linear grid
bound with the reference's 1% spacing margin, not full nonlinear stability or gradient
conditioning. The automatic adviser does not change time steps inside a simulation.

```sh
pixi run ranges-check     # Boundaries, live callbacks, plots, and fresh-kernel execution
```

Executed copies are written to `.notebook-build/`. No waveform simulation or Octave
installation is needed for these two explorers.

## Differentiable forward-mode simulator

[`src/kc_string_forward.py`](src/kc_string_forward.py) provides `ForwardString`,
a first-order differentiable CPU simulator. Its C++ kernel uses raw arrays with
nine columns per grid point: displacement and eight parameter derivatives.
One native call runs the entire time loop. The loss stays in PyTorch, and backward
contracts its audio gradient with the saved output tangents. Adam and L-BFGS work
through the usual `loss.backward()` interface; second derivatives are unsupported.

Run this example from the repository root in the Pixi environment:

```python
import torch
from src.kc_string_forward import ForwardString, PARAMETER_NAMES

torch.set_num_threads(1)
string = ForwardString(intervals=128, sample_rate=44_100)
theta = torch.nn.Parameter(torch.ones(8, dtype=torch.float64))
optimizer = torch.optim.Adam([theta], lr=1e-3)

optimizer.zero_grad()
audio = string(theta, duration_seconds=0.06)
loss = (audio / 0.002).square().mean()  # Replace with your own differentiable loss.
loss.backward()
optimizer.step()
```

`theta` contains relative multipliers of the reference density, tension, radius,
Young's modulus, length, excitation amplitude, `sigma_0`, and `sigma_1`, in that
order (`PARAMETER_NAMES`). The damping coefficients are independent parameters
here. Keep the grid and sampling rate fixed, and constrain parameter updates to
the feasible region described by the range notebooks. Inputs are not clipped.
The defaults here use no oversampling; they differ from the original MATLAB
comparison's much finer grid.

`string.output_and_tangents(theta, duration_seconds)` also returns detached audio
and its `(n_out, 8)` Jacobian. `output_stride` in the constructor selects every nth
output and tangent row without skipping integration steps. This is observation
subsampling, not anti-alias filtering. Keep stride one and apply an appropriate
PyTorch filter when producing downsampled audio. The two reference startup
samples are included. Working state memory is fixed; saving the output Jacobian
costs 64 bytes per returned sample (about 10.8 MiB for 4 seconds at 44,100 Hz).

The first call builds a host-optimised C++ extension with strict floating-point
arithmetic and caches it. A C++ compiler is required (Xcode Command Line Tools on
macOS); Ninja is included in Pixi. The kernel has no OpenMP loop. It releases the
Python GIL and uses independent buffers, so separate simulations can run in
threads after warming up the extension. Parallel speedup has not been benchmarked.

The compiled nine-column PyTorch implementation (`DualReference`) and compiled
checkpointed reverse-mode remain validation references, not production modes.
The checks cover audio, gradients, subsampling, custom spectral losses, optimizer
integration and agreement with the original Octave script.

```sh
pixi run forward-check
pixi run test
pixi run python benchmarks/native_autograd.py
```

## Adam and L-BFGS experiment

```sh
pixi run python experiments/optimize_cel.py
pixi run python experiments/plot_optimization_cel.py  # Replot saved results only.
```

Both optimizers start from the same signed 10% perturbation of all eight
parameters and fit a 4-second reference signal at 44,100 Hz with 128 intervals.
Excitation shape and midpoint observation stay fixed. The loss is the upstream
`STFTCumulativeEnergyLoss(sample_rate=44100)` with all other defaults; the
`cel-library` source commit is pinned in the Pixi environment. Target features
are cached without changing the loss.

Optimisation uses log parameter multipliers. Positive multipliers must remain
between 0.25 and 4 and satisfy the fixed-grid linear stability bound; infeasible
proposals are rejected before simulation. No such rejections occurred in the
saved runs. Each run stops when its best loss improves by less than 10% over
20 completed optimizer steps. L-BFGS line-search evaluations are counted
separately from steps.

Results and complete trajectories are saved in
[`experiments/optimization_cel_results.json`](experiments/optimization_cel_results.json).
Loss curves, target/initial/predicted spectrograms, parameter comparisons and
common-gain WAV files are generated in `.notebook-build/optimization_cel/`.
Spectrograms share a colour scale and use the same STFT settings as the loss.
The best-loss checkpoint is used for predictions.

A good signal fit does not uniquely identify every physical parameter. For
example, scaling density, tension and Young's modulus together leaves the
ratios in this model unchanged. Parameter recovery is reported separately
from the loss and waveform error.

## Interactive optimisation notebook

Open [`notebooks/optimization_workbench.ipynb`](notebooks/optimization_workbench.ipynb)
in the Pixi kernel and run all cells once. Its six panels let you:

1. Enter target parameters, audio rate, duration, grid and oversampling; select
   any of the eight parameters with checkboxes; generate and play the target.
2. Enter initial parameters independently, or copy/perturb the target; generate
   and play the initial signal using the same sampling configuration.
3. Run Adam with the selected parameters.
4. Play its best prediction and inspect parameters, loss curves and spectrograms.
5. Run L-BFGS independently from the same initial signal.
6. Play and inspect its best prediction in the same way.

Unchecked parameters remain at their initial values. Both optimizers keep the
20-step / 10% patience rule. Edits invalidate affected results. The first sine
mode and midpoint listener remain fixed; position controls are not part of this
model. Oversampling uses a differentiable low-pass filter before decimation.
A live notebook kernel is required for buttons and playback.

Run `pixi run workbench-check` to execute the notebook and test selections,
independent starts, zero selection/damping, stale-result invalidation, resampling
and the patience rule. Supporting code is in `src/optimization_workbench.py`.

## Fundamental-frequency / inharmonicity synthesis

[`notebooks/inharmonicity_synthesis.ipynb`](notebooks/inharmonicity_synthesis.ipynb)
is a separate **synthesis-only** notebook. It controls the small-amplitude
fundamental `f1` (including stiffness) and inharmonicity `B`, deriving tension and
Young's modulus at the selected density, radius and length. It includes the
`B = 0` limit, audio playback, waveform/spectrum plots, predicted partial
frequencies and an optional multimode excitation to make partial spacing audible.
There are no losses or optimizers. Run all cells in the Pixi kernel, then click
**Synthesise**. Run `pixi run inharmonicity-check` to check the mapping and waveform.

[`notebooks/inharmonicity_fixed_grid.ipynb`](notebooks/inharmonicity_fixed_grid.ipynb)
provides synthesis with a **fixed number of spatial intervals and automatic
oversampling**. Its live estimate shows the smallest integer factor satisfying
the reference linear stability bound with a 1% spacing margin, the internal
sampling rate, and the number of simulation steps. The grid never automatically
coarsens; audio is filtered and returned at the chosen output rate. A configurable
oversampling ceiling and a ten-million-step limit prevent unexpectedly large
renders. The notebook also reports linear modal detuning: satisfying stability
preserves neither exact modal frequencies nor guaranteed nonlinear accuracy.
Run `pixi run fixed-grid-check` to check rate selection, waveform agreement and
the interactive controls.
