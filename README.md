Accompanying code for the DMRN+21 submission "Differentiable Kirchhoff Carrier Stiff String" to the [Digital Music Research Network One-day Workshop 2026](https://www.qmul.ac.uk/dmrn/dmrn21/), scheduled for Friday, 11 December 2026. The implementation is based on [Russo, Ducceschi & Bilbao (2026)](https://github.com/Nemus-Project/nLinStringsConv-SAV/blob/main/KCstring/KCStringSAV_noSplit.m)'s `KCStringSAV_nosplit.m` with `SAV = false`.

## Installation

From a clone of this repository:

```sh
# macOS / Linux
curl -fsSL https://pixi.sh/install.sh | sh
export PATH="$HOME/.pixi/bin:$PATH"
git submodule update --init --recursive
pixi install

# Windows PowerShell alternative for installing Pixi (run separately):
# powershell -ExecutionPolicy Bypass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

Use the project’s `.pixi/envs/default/bin/python` as the VS Code Jupyter kernel
(`.pixi/envs/default/python.exe` on Windows). Run all cells, then use the controls.
The native CPU kernel needs a C++ compiler: Xcode Command Line Tools on macOS,
a C++ toolchain on Linux, or Visual Studio Build Tools on Windows. The first call
builds the extension; later calls reuse it. Octave is included on macOS arm64 and
Linux. Windows reference comparisons require a separate Octave installation.

## Start here

| Notebook | Purpose |
| --- | --- |
| [Learning the string](notebooks/kc_string.ipynb) | Physics and calculus from first principles, rendered derivations, standalone PyTorch implementation and Octave comparison. |
| [Synthesis](notebooks/synthesis.ipynb) | Define ranges, prepare a fixed grid, and explore fundamental frequency, inharmonicity, amplitude, radius and damping. |
| [Optimisation](notebooks/optimization.ipynb) | Set target and initial values with sliders, run Adam and L-BFGS independently, then inspect audio, spectrograms and loss histories. |

The teaching notebook intentionally contains its own readable implementation.
The synthesis and optimisation notebooks share production modules under `src/`.

## One parameter box, one grid

The six coordinates are **f₀, B, amplitude, radius, σ₀ and σ₁**. Density and length
are fixed. Here f₀ is the tension-only fundamental; the small-amplitude first-mode
frequency including stiffness is **f₁ = f₀ √(1+B)**. Tension and Young’s modulus
are derived from these coordinates, not independently fitted.

1. Enter ranges and click **Apply ranges / prepare grid**.
2. Preserve at least the nominal reference string’s ×1 spatial resolution:
   148 intervals at 44.1 kHz, with reference damping.
3. Starting at ×1, choose the smallest integer oversampling factor that supports
   that resolution at the worst corner of the box. Then choose the largest stable
   even interval count, using Russo’s exact 1% spacing margin.
4. Keep the grid and internal rate fixed for every target, initial and fitted
   parameter vector inside the box. Filter before returning audio at the output rate.

The default box selects **×3, 132,300 Hz internally and 176 intervals**. Editing
ranges or render settings invalidates prepared audio and results; moving parameter
sliders inside the box leaves the discretisation unchanged. Resource limits reject
expensive configurations instead of silently reducing spatial resolution.

Stability is the selection criterion. Linear damped modal detuning is reported as
a diagnostic; there is no automatic accuracy refinement, spectral masking or
parameter-dependent time step. Stability and energy conservation alone do not
establish accurate nonlinear pitch glides. See [numerical conventions](docs/numerics.md).

## Differentiation and fitting

`src/kc_string_forward.py` retains the low-level `ForwardString` API and eight
material multipliers. A raw-array C++ loop propagates the value and eight material
sensitivities. The six-coordinate layer chains through that Jacobian. PyTorch’s
backward contracts the saved audio tangents with the loss gradient; no reverse
simulation is needed. This supports ordinary Adam and L-BFGS, **first derivatives
only**, CPU float64. The evolving state uses O(N·8) memory; storing audio tangents
uses O(output samples·8), so total memory is not independent of duration. With
oversampling, the current implementation stores internal-rate tangents before
filtered downsampling.

The default loss is the pinned `cels-audio` `STFTCumulativeEnergyLoss` with FFT
1024, hop 256 and no log weighting. Every iteration uses the full four-second
signal and full-band loss. There are no staged windows or loss changes. Positive
parameters use bounded log interpolation; damping uses bounded linear
interpolation. Learning rates act on unconstrained sigmoid coordinates. Selected
endpoint starts are nudged inward for finite logits and recorded explicitly;
frozen parameters remain exact.

Both optimisers stop when best loss improves by less than 10% over 20 completed
iterations, or at the configured maximum (default 500). Line-search evaluations
are counted separately. Six coordinates do not guarantee six well-conditioned
observable directions: excitation, pickup and damping affect identifiability.
Perfect recovery is not an acceptance criterion.

```python
import torch
from src.parameters import ParameterBox, reference_parameters
from src.grid import GridConfig
from src.synthesis import Renderer
from src.optimization import fit

torch.set_num_threads(1)
box = ParameterBox()
renderer = Renderer(GridConfig(box), duration=4., excitation="multimode")
truth = reference_parameters()
target = renderer.synthesize(truth)
result = fit(renderer, target, box.perturb(truth), optimizer_name="adam")
```

## Validation and experiments

```sh
pixi run test                 # Native/PyTorch comparison with the actual Octave reference
pixi run parameters-check     # Parameter map, exact corner bound, minimal OF / maximal N
pixi run synthesis-check      # Excitations, filtered output and six-coordinate gradients
pixi run optimization-check   # Bounded fits, frozen parameters and patience
pixi run workbench-check      # Both notebooks in fresh kernels; controls and plots
pixi run numerics-check       # Linear closed form, energy and nonlinear gradient checks
pixi run forward-check        # Native vs compiled dual and checkpointed reverse references
pixi run notebook-check       # Teaching notebook and its Octave comparison
pixi run recovery             # Four-second, six-parameter Adam and L-BFGS experiment
```

Notebook fits save JSON to ignored `output/`; executed test notebooks and previews
are under ignored `.notebook-build/`. Reproducible experiment records live in
[`experiments/results/`](experiments/results/). The earlier performance measurements
and independent differentiation references remain in [`benchmarks/`](benchmarks/).
Their 0.51-second four-second result used **128 intervals and ×1**, not the new
176-interval, ×3 defaults; see [result provenance](docs/results.md).

Superseded exploratory notebooks and the material-coordinate fitting scripts are
recoverable from baseline commit `2d717c5`. They are no longer active entry points.
The upstream SAV repository remains an unmodified submodule. Fractional time
steps, dynamic grids and second-order optimisation are outside the main workflow.
