# CPU and CUDA backends

`Renderer(..., device="cuda", backend="modal")` runs the full-grid modal
implementation; `backend="spatial"` selects the fused spatial implementation.
The low-level `ForwardString` selects the device from its input tensor.
`backend="native"` remains the original CPU default; on CUDA it selects the
spatial CUDA kernel. All paths use float64 and first derivatives only.

```python
from src.parameters import ParameterBox
from src.grid import GridConfig
from src.synthesis import Renderer
from src.optimization import fit

box = ParameterBox()
grid = GridConfig(box)
renderer = Renderer(grid, duration=4., device="cuda", backend="modal")
target = renderer.synthesize(box.defaults())
result = fit(renderer, target, box.perturb(box.defaults()), optimizer_name="adam")
```

CPU input values may be passed to `Renderer`; the differentiable device transfer
is explicit inside that renderer. For repeated fitting, `fit` moves the parameter
vector and target once. Returned waveforms and fitted tensors live on the selected
device. Use `.detach().cpu()` for plotting or NumPy export. Notebook controls expose
the same choices, and re-preparation is required after changing them.

## What is preserved

The modal backend is a change of coordinates of the **discrete** spatial
operator. With N intervals, the orthonormal sine matrix has entries
`S[j,k] = sqrt(2/N) sin(pi*j*k/N)`, j,k=1,...,N-1, and the dimensionless
second-difference eigenvalues are `-4 s²` for the three-point stencil
(`order=2`) and `-4 s² (1 + s²/3)` for the default five-point stencil
(`order=4`), `s = sin(pi*k/(2*N))`, both from
`kc_string_forward.discrete_eigenvalues`. Every CUDA kernel takes this table as
input and never recomputes it. Physical spacing and its length derivatives enter
through the coefficients. All N-1 modes are retained. The fourth derivative is
the square of that same operator; the spatial backend applies the five-point
second difference and its nine-point square directly on the odd extension of
the grid. Startup states are computed by the existing implementation and
transformed together with their material sensitivities. The pickup is a per-item dual weight vector: the sine interpolant at the pickup position (parameter 10, a fraction of L), whose value column is the corresponding row of S when the pickup sits on a node (the
nominal midpoint) and the interpolant otherwise. The adjoint kernels also return
d loss/d w per mode, which the Python wrappers chain through the weights' pickup tangent.

Thus the temporal recurrence, simply supported boundary conditions, nonlinear
coupling, parameter box, excitation, and output filter remain unchanged. This
is not continuous-modal synthesis, mode truncation, or a different SAV scheme.
Floating-point arithmetic and reduction order differ, so cross-backend agreement
is numerical rather than bitwise.

For spatially uniform inverse mass m, the nonlinear solve is rearranged as

```
a = dot(q, previous)
b = dot(q, q)
c = dot(q, linear)
next = m * (linear - q * (a + m*c)/(1 + m*b))
```

The three sums can be reduced together. Every quantity carries its primal value
and ten tangents (eight material parameters, then the pluck and pickup positions). Fixed spatial-spacing factors are absorbed into
coefficients once per call. Synthesis without gradients uses a one-column kernel.

CUDA uses one thread block per independent string and loops over time on-device.
The modal path uses one warp per sensitivity column, with spatially contiguous
shared state and precomputed modal coefficients. This parallelises derivative
arithmetic as well as modes and avoids the register pressure of nine-column
per-thread arithmetic. A register-state variant and the general shared/global
workspace paths handle configurations outside this path's memory capacity.
Spatial kernels use shared memory, with global workspace for larger grids. There is no inter-block
synchronisation. `chunk_steps` bounds the number of steps per kernel launch
(default 8192; zero means the whole render). `threads` accepts 128 or 256 for spatial and fallback kernels. The modal
channel-parallel path uses 32 threads per column (288 with tangents), independent
of that tuning setting.

The internal `src.accelerated.evaluate` accepts `(batch,8)` material multipliers
on a common grid, duration, rate and excitation. It returns independent audio and
tangent arrays; it deliberately does not implement batched optimisers or a public
batched parameter-box interface. Initialization is vectorised across batch items.
Tangent storage still scales with internal-rate duration and batch size.

## Cluster environment and commands

The original Pixi CPU environment is unchanged. The separate Linux GPU environment
uses `hpc/requirements.lock` (PyTorch 2.14.1, pinned cels source) and
`hpc/environment.sh`. Setup records the fully resolved versions on scratch.
On this cluster the PyTorch wheel supplies CUDA 13.0 and nvcc is provided by the
CUDA 13.1.1 module. The notebook/reference worker uses a separate scratch Pixi
cache for Octave 10.3.0, avoiding incomplete entries in older shared caches. The extension targets A100 and Hopper (SM80/SM90).

All GPU setup, builds, validation and benchmarks are submitted with `hpc_sonic`:

```sh
python ../hpc_sonic/hpc_sonic.py hpc/setup.json --constraint ampere --max-gpus 1 --submit --state /data/scratch/acw794/kc-string-gpu/jobs/setup-new.json
python ../hpc_sonic/hpc_sonic.py hpc/check.json --constraint ampere --max-gpus 1 --submit --state /data/scratch/acw794/kc-string-gpu/jobs/check-new.json
python ../hpc_sonic/hpc_sonic.py hpc/benchmark.json --constraint hopper --max-gpus 1 --submit --state /data/scratch/acw794/kc-string-gpu/jobs/hopper-new.json
python ../hpc_sonic/hpc_sonic.py hpc/recovery.json --constraint hopper --max-gpus 1 --submit --state /data/scratch/acw794/kc-string-gpu/jobs/recovery-new.json
python ../hpc_sonic/hpc_sonic.py hpc/notebooks.json --constraint ampere --max-gpus 1 --submit --state /data/scratch/acw794/kc-string-gpu/jobs/notebooks-new.json
```

Use a fresh submission-record path each time; inspect prior jobs before retrying.
The helper supports `sae` with account `pilot_sae_gpu`, plus existing `gpushort`
and `andrena`. Hardware classes use separate campaigns and runtime estimates.
Queue estimates are advisory. Build directories are separated by host to avoid
reusing CPU-native binaries across incompatible nodes.

The check worker tests values, tangents, parameter-chain gradients, primal-only
execution, subsampling, zero steps, odd chunk boundaries, batched isolation,
full-state energy conservation, long renders, CUDA streams and large-grid global
workspace. The benchmark reports matched CPU/CUDA execution, separate warmup,
primal synthesis, forward tangents, gradient contraction, full CeL evaluations,
launch tuning, and batch throughput. It does not infer fitting speed from
kernel-only timings.

## Relation to the suggested paper

Lee, Choi and Lee, [String Sound Synthesizer on GPU-accelerated Finite Difference
Scheme](https://arxiv.org/abs/2311.18505), describe spatial and batch parallelism
in a PyTorch string simulator. Those parallel dimensions also apply here. This
implementation additionally parallelises the ten sensitivity columns while
keeping the existing Russo-derived recurrence and forward-mode differentiation.
Time remains sequential within each string. The modal transform diagonalises the
linear spatial operators without removing any discrete modes.

See [measured GPU results](gpu_results.md) for latency, throughput and validation.
