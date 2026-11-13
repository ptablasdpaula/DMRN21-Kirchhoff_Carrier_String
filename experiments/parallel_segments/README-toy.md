# Parallel segment adjoint: dense toy scan

This is a research proof of concept for the user's parallel-segment idea. It is
**not a replacement for the production kernel**, and does not establish a
full-grid speedup.

Each segment is anchored by an independently saved pair of end states. All
segments reconstruct their states backwards concurrently. While reconstructing,
each builds an affine map from the ending state adjoint to its starting adjoint,
plus a map for eight coefficient-gradient directions. An associative suffix
scan connects these maps and accumulates the parameter gradients.

The trajectory follows the same nonlinear diagonal-plus-rank-one recurrence as
our modal FDTD backend, but uses synthetic stable coefficients and short toy
signals. The eight directions are arbitrary coefficient perturbations rather
than the production parameter box. Gradients for the initial states are returned
separately and can be chained through an initial-condition map. Audio filtering
and losses beyond an arbitrary upstream signal are not represented here.

In CPU and A100 GPU job 34308162, tests passed for 2, 4, 8 and 16 modes, batch
1 and 3, and chunks of 1, 4 and 8 steps. State and eight parameter gradients
matched independent PyTorch autograd. The scaling run tested up to 64 modes,
batch 4 and 256 steps. Recorded [raw measurements](toy-a100-34308162.json) are
for **eager PyTorch**, including Python and synchronization overhead, and cannot
be compared directly with optimized production CUDA kernels.

For a production four-second signal at 132,300 Hz, N=176 intervals (175 modes),
batch 128 and chunk length 256, these dense state and parameter segment maps
alone require approximately 266.1 GB (decimal), before intermediate scan copies,
checkpoints, output, loss and gradients. A 1024-step chunk reduces that base
storage to 66.5 GB, but extra scan buffers and numerical stability still have to
be considered. These are storage estimates, not measured full-grid allocations.
The local Jacobian-times-map update exploits the diagonal and rank-one structure
in O(N²) operations per step, but composing the dense maps requires expensive
matrix products and multiple scan copies. A production approach needs either a
compact representation of composed Jacobians or another way of avoiding dense
per-segment maps; no such representation has yet been established here.

The alternate full-grid GPU experiment that reconstructs segments concurrently
and then performs a sequential adjoint is separate. Neither approach has been
shown to avoid cross-segment gradient dependencies for free.
