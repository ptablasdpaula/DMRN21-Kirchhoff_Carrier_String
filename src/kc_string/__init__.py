"""Kirchhoff–Carrier string synthesis with a hand-written GPU adjoint.

Modules:
    physics     the nominal steel string and closed-form string quantities
    scheme      the finite-difference scheme, carried with forward-mode tangents
    boxes       bounded parameter boxes and the benchmark workload
    grid        one stable grid per parameter box
    adjoint     the hand-written adjoint as a CUDA kernel (the paper's method)
    render      batched rendering at an output rate through the kernel
    jax_scheme  the same scheme differentiated automatically by JAX (the paper's baselines)
"""
