# Four-second benchmark listening comparison

`01-fdtd.wav` and `02-jax-175.wav` are the first batch item from the A100
40 GB pilot, job 34555111. `fdtd-then-jax.wav` plays the same FDTD waveform,
one second of silence, and then the JAX waveform. Its stereo channels are
identical; the individual files are mono. All audio is 44.1 kHz / PCM24.

The files originate from
`/data/scratch/acw794/kc-string-gpu/native-rate-pilot-34555111-logs/diaz-44100-m175-b1.npz`.
The producer `benchmarks/native_rate_worker.py` writes JAX to `audio` and the
independently generated FDTD waveform to `reference_audio`. The FDTD audio is
the native CPU evaluation of the same discrete recurrence, not an exported
three-scalar CUDA rendering; the latter has been separately validated against
the CPU implementation. The FDTD waveform was simulated at 132.3 kHz and
antialias-filtered to 44.1 kHz; JAX used 175
modes at 44.1 kHz. These are the same physical values and the same first-item
signals used for the reported 3.28% relative L2 comparison. The excitation
contains eight sine modes and is not the raised cosine.

The same gain is applied to both signals, taking the larger original peak to
0.9. No equalization, per-signal normalization, fading or other processing
is applied; both the level relationship and the initial conditions are
preserved apart from PCM24 quantization. Run `export_audio.py` with the
project's scratch Python environment to reproduce the files.
