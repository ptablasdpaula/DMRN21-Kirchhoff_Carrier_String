# Spatial stencil study: nominal string, raised cosine, 1 s

Reference: modal 123 modes, 44.1 kHz, band-matched (1x project filter). Time step unchanged; only the spatial operator changes.

| Intervals | Oversampling | Order | Work vs benchmark | Stability score | Waveform rel. L2 | FFT-mag. rel. L2 |
|---:|---:|---|---:|---:|---:|---:|
| 176 | 2x | 4 | 0.67 | 0.724 | 1.032e-02 | 5.075e-03 |
| 152 | 3x | 4 | 0.86 | 0.190 | 4.256e-03 | 2.099e-03 |
| 176 | 4x | 4 | 1.33 | 0.181 | 2.730e-03 | 1.210e-03 |
