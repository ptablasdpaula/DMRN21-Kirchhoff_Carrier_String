# Spatial stencil study: nominal string, raised cosine, 1 s

Reference: modal 123 modes, 44.1 kHz, band-matched (1x project filter). Time step unchanged; only the spatial operator changes.

| Intervals | Oversampling | Order | Work vs benchmark | Stability score | Waveform rel. L2 | FFT-mag. rel. L2 |
|---:|---:|---|---:|---:|---:|---:|
| 176 | 2x | 2 | 1.00 | 0.431 | 1.126e-01 | 3.908e-02 |
| 176 | 2x | 4 | 1.00 | 0.724 | 1.032e-02 | 5.075e-03 |
| 176 | 3x | 2 | 1.50 | 0.192 | 1.168e-01 | 4.094e-02 |
| 176 | 3x | 4 | 1.50 | 0.322 | 4.531e-03 | 2.205e-03 |
| 176 | 6x | 4 | 3.00 | 0.081 | 1.782e-03 | 7.185e-04 |
