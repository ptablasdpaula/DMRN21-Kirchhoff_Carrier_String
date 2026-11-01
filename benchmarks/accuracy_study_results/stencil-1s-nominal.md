# Spatial stencil study: nominal string, raised cosine, 1 s

Reference: modal 123 modes, 44.1 kHz, band-matched (1x project filter). Time step unchanged; only the spatial operator changes.

| Intervals | Oversampling | Order | Work vs benchmark | Stability score | Waveform rel. L2 | FFT-mag. rel. L2 |
|---:|---:|---|---:|---:|---:|---:|
| 176 | 3x | 2 | 1.00 | 0.192 | 1.168e-01 | 4.094e-02 |
| 176 | 3x | 4 | 1.00 | 0.322 | 4.531e-03 | 2.205e-03 |
| 176 | 3x | 6 | 1.00 | 0.405 | 5.020e-03 | 2.464e-03 |
| 176 | 3x | spectral | 1.00 | 1.015 | 5.027e-03 | 2.470e-03 |
| 176 | 6x | 2 | 2.00 | 0.048 | 1.188e-01 | 4.204e-02 |
| 176 | 6x | 4 | 2.00 | 0.081 | 1.782e-03 | 7.185e-04 |
| 176 | 6x | spectral | 2.00 | 0.254 | 1.816e-03 | 6.993e-04 |
| 256 | 4x | 4 | 1.94 | 0.737 | 3.004e-03 | 1.382e-03 |
| 256 | 4x | 6 | 1.94 | 0.937 | 3.129e-03 | 1.449e-03 |
| 256 | 6x | 4 | 2.91 | 0.328 | 1.726e-03 | 5.944e-04 |
