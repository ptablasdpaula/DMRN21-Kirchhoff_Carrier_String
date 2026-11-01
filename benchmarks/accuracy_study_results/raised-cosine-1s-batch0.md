# Accuracy study: batch0 string, raised cosine, 1 s, outputs at 44.1 kHz

Reference: `fdtd-1088-48x` (h = 1.01 h_min on Russo et al.'s path). f0 = 78.90 Hz, B = 3.007e-04; 120 continuous modes lie below 22.05 kHz.

| Candidate | Reference | Waveform rel. L2 | FFT-magnitude rel. L2 | Note |
|---|---|---:|---:|---|
| fdtd-176-3x | fdtd-1088-48x | 1.178e-01 | 3.427e-02 | benchmark FDTD grid vs grid-converged FDTD |
| fdtd-264-3x | fdtd-1088-48x | 5.265e-02 | 1.602e-02 | refinement path vs finest |
| fdtd-380-6x | fdtd-1088-48x | 2.604e-02 | 8.513e-03 | refinement path vs finest |
| fdtd-540-12x | fdtd-1088-48x | 1.179e-02 | 4.012e-03 | refinement path vs finest |
| fdtd-768-24x | fdtd-1088-48x | 4.064e-03 | 1.481e-03 | refinement path vs finest |
| fdtd-264-3x | fdtd-380-6x | 2.879e-02 | 8.327e-03 | successive refinements |
| fdtd-380-6x | fdtd-540-12x | 1.509e-02 | 4.982e-03 | successive refinements |
| fdtd-540-12x | fdtd-768-24x | 7.975e-03 | 2.825e-03 | successive refinements |
| fdtd-768-24x | fdtd-1088-48x | 4.064e-03 | 1.481e-03 | successive refinements |
| modal-100-44.1k-raw | fdtd-1088-48x | 4.085e-03 | 1.567e-03 | modal vs grid-converged FDTD |
| modal-100-44.1k-band | fdtd-1088-48x | 4.085e-03 | 1.567e-03 | modal vs grid-converged FDTD |
| modal-100-132.3k | fdtd-1088-48x | 4.087e-03 | 1.567e-03 | modal vs grid-converged FDTD |
| modal-120-44.1k-raw | fdtd-1088-48x | 4.086e-03 | 1.567e-03 | modal vs grid-converged FDTD |
| modal-120-44.1k-band | fdtd-1088-48x | 4.086e-03 | 1.567e-03 | modal vs grid-converged FDTD |
| modal-120-132.3k | fdtd-1088-48x | 4.087e-03 | 1.567e-03 | modal vs grid-converged FDTD |
| modal-100-44.1k-raw | fdtd-176-3x | 1.209e-01 | 3.512e-02 | modal vs benchmark FDTD grid (the sweep diagnostic uses the raw 44.1k signal) |
| modal-100-44.1k-band | fdtd-176-3x | 1.209e-01 | 3.512e-02 | modal vs benchmark FDTD grid (the sweep diagnostic uses the raw 44.1k signal) |
| modal-100-132.3k | fdtd-176-3x | 1.209e-01 | 3.512e-02 | modal vs benchmark FDTD grid (the sweep diagnostic uses the raw 44.1k signal) |
| modal-120-44.1k-raw | fdtd-176-3x | 1.209e-01 | 3.512e-02 | modal vs benchmark FDTD grid (the sweep diagnostic uses the raw 44.1k signal) |
| modal-120-44.1k-band | fdtd-176-3x | 1.209e-01 | 3.512e-02 | modal vs benchmark FDTD grid (the sweep diagnostic uses the raw 44.1k signal) |
| modal-120-132.3k | fdtd-176-3x | 1.209e-01 | 3.512e-02 | modal vs benchmark FDTD grid (the sweep diagnostic uses the raw 44.1k signal) |
| modal-100-44.1k-raw | modal-120-44.1k-raw | 1.186e-05 | 8.352e-06 | modal truncation |
| modal-100-44.1k-band | modal-100-132.3k | 4.731e-06 | 2.458e-06 | modal time stepping |
| modal-120-44.1k-band | modal-120-132.3k | 5.303e-06 | 3.305e-06 | modal time stepping |
| fdtd-176-3x | modal-120-44.1k-band | 1.209e-01 | 3.512e-02 | FDTD grid vs sub-Nyquist modal (band-matched) |
| fdtd-264-3x | modal-120-44.1k-band | 5.605e-02 | 1.716e-02 | FDTD grid vs sub-Nyquist modal (band-matched) |
| fdtd-380-6x | modal-120-44.1k-band | 2.963e-02 | 9.675e-03 | FDTD grid vs sub-Nyquist modal (band-matched) |
| fdtd-540-12x | modal-120-44.1k-band | 1.555e-02 | 5.238e-03 | FDTD grid vs sub-Nyquist modal (band-matched) |
| fdtd-768-24x | modal-120-44.1k-band | 7.998e-03 | 2.910e-03 | FDTD grid vs sub-Nyquist modal (band-matched) |
| fdtd-1088-48x | modal-120-44.1k-band | 4.086e-03 | 1.567e-03 | FDTD grid vs sub-Nyquist modal (band-matched) |

Error ratios between refinement levels; second order in space predicts the (h_coarse/h_fine)^2 column (about 2 on this path, where h shrinks by sqrt 2 per level):

| Coarse | Fine | Against | Error ratio | (h_c/h_f)^2 |
|---|---|---|---:|---:|
| fdtd-264-3x | fdtd-380-6x | next level | 1.91 | 2.07 |
| fdtd-380-6x | fdtd-540-12x | next level | 1.89 | 2.02 |
| fdtd-540-12x | fdtd-768-24x | next level | 1.96 | 2.02 |
| fdtd-176-3x | fdtd-264-3x | modal-120-44.1k-band | 2.16 | 2.25 |
| fdtd-264-3x | fdtd-380-6x | modal-120-44.1k-band | 1.89 | 2.07 |
| fdtd-380-6x | fdtd-540-12x | modal-120-44.1k-band | 1.90 | 2.02 |
| fdtd-540-12x | fdtd-768-24x | modal-120-44.1k-band | 1.94 | 2.02 |
| fdtd-768-24x | fdtd-1088-48x | modal-120-44.1k-band | 1.96 | 2.01 |

Finest grid `fdtd-1088-48x`: last increment 4.064e-03, so its estimated distance to the limit is 4.036e-03 (second-order Richardson, r = 2.01).

| Configuration | Seconds | Details |
|---|---:|---|
| fdtd-176-3x | 1.6 | {"solver": "fdtd", "intervals": 176, "oversampling": 3, "rate": 132300, "spacing": 0.005681818181818182, "excitation": "raised_cosine", "stability_score": 0.21835411699361143} |
| fdtd-264-3x | 2.3 | {"solver": "fdtd", "intervals": 264, "oversampling": 3, "rate": 132300, "spacing": 0.003787878787878788, "excitation": "raised_cosine:0.24431818181818182", "stability_score": 0.9782361488482229, "russo_time_step": 7.64222351594367e-06} |
| fdtd-380-6x | 6.7 | {"solver": "fdtd", "intervals": 380, "oversampling": 6, "rate": 264600, "spacing": 0.002631578947368421, "excitation": "raised_cosine:0.24431818181818182", "stability_score": 0.9936083709075608, "russo_time_step": 3.79142934384752e-06} |
| fdtd-540-12x | 18.7 | {"solver": "fdtd", "intervals": 540, "oversampling": 12, "rate": 529200, "spacing": 0.001851851851851852, "excitation": "raised_cosine:0.24431818181818182", "stability_score": 0.9859981505198349, "russo_time_step": 1.9030187593502354e-06} |
| fdtd-768-24x | 53.1 | {"solver": "fdtd", "intervals": 768, "oversampling": 24, "rate": 1058400, "spacing": 0.0013020833333333333, "excitation": "raised_cosine:0.24431818181818182", "stability_score": 0.9948364915287865, "russo_time_step": 9.472719282656383e-07} |
| fdtd-1088-48x | 158.5 | {"solver": "fdtd", "intervals": 1088, "oversampling": 48, "rate": 2116800, "spacing": 0.0009191176470588235, "excitation": "raised_cosine:0.24431818181818182", "stability_score": 0.9949971184847305, "russo_time_step": 4.73597721242073e-07} |
| modal-100-44.1k-raw | 3.7 | {"filtered": false, "solver": "modal", "modes": 100, "rate": 44100, "excitation": "raised_cosine", "integrator": "jaxdiffmodal.time_integrators.solve_tf"} |
| modal-100-44.1k-band | 0.0 | {"filtered": true, "solver": "modal", "modes": 100, "rate": 44100, "excitation": "raised_cosine", "integrator": "jaxdiffmodal.time_integrators.solve_tf"} |
| modal-100-132.3k | 1.3 | {"filtered": true, "solver": "modal", "modes": 100, "rate": 132300, "excitation": "raised_cosine", "integrator": "jaxdiffmodal.time_integrators.solve_tf"} |
| modal-120-44.1k-raw | 1.6 | {"filtered": false, "solver": "modal", "modes": 120, "rate": 44100, "excitation": "raised_cosine", "integrator": "jaxdiffmodal.time_integrators.solve_tf"} |
| modal-120-44.1k-band | 0.0 | {"filtered": true, "solver": "modal", "modes": 120, "rate": 44100, "excitation": "raised_cosine", "integrator": "jaxdiffmodal.time_integrators.solve_tf"} |
| modal-120-132.3k | 1.4 | {"filtered": true, "solver": "modal", "modes": 120, "rate": 132300, "excitation": "raised_cosine", "integrator": "jaxdiffmodal.time_integrators.solve_tf"} |
