# Accuracy study: batch0 string, raised cosine, 1 s, outputs at 44.1 kHz, spatial order 4

Reference: `fdtd-942-48x` (h = 1.01 sqrt(gamma) h_min on Russo et al.'s path, gamma = 1.33333). f0 = 78.90 Hz, B = 3.007e-04; 120 continuous modes lie below 22.05 kHz.

| Candidate | Reference | Waveform rel. L2 | FFT-magnitude rel. L2 | Note |
|---|---|---:|---:|---|
| fdtd-176-2x | fdtd-942-48x | 1.147e-02 | 5.080e-03 | benchmark FDTD grid vs grid-converged FDTD |
| fdtd-228-3x | fdtd-942-48x | 5.274e-03 | 2.392e-03 | refinement path vs finest |
| fdtd-328-6x | fdtd-942-48x | 1.679e-03 | 6.475e-04 | refinement path vs finest |
| fdtd-468-12x | fdtd-942-48x | 6.841e-04 | 1.626e-04 | refinement path vs finest |
| fdtd-664-24x | fdtd-942-48x | 2.384e-04 | 3.488e-05 | refinement path vs finest |
| fdtd-228-3x | fdtd-328-6x | 4.010e-03 | 1.843e-03 | successive refinements |
| fdtd-328-6x | fdtd-468-12x | 1.097e-03 | 4.884e-04 | successive refinements |
| fdtd-468-12x | fdtd-664-24x | 4.539e-04 | 1.281e-04 | successive refinements |
| fdtd-664-24x | fdtd-942-48x | 2.384e-04 | 3.488e-05 | successive refinements |
| modal-100-44.1k-raw | fdtd-942-48x | 2.544e-04 | 1.874e-05 | modal vs grid-converged FDTD |
| modal-100-44.1k-band | fdtd-942-48x | 2.544e-04 | 1.892e-05 | modal vs grid-converged FDTD |
| modal-100-88.2k | fdtd-942-48x | 2.569e-04 | 1.821e-05 | modal vs grid-converged FDTD |
| modal-120-44.1k-raw | fdtd-942-48x | 2.555e-04 | 1.984e-05 | modal vs grid-converged FDTD |
| modal-120-44.1k-band | fdtd-942-48x | 2.553e-04 | 1.866e-05 | modal vs grid-converged FDTD |
| modal-120-88.2k | fdtd-942-48x | 2.570e-04 | 1.787e-05 | modal vs grid-converged FDTD |
| modal-100-44.1k-raw | fdtd-176-2x | 1.145e-02 | 5.093e-03 | modal vs benchmark FDTD grid (the sweep diagnostic uses the raw 44.1k signal) |
| modal-100-44.1k-band | fdtd-176-2x | 1.145e-02 | 5.093e-03 | modal vs benchmark FDTD grid (the sweep diagnostic uses the raw 44.1k signal) |
| modal-100-88.2k | fdtd-176-2x | 1.145e-02 | 5.092e-03 | modal vs benchmark FDTD grid (the sweep diagnostic uses the raw 44.1k signal) |
| modal-120-44.1k-raw | fdtd-176-2x | 1.145e-02 | 5.093e-03 | modal vs benchmark FDTD grid (the sweep diagnostic uses the raw 44.1k signal) |
| modal-120-44.1k-band | fdtd-176-2x | 1.145e-02 | 5.093e-03 | modal vs benchmark FDTD grid (the sweep diagnostic uses the raw 44.1k signal) |
| modal-120-88.2k | fdtd-176-2x | 1.145e-02 | 5.092e-03 | modal vs benchmark FDTD grid (the sweep diagnostic uses the raw 44.1k signal) |
| modal-100-44.1k-raw | modal-120-44.1k-raw | 1.186e-05 | 8.352e-06 | modal truncation |
| modal-100-44.1k-band | modal-100-88.2k | 3.884e-06 | 1.951e-06 | modal time stepping |
| modal-120-44.1k-band | modal-120-88.2k | 4.638e-06 | 2.952e-06 | modal time stepping |
| fdtd-176-2x | modal-120-44.1k-band | 1.145e-02 | 5.093e-03 | FDTD grid vs sub-Nyquist modal (band-matched) |
| fdtd-228-3x | modal-120-44.1k-band | 5.325e-03 | 2.405e-03 | FDTD grid vs sub-Nyquist modal (band-matched) |
| fdtd-328-6x | modal-120-44.1k-band | 1.860e-03 | 6.620e-04 | FDTD grid vs sub-Nyquist modal (band-matched) |
| fdtd-468-12x | modal-120-44.1k-band | 9.245e-04 | 1.784e-04 | FDTD grid vs sub-Nyquist modal (band-matched) |
| fdtd-664-24x | modal-120-44.1k-band | 4.915e-04 | 5.217e-05 | FDTD grid vs sub-Nyquist modal (band-matched) |
| fdtd-942-48x | modal-120-44.1k-band | 2.553e-04 | 1.866e-05 | FDTD grid vs sub-Nyquist modal (band-matched) |

Error ratios between refinement levels; order 4 in space predicts the (h_coarse/h_fine)^4 column for the spatial part (the second-order time step contributes (k_c/k_f)^2 = 4 per level):

| Coarse | Fine | Against | Error ratio | (h_c/h_f)^4 |
|---|---|---|---:|---:|
| fdtd-228-3x | fdtd-328-6x | next level | 3.65 | 4.28 |
| fdtd-328-6x | fdtd-468-12x | next level | 2.42 | 4.14 |
| fdtd-468-12x | fdtd-664-24x | next level | 1.90 | 4.05 |
| fdtd-176-2x | fdtd-228-3x | modal-120-44.1k-band | 2.15 | 2.82 |
| fdtd-228-3x | fdtd-328-6x | modal-120-44.1k-band | 2.86 | 4.28 |
| fdtd-328-6x | fdtd-468-12x | modal-120-44.1k-band | 2.01 | 4.14 |
| fdtd-468-12x | fdtd-664-24x | modal-120-44.1k-band | 1.88 | 4.05 |
| fdtd-664-24x | fdtd-942-48x | modal-120-44.1k-band | 1.92 | 4.05 |

Finest grid `fdtd-942-48x`: last increment 2.384e-04, so its estimated distance to the limit is 7.814e-05 (Richardson with r = 4.05).

| Configuration | Seconds | Details |
|---|---:|---|
| fdtd-176-2x | 2.3 | {"solver": "fdtd", "intervals": 176, "oversampling": 2, "rate": 88200, "order": 4, "spacing": 0.005681818181818182, "excitation": "raised_cosine", "stability_score": 0.8279341400648623} |
| fdtd-228-3x | 3.4 | {"solver": "fdtd", "intervals": 228, "oversampling": 3, "rate": 132300, "order": 4, "spacing": 0.0043859649122807015, "excitation": "raised_cosine:0.24431818181818182", "stability_score": 0.968043873961065, "russo_time_step": 1.007016819773381e-05} |
| fdtd-328-6x | 9.8 | {"solver": "fdtd", "intervals": 328, "oversampling": 6, "rate": 264600, "order": 4, "spacing": 0.003048780487804878, "excitation": "raised_cosine:0.24431818181818182", "stability_score": 0.9808619033574806, "russo_time_step": 5.043566614175584e-06} |
| fdtd-468-12x | 28.1 | {"solver": "fdtd", "intervals": 468, "oversampling": 12, "rate": 529200, "order": 4, "spacing": 0.002136752136752137, "excitation": "raised_cosine:0.24431818181818182", "stability_score": 0.9888816297203292, "russo_time_step": 2.5224147497216062e-06} |
| fdtd-664-24x | 80.1 | {"solver": "fdtd", "intervals": 664, "oversampling": 24, "rate": 1058400, "order": 4, "spacing": 0.0015060240963855422, "excitation": "raised_cosine:0.24431818181818182", "stability_score": 0.9882732086461802, "russo_time_step": 1.2643772440210227e-06} |
| fdtd-942-48x | 226.9 | {"solver": "fdtd", "intervals": 942, "oversampling": 48, "rate": 2116800, "order": 4, "spacing": 0.0010615711252653928, "excitation": "raised_cosine:0.24431818181818182", "stability_score": 0.994005822727976, "russo_time_step": 6.310689244909257e-07} |
| modal-100-44.1k-raw | 5.7 | {"filtered": false, "solver": "modal", "modes": 100, "rate": 44100, "excitation": "raised_cosine", "integrator": "jaxdiffmodal.time_integrators.solve_tf"} |
| modal-100-44.1k-band | 0.0 | {"filtered": true, "solver": "modal", "modes": 100, "rate": 44100, "excitation": "raised_cosine", "integrator": "jaxdiffmodal.time_integrators.solve_tf"} |
| modal-100-88.2k | 0.6 | {"filtered": true, "solver": "modal", "modes": 100, "rate": 88200, "excitation": "raised_cosine", "integrator": "jaxdiffmodal.time_integrators.solve_tf"} |
| modal-120-44.1k-raw | 1.5 | {"filtered": false, "solver": "modal", "modes": 120, "rate": 44100, "excitation": "raised_cosine", "integrator": "jaxdiffmodal.time_integrators.solve_tf"} |
| modal-120-44.1k-band | 0.0 | {"filtered": true, "solver": "modal", "modes": 120, "rate": 44100, "excitation": "raised_cosine", "integrator": "jaxdiffmodal.time_integrators.solve_tf"} |
| modal-120-88.2k | 0.6 | {"filtered": true, "solver": "modal", "modes": 120, "rate": 88200, "excitation": "raised_cosine", "integrator": "jaxdiffmodal.time_integrators.solve_tf"} |
