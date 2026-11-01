# Accuracy study: nominal string, raised cosine, 1 s, outputs at 44.1 kHz, spatial order 4

Reference: `fdtd-968-48x` (h = 1.01 sqrt(gamma) h_min on Russo et al.'s path, gamma = 1.33333). f0 = 76.57 Hz, B = 2.876e-04; 123 continuous modes lie below 22.05 kHz.

| Candidate | Reference | Waveform rel. L2 | FFT-magnitude rel. L2 | Note |
|---|---|---:|---:|---|
| fdtd-176-2x | fdtd-968-48x | 1.033e-02 | 5.063e-03 | benchmark FDTD grid vs grid-converged FDTD |
| fdtd-236-3x | fdtd-968-48x | 4.812e-03 | 2.366e-03 | refinement path vs finest |
| fdtd-338-6x | fdtd-968-48x | 1.594e-03 | 6.413e-04 | refinement path vs finest |
| fdtd-480-12x | fdtd-968-48x | 6.653e-04 | 1.594e-04 | refinement path vs finest |
| fdtd-682-24x | fdtd-968-48x | 2.326e-04 | 3.427e-05 | refinement path vs finest |
| fdtd-236-3x | fdtd-338-6x | 3.617e-03 | 1.818e-03 | successive refinements |
| fdtd-338-6x | fdtd-480-12x | 1.020e-03 | 4.846e-04 | successive refinements |
| fdtd-480-12x | fdtd-682-24x | 4.395e-04 | 1.256e-04 | successive refinements |
| fdtd-682-24x | fdtd-968-48x | 2.326e-04 | 3.427e-05 | successive refinements |
| modal-100-44.1k-raw | fdtd-968-48x | 2.481e-04 | 1.885e-05 | modal vs grid-converged FDTD |
| modal-100-44.1k-band | fdtd-968-48x | 2.482e-04 | 1.898e-05 | modal vs grid-converged FDTD |
| modal-100-88.2k | fdtd-968-48x | 2.500e-04 | 1.837e-05 | modal vs grid-converged FDTD |
| modal-123-44.1k-raw | fdtd-968-48x | 2.498e-04 | 2.062e-05 | modal vs grid-converged FDTD |
| modal-123-44.1k-band | fdtd-968-48x | 2.495e-04 | 1.863e-05 | modal vs grid-converged FDTD |
| modal-123-88.2k | fdtd-968-48x | 2.501e-04 | 1.799e-05 | modal vs grid-converged FDTD |
| modal-100-44.1k-raw | fdtd-176-2x | 1.032e-02 | 5.074e-03 | modal vs benchmark FDTD grid (the sweep diagnostic uses the raw 44.1k signal) |
| modal-100-44.1k-band | fdtd-176-2x | 1.032e-02 | 5.074e-03 | modal vs benchmark FDTD grid (the sweep diagnostic uses the raw 44.1k signal) |
| modal-100-88.2k | fdtd-176-2x | 1.032e-02 | 5.074e-03 | modal vs benchmark FDTD grid (the sweep diagnostic uses the raw 44.1k signal) |
| modal-123-44.1k-raw | fdtd-176-2x | 1.032e-02 | 5.074e-03 | modal vs benchmark FDTD grid (the sweep diagnostic uses the raw 44.1k signal) |
| modal-123-44.1k-band | fdtd-176-2x | 1.032e-02 | 5.074e-03 | modal vs benchmark FDTD grid (the sweep diagnostic uses the raw 44.1k signal) |
| modal-123-88.2k | fdtd-176-2x | 1.032e-02 | 5.074e-03 | modal vs benchmark FDTD grid (the sweep diagnostic uses the raw 44.1k signal) |
| modal-100-44.1k-raw | modal-123-44.1k-raw | 1.428e-05 | 1.005e-05 | modal truncation |
| modal-100-44.1k-band | modal-100-88.2k | 2.940e-06 | 1.622e-06 | modal time stepping |
| modal-123-44.1k-band | modal-123-88.2k | 3.641e-06 | 2.565e-06 | modal time stepping |
| fdtd-176-2x | modal-123-44.1k-band | 1.032e-02 | 5.075e-03 | FDTD grid vs sub-Nyquist modal (band-matched) |
| fdtd-236-3x | modal-123-44.1k-band | 4.874e-03 | 2.378e-03 | FDTD grid vs sub-Nyquist modal (band-matched) |
| fdtd-338-6x | modal-123-44.1k-band | 1.781e-03 | 6.554e-04 | FDTD grid vs sub-Nyquist modal (band-matched) |
| fdtd-480-12x | modal-123-44.1k-band | 9.025e-04 | 1.750e-04 | FDTD grid vs sub-Nyquist modal (band-matched) |
| fdtd-682-24x | modal-123-44.1k-band | 4.803e-04 | 5.150e-05 | FDTD grid vs sub-Nyquist modal (band-matched) |
| fdtd-968-48x | modal-123-44.1k-band | 2.495e-04 | 1.863e-05 | FDTD grid vs sub-Nyquist modal (band-matched) |

Error ratios between refinement levels; order 4 in space predicts the (h_coarse/h_fine)^4 column for the spatial part (the second-order time step contributes (k_c/k_f)^2 = 4 per level):

| Coarse | Fine | Against | Error ratio | (h_c/h_f)^4 |
|---|---|---|---:|---:|
| fdtd-236-3x | fdtd-338-6x | next level | 3.55 | 4.21 |
| fdtd-338-6x | fdtd-480-12x | next level | 2.32 | 4.07 |
| fdtd-480-12x | fdtd-682-24x | next level | 1.89 | 4.08 |
| fdtd-176-2x | fdtd-236-3x | modal-123-44.1k-band | 2.12 | 3.23 |
| fdtd-236-3x | fdtd-338-6x | modal-123-44.1k-band | 2.74 | 4.21 |
| fdtd-338-6x | fdtd-480-12x | modal-123-44.1k-band | 1.97 | 4.07 |
| fdtd-480-12x | fdtd-682-24x | modal-123-44.1k-band | 1.88 | 4.08 |
| fdtd-682-24x | fdtd-968-48x | modal-123-44.1k-band | 1.92 | 4.06 |

Finest grid `fdtd-968-48x`: last increment 2.326e-04, so its estimated distance to the limit is 7.606e-05 (Richardson with r = 4.06).

| Configuration | Seconds | Details |
|---|---:|---|
| fdtd-176-2x | 2.3 | {"solver": "fdtd", "intervals": 176, "oversampling": 2, "rate": 88200, "order": 4, "spacing": 0.005681818181818182, "excitation": "raised_cosine", "stability_score": 0.7512319895466758} |
| fdtd-236-3x | 3.5 | {"solver": "fdtd", "intervals": 236, "oversampling": 3, "rate": 132300, "order": 4, "spacing": 0.00423728813559322, "excitation": "raised_cosine:0.24431818181818182", "stability_score": 0.9982828102988738, "russo_time_step": 9.920012393381999e-06} |
| fdtd-338-6x | 10.2 | {"solver": "fdtd", "intervals": 338, "oversampling": 6, "rate": 264600, "order": 4, "spacing": 0.0029585798816568047, "excitation": "raised_cosine:0.24431818181818182", "stability_score": 0.9953224895198436, "russo_time_step": 5.0074131302310305e-06} |
| fdtd-480-12x | 28.6 | {"solver": "fdtd", "intervals": 480, "oversampling": 12, "rate": 529200, "order": 4, "spacing": 0.0020833333333333333, "excitation": "raised_cosine:0.24431818181818182", "stability_score": 0.9853619564972164, "russo_time_step": 2.5269822245232483e-06} |
| fdtd-682-24x | 82.7 | {"solver": "fdtd", "intervals": 682, "oversampling": 24, "rate": 1058400, "order": 4, "spacing": 0.001466275659824047, "excitation": "raised_cosine:0.24431818181818182", "stability_score": 0.9904416572684712, "russo_time_step": 1.2630165925367932e-06} |
| fdtd-968-48x | 238.5 | {"solver": "fdtd", "intervals": 968, "oversampling": 48, "rate": 2116800, "order": 4, "spacing": 0.0010330578512396695, "excitation": "raised_cosine:0.24431818181818182", "stability_score": 0.9981503217264711, "russo_time_step": 6.297638330271302e-07} |
| modal-100-44.1k-raw | 10.9 | {"filtered": false, "solver": "modal", "modes": 100, "rate": 44100, "excitation": "raised_cosine", "integrator": "jaxdiffmodal.time_integrators.solve_tf"} |
| modal-100-44.1k-band | 0.0 | {"filtered": true, "solver": "modal", "modes": 100, "rate": 44100, "excitation": "raised_cosine", "integrator": "jaxdiffmodal.time_integrators.solve_tf"} |
| modal-100-88.2k | 0.6 | {"filtered": true, "solver": "modal", "modes": 100, "rate": 88200, "excitation": "raised_cosine", "integrator": "jaxdiffmodal.time_integrators.solve_tf"} |
| modal-123-44.1k-raw | 1.6 | {"filtered": false, "solver": "modal", "modes": 123, "rate": 44100, "excitation": "raised_cosine", "integrator": "jaxdiffmodal.time_integrators.solve_tf"} |
| modal-123-44.1k-band | 0.0 | {"filtered": true, "solver": "modal", "modes": 123, "rate": 44100, "excitation": "raised_cosine", "integrator": "jaxdiffmodal.time_integrators.solve_tf"} |
| modal-123-88.2k | 0.6 | {"filtered": true, "solver": "modal", "modes": 123, "rate": 88200, "excitation": "raised_cosine", "integrator": "jaxdiffmodal.time_integrators.solve_tf"} |
