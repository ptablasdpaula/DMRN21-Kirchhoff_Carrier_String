# Accuracy study: nominal string, raised cosine, 1 s, outputs at 44.1 kHz, spatial order 2

Reference: `fdtd-1118-48x` (h = 1.01 sqrt(gamma) h_min on Russo et al.'s path, gamma = 1). f0 = 76.57 Hz, B = 2.876e-04; 123 continuous modes lie below 22.05 kHz.

| Candidate | Reference | Waveform rel. L2 | FFT-magnitude rel. L2 | Note |
|---|---|---:|---:|---|
| fdtd-176-2x | fdtd-1118-48x | 1.098e-01 | 3.796e-02 | benchmark FDTD grid vs grid-converged FDTD |
| fdtd-272-3x | fdtd-1118-48x | 4.803e-02 | 1.701e-02 | refinement path vs finest |
| fdtd-390-6x | fdtd-1118-48x | 2.389e-02 | 8.790e-03 | refinement path vs finest |
| fdtd-556-12x | fdtd-1118-48x | 1.071e-02 | 4.259e-03 | refinement path vs finest |
| fdtd-788-24x | fdtd-1118-48x | 3.717e-03 | 1.520e-03 | refinement path vs finest |
| fdtd-272-3x | fdtd-390-6x | 2.602e-02 | 8.925e-03 | successive refinements |
| fdtd-390-6x | fdtd-556-12x | 1.391e-02 | 5.128e-03 | successive refinements |
| fdtd-556-12x | fdtd-788-24x | 7.194e-03 | 2.962e-03 | successive refinements |
| fdtd-788-24x | fdtd-1118-48x | 3.717e-03 | 1.520e-03 | successive refinements |
| modal-100-44.1k-raw | fdtd-1118-48x | 3.715e-03 | 1.560e-03 | modal vs grid-converged FDTD |
| modal-100-44.1k-band | fdtd-1118-48x | 3.715e-03 | 1.560e-03 | modal vs grid-converged FDTD |
| modal-100-88.2k | fdtd-1118-48x | 3.717e-03 | 1.560e-03 | modal vs grid-converged FDTD |
| modal-123-44.1k-raw | fdtd-1118-48x | 3.716e-03 | 1.560e-03 | modal vs grid-converged FDTD |
| modal-123-44.1k-band | fdtd-1118-48x | 3.716e-03 | 1.560e-03 | modal vs grid-converged FDTD |
| modal-123-88.2k | fdtd-1118-48x | 3.717e-03 | 1.560e-03 | modal vs grid-converged FDTD |
| modal-100-44.1k-raw | fdtd-176-2x | 1.126e-01 | 3.908e-02 | modal vs benchmark FDTD grid (the sweep diagnostic uses the raw 44.1k signal) |
| modal-100-44.1k-band | fdtd-176-2x | 1.126e-01 | 3.908e-02 | modal vs benchmark FDTD grid (the sweep diagnostic uses the raw 44.1k signal) |
| modal-100-88.2k | fdtd-176-2x | 1.126e-01 | 3.908e-02 | modal vs benchmark FDTD grid (the sweep diagnostic uses the raw 44.1k signal) |
| modal-123-44.1k-raw | fdtd-176-2x | 1.126e-01 | 3.908e-02 | modal vs benchmark FDTD grid (the sweep diagnostic uses the raw 44.1k signal) |
| modal-123-44.1k-band | fdtd-176-2x | 1.126e-01 | 3.908e-02 | modal vs benchmark FDTD grid (the sweep diagnostic uses the raw 44.1k signal) |
| modal-123-88.2k | fdtd-176-2x | 1.126e-01 | 3.908e-02 | modal vs benchmark FDTD grid (the sweep diagnostic uses the raw 44.1k signal) |
| modal-100-44.1k-raw | modal-123-44.1k-raw | 1.428e-05 | 1.005e-05 | modal truncation |
| modal-100-44.1k-band | modal-100-88.2k | 2.940e-06 | 1.622e-06 | modal time stepping |
| modal-123-44.1k-band | modal-123-88.2k | 3.641e-06 | 2.565e-06 | modal time stepping |
| fdtd-176-2x | modal-123-44.1k-band | 1.126e-01 | 3.908e-02 | FDTD grid vs sub-Nyquist modal (band-matched) |
| fdtd-272-3x | modal-123-44.1k-band | 5.116e-02 | 1.821e-02 | FDTD grid vs sub-Nyquist modal (band-matched) |
| fdtd-390-6x | modal-123-44.1k-band | 2.719e-02 | 1.009e-02 | FDTD grid vs sub-Nyquist modal (band-matched) |
| fdtd-556-12x | modal-123-44.1k-band | 1.416e-02 | 5.549e-03 | FDTD grid vs sub-Nyquist modal (band-matched) |
| fdtd-788-24x | modal-123-44.1k-band | 7.311e-03 | 2.949e-03 | FDTD grid vs sub-Nyquist modal (band-matched) |
| fdtd-1118-48x | modal-123-44.1k-band | 3.716e-03 | 1.560e-03 | FDTD grid vs sub-Nyquist modal (band-matched) |

Error ratios between refinement levels; order 2 in space predicts the (h_coarse/h_fine)^2 column for the spatial part (the second-order time step contributes (k_c/k_f)^2 = 4 per level):

| Coarse | Fine | Against | Error ratio | (h_c/h_f)^2 |
|---|---|---|---:|---:|
| fdtd-272-3x | fdtd-390-6x | next level | 1.87 | 2.06 |
| fdtd-390-6x | fdtd-556-12x | next level | 1.93 | 2.03 |
| fdtd-556-12x | fdtd-788-24x | next level | 1.94 | 2.01 |
| fdtd-176-2x | fdtd-272-3x | modal-123-44.1k-band | 2.20 | 2.39 |
| fdtd-272-3x | fdtd-390-6x | modal-123-44.1k-band | 1.88 | 2.06 |
| fdtd-390-6x | fdtd-556-12x | modal-123-44.1k-band | 1.92 | 2.03 |
| fdtd-556-12x | fdtd-788-24x | modal-123-44.1k-band | 1.94 | 2.01 |
| fdtd-788-24x | fdtd-1118-48x | modal-123-44.1k-band | 1.97 | 2.01 |

Finest grid `fdtd-1118-48x`: last increment 3.717e-03, so its estimated distance to the limit is 3.670e-03 (Richardson with r = 2.01).

| Configuration | Seconds | Details |
|---|---:|---|
| fdtd-176-2x | 1.9 | {"solver": "fdtd", "intervals": 176, "oversampling": 2, "rate": 88200, "order": 2, "spacing": 0.005681818181818182, "excitation": "raised_cosine", "stability_score": 0.4464862743105605} |
| fdtd-272-3x | 3.1 | {"solver": "fdtd", "intervals": 272, "oversampling": 3, "rate": 132300, "order": 2, "spacing": 0.003676470588235294, "excitation": "raised_cosine:0.24431818181818182", "stability_score": 0.9912205339828284, "russo_time_step": 7.591990233360138e-06} |
| fdtd-390-6x | 8.9 | {"solver": "fdtd", "intervals": 390, "oversampling": 6, "rate": 264600, "order": 2, "spacing": 0.002564102564102564, "excitation": "raised_cosine:0.24431818181818182", "stability_score": 0.9924578044822815, "russo_time_step": 3.793627545066959e-06} |
| fdtd-556-12x | 25.4 | {"solver": "fdtd", "intervals": 556, "oversampling": 12, "rate": 529200, "order": 2, "spacing": 0.0017985611510791368, "excitation": "raised_cosine:0.24431818181818182", "stability_score": 0.9976500267004421, "russo_time_step": 1.891869736292155e-06} |
| fdtd-788-24x | 71.8 | {"solver": "fdtd", "intervals": 788, "oversampling": 24, "rate": 1058400, "order": 2, "spacing": 0.0012690355329949238, "excitation": "raised_cosine:0.24431818181818182", "stability_score": 0.992912862996466, "russo_time_step": 9.481894804581555e-07} |
| fdtd-1118-48x | 205.9 | {"solver": "fdtd", "intervals": 1118, "oversampling": 48, "rate": 2116800, "order": 2, "spacing": 0.0008944543828264759, "excitation": "raised_cosine:0.24431818181818182", "stability_score": 0.9990398951324009, "russo_time_step": 4.7263821067068984e-07} |
| modal-100-44.1k-raw | 4.2 | {"filtered": false, "solver": "modal", "modes": 100, "rate": 44100, "excitation": "raised_cosine", "integrator": "jaxdiffmodal.time_integrators.solve_tf"} |
| modal-100-44.1k-band | 0.0 | {"filtered": true, "solver": "modal", "modes": 100, "rate": 44100, "excitation": "raised_cosine", "integrator": "jaxdiffmodal.time_integrators.solve_tf"} |
| modal-100-88.2k | 0.6 | {"filtered": true, "solver": "modal", "modes": 100, "rate": 88200, "excitation": "raised_cosine", "integrator": "jaxdiffmodal.time_integrators.solve_tf"} |
| modal-123-44.1k-raw | 1.6 | {"filtered": false, "solver": "modal", "modes": 123, "rate": 44100, "excitation": "raised_cosine", "integrator": "jaxdiffmodal.time_integrators.solve_tf"} |
| modal-123-44.1k-band | 0.0 | {"filtered": true, "solver": "modal", "modes": 123, "rate": 44100, "excitation": "raised_cosine", "integrator": "jaxdiffmodal.time_integrators.solve_tf"} |
| modal-123-88.2k | 0.7 | {"filtered": true, "solver": "modal", "modes": 123, "rate": 88200, "excitation": "raised_cosine", "integrator": "jaxdiffmodal.time_integrators.solve_tf"} |
