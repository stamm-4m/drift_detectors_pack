# Supplementary results: per-batch, per-phase drift tables

These tables expand the in-text Tables 1 and 2 of the AI4D 2026 paper
(*A drift detection toolkit for soft-sensor monitoring in industrial
processes: a penicillin fermentation use case*) to all ten test batches
(91-100) and all four canonical fermentation phases. Table 1 in the paper
itself reports only the three representative fault batches
(substrate-feed 91 and 95; aeration 94); the remaining seven batches
--- the two in-control references (92, 93), the near-in-control
batches (96, 98), and the other fault batches (97, 99, 100) --- are
tabulated below.

All values are computed against the LSTM training set (in-control batches
1-60) of Metcalfe *et al.* (2025).

Source JSON for the same numbers:

- `expA_per_phase.json` --- PSI / KS per fermentation phase (Tables S0, S1).
- `expA_multivar_per_phase.json` --- MMD / PCA-CD / KDQ-Tree per phase (Table S3).
- `expB_per_phase.json` --- soft-sensor CV / MAE / RMSE and MDM per phase (Table S2 and Table 2 in the paper).

---

## Table S0. Per-variable PSI per fermentation phase, all ten test batches

| Phase | Batch | PSI Fs | PSI Aer | PSI DO2 | PSI CO2 |
|---|---|---|---|---|---|
| lag | 91 | 3.443 | 0.000 | 1.587 | 3.788 |
| lag | 92 | 0.000 | 0.000 | 1.063 | 0.278 |
| lag | 93 | 0.000 | 0.000 | 1.149 | 0.296 |
| lag | 94 | 0.000 | 2.391 | 1.775 | 2.164 |
| lag | 95 | 3.443 | 0.000 | 2.511 | 3.794 |
| lag | 96 | 0.000 | 0.000 | 0.292 | 0.049 |
| lag | 97 | 3.443 | 0.000 | 2.601 | 3.789 |
| lag | 98 | 0.000 | 0.000 | 0.345 | 0.308 |
| lag | 99 | 0.000 | 2.391 | 0.984 | 2.167 |
| lag | 100 | 0.000 | 2.391 | 1.783 | 2.295 |
| log | 91 | 3.755 | 0.000 | 0.662 | 6.622 |
| log | 92 | 0.000 | 0.000 | 2.335 | 0.473 |
| log | 93 | 0.000 | 0.000 | 2.743 | 0.433 |
| log | 94 | 0.000 | 1.888 | 2.983 | 1.917 |
| log | 95 | 3.755 | 0.000 | 3.660 | 6.530 |
| log | 96 | 0.000 | 0.000 | 0.470 | 0.421 |
| log | 97 | 3.755 | 0.000 | 2.481 | 6.426 |
| log | 98 | 0.000 | 0.000 | 2.349 | 0.549 |
| log | 99 | 0.000 | 1.888 | 1.833 | 1.844 |
| log | 100 | 0.000 | 1.888 | 0.187 | 2.080 |
| stationary | 91 | 3.741 | 0.000 | 14.858 | 24.209 |
| stationary | 92 | 3.741 | 0.000 | 1.475 | 2.166 |
| stationary | 93 | 3.741 | 0.000 | 4.195 | 2.386 |
| stationary | 94 | 3.741 | 0.000 | 16.241 | 14.881 |
| stationary | 95 | 3.741 | 0.000 | 18.440 | 24.204 |
| stationary | 96 | 3.741 | 0.000 | 5.256 | 2.379 |
| stationary | 97 | 3.741 | 0.000 | 5.571 | 2.960 |
| stationary | 98 | 3.741 | 0.000 | 2.371 | 2.449 |
| stationary | 99 | 3.741 | 0.000 | 8.968 | 11.020 |
| stationary | 100 | 3.741 | 0.000 | 4.194 | 2.741 |
| death | 91 | 6.528 | 0.062 | 12.248 | 28.028 |
| death | 92 | 8.055 | 1.014 | 11.632 | 9.390 |
| death | 93 | 8.055 | 1.029 | 11.262 | 10.709 |
| death | 94 | 8.055 | 1.014 | 15.155 | 16.345 |
| death | 95 | 20.795 | 1.026 | 16.383 | 19.867 |
| death | 96 | 8.055 | 1.014 | 8.289 | 7.462 |
| death | 97 | 7.868 | 1.015 | 10.989 | 7.484 |
| death | 98 | 8.055 | 1.014 | 9.498 | 7.818 |
| death | 99 | 8.055 | 0.057 | 8.265 | 15.623 |
| death | 100 | 8.055 | 1.014 | 10.727 | 11.898 |

## Table S1. Kolmogorov-Smirnov statistic per fermentation phase, batches 91-100 vs reference 1-60

| Phase | Batch | KS Fs | KS Aer | KS DO2 | KS CO2 |
|---|---|---|---|---|---|
| lag | 91 | 0.336 | 0.000 | 0.288 | 0.261 |
| lag | 92 | 0.000 | 0.000 | 0.118 | 0.052 |
| lag | 93 | 0.000 | 0.000 | 0.223 | 0.092 |
| lag | 94 | 0.000 | 0.141 | 0.128 | 0.150 |
| lag | 95 | 0.336 | 0.000 | 0.352 | 0.240 |
| lag | 96 | 0.000 | 0.000 | 0.143 | 0.060 |
| lag | 97 | 0.336 | 0.000 | 0.320 | 0.263 |
| lag | 98 | 0.000 | 0.000 | 0.094 | 0.113 |
| lag | 99 | 0.000 | 0.141 | 0.156 | 0.169 |
| lag | 100 | 0.000 | 0.141 | 0.129 | 0.234 |
| log | 91 | 0.182 | 0.000 | 0.379 | 0.398 |
| log | 92 | 0.000 | 0.000 | 0.307 | 0.182 |
| log | 93 | 0.000 | 0.000 | 0.371 | 0.104 |
| log | 94 | 0.000 | 0.113 | 0.426 | 0.182 |
| log | 95 | 0.182 | 0.000 | 0.531 | 0.506 |
| log | 96 | 0.000 | 0.000 | 0.156 | 0.138 |
| log | 97 | 0.182 | 0.000 | 0.197 | 0.383 |
| log | 98 | 0.000 | 0.000 | 0.208 | 0.110 |
| log | 99 | 0.000 | 0.113 | 0.276 | 0.113 |
| log | 100 | 0.000 | 0.113 | 0.198 | 0.368 |
| stationary | 91 | 0.207 | 0.000 | 0.842 | 0.968 |
| stationary | 92 | 0.207 | 0.000 | 0.426 | 0.210 |
| stationary | 93 | 0.207 | 0.000 | 0.549 | 0.162 |
| stationary | 94 | 0.207 | 0.000 | 0.910 | 0.818 |
| stationary | 95 | 0.207 | 0.000 | 0.947 | 0.965 |
| stationary | 96 | 0.207 | 0.000 | 0.352 | 0.171 |
| stationary | 97 | 0.207 | 0.000 | 0.357 | 0.227 |
| stationary | 98 | 0.207 | 0.000 | 0.296 | 0.210 |
| stationary | 99 | 0.207 | 0.000 | 0.608 | 0.727 |
| stationary | 100 | 0.207 | 0.000 | 0.537 | 0.418 |
| death | 91 | 0.274 | 0.073 | 0.787 | 0.986 |
| death | 92 | 0.274 | 0.064 | 0.701 | 0.470 |
| death | 93 | 0.274 | 0.064 | 0.625 | 0.786 |
| death | 94 | 0.274 | 0.064 | 0.945 | 0.899 |
| death | 95 | 1.000 | 0.064 | 0.942 | 0.986 |
| death | 96 | 0.274 | 0.064 | 0.416 | 0.331 |
| death | 97 | 0.563 | 0.064 | 0.375 | 0.356 |
| death | 98 | 0.274 | 0.064 | 0.416 | 0.480 |
| death | 99 | 0.274 | 0.045 | 0.600 | 0.885 |
| death | 100 | 0.274 | 0.064 | 0.485 | 0.679 |

## Table S2. MAE (g/L) per soft sensor, per fermentation phase

| Phase | Batch | MAE CART | MAE M5 | MAE CUBIST | MAE RF |
|---|---|---|---|---|---|
| lag | 91 | 0.132 | 0.091 | 1.304 | 1.854 |
| lag | 92 | 0.016 | 0.053 | 3.052 | 1.426 |
| lag | 93 | 0.033 | 0.061 | 2.654 | 1.480 |
| lag | 94 | 0.588 | 1.502 | 3.127 | 2.315 |
| lag | 95 | 0.182 | 0.135 | 1.348 | 1.451 |
| lag | 96 | 0.024 | 0.044 | 3.429 | 1.736 |
| lag | 97 | 0.156 | 0.114 | 1.335 | 1.336 |
| lag | 98 | 0.011 | 0.042 | 3.224 | 1.548 |
| lag | 99 | 0.589 | 1.452 | 3.398 | 2.513 |
| lag | 100 | 0.627 | 1.403 | 3.370 | 2.538 |
| log | 91 | 3.925 | 5.830 | 6.104 | 5.758 |
| log | 92 | 0.523 | 1.110 | 1.496 | 1.520 |
| log | 93 | 0.871 | 1.329 | 1.891 | 1.466 |
| log | 94 | 1.556 | 2.511 | 2.526 | 2.456 |
| log | 95 | 2.950 | 2.923 | 3.479 | 4.135 |
| log | 96 | 1.339 | 1.753 | 2.502 | 2.272 |
| log | 97 | 2.721 | 3.224 | 4.374 | 4.639 |
| log | 98 | 0.785 | 1.480 | 1.606 | 1.442 |
| log | 99 | 0.915 | 3.693 | 3.433 | 2.808 |
| log | 100 | 2.587 | 4.808 | 4.660 | 3.961 |
| stationary | 91 | 6.330 | 7.283 | 5.590 | 5.994 |
| stationary | 92 | 1.991 | 3.578 | 2.417 | 2.103 |
| stationary | 93 | 4.080 | 4.871 | 6.814 | 5.523 |
| stationary | 94 | 6.797 | 7.072 | 4.939 | 5.093 |
| stationary | 95 | 4.371 | 4.904 | 2.832 | 3.500 |
| stationary | 96 | 2.170 | 2.151 | 3.138 | 2.661 |
| stationary | 97 | 1.923 | 1.890 | 1.326 | 1.388 |
| stationary | 98 | 3.294 | 4.519 | 5.039 | 4.201 |
| stationary | 99 | 4.804 | 6.990 | 5.133 | 6.055 |
| stationary | 100 | 13.341 | 12.184 | 10.680 | 11.943 |
| death | 91 | 5.552 | 7.779 | 3.533 | 2.801 |
| death | 92 | 1.899 | 1.336 | 1.578 | 0.582 |
| death | 93 | 7.340 | 10.243 | 12.151 | 10.114 |
| death | 94 | 5.712 | 4.761 | 3.703 | 5.062 |
| death | 95 | 4.101 | 16.718 | 7.382 | 0.977 |
| death | 96 | 3.348 | 5.418 | 6.690 | 5.566 |
| death | 97 | 4.972 | 4.164 | 0.981 | 2.132 |
| death | 98 | 8.055 | 8.053 | 12.620 | 9.592 |
| death | 99 | 3.894 | 3.465 | 1.741 | 2.716 |
| death | 100 | 9.547 | 7.527 | 9.462 | 10.081 |

## Table S3. Multivariate drift detectors per fermentation phase

MMD uses the median-distance bandwidth heuristic. PCA-CD reports a
KL-divergence-style score and is sensitive to scale (see Section 6 of
the paper); a clean separation between fault and in-control batches
requires per-variable standardisation that the current evaluation does
not apply. KDQ-Tree is reported on the same partition.

| Phase | Batch | MMD (median) | PCA-CD | KDQ-Tree |
|---|---|---|---|---|
| lag | 91 | 0.164 | 8.593e+04 | 1.000 |
| lag | 92 | 0.089 | 3.352e+06 | 0.987 |
| lag | 93 | 0.030 | 6.515e+05 | 0.999 |
| lag | 94 | 0.025 | 5.095e+05 | 0.999 |
| lag | 95 | 0.093 | 1.777e+06 | 0.989 |
| lag | 96 | 0.030 | 6.265e+05 | 0.997 |
| lag | 97 | 0.096 | 7.563e+05 | 1.000 |
| lag | 98 | 0.016 | 1.390e+05 | 0.990 |
| lag | 99 | 0.408 | 2.272e+07 | 1.000 |
| lag | 100 | 0.023 | 2.998e+05 | 0.993 |
| log | 91 | 0.039 | 7.370e+07 | 1.000 |
| log | 92 | 0.004 | 1.349e+07 | 0.968 |
| log | 93 | 0.009 | 4.254e+06 | 1.000 |
| log | 94 | 0.001 | 4.197e+06 | 1.000 |
| log | 95 | 0.049 | 8.459e+07 | 1.000 |
| log | 96 | 0.023 | 3.923e+07 | 1.000 |
| log | 97 | 0.001 | 6.477e+05 | 1.000 |
| log | 98 | 0.007 | 1.329e+07 | 0.984 |
| log | 99 | 0.050 | 1.054e+08 | 1.000 |
| log | 100 | 0.013 | 1.299e+07 | 1.000 |
| stationary | 91 | 0.460 | 4.342e+08 | 1.000 |
| stationary | 92 | 0.018 | 6.538e+06 | 1.000 |
| stationary | 93 | 0.162 | 9.984e+07 | 1.000 |
| stationary | 94 | 0.098 | 6.633e+07 | 1.000 |
| stationary | 95 | 0.498 | 4.870e+08 | 1.000 |
| stationary | 96 | 0.200 | 1.234e+08 | 0.999 |
| stationary | 97 | 0.024 | 1.064e+07 | 0.994 |
| stationary | 98 | 0.067 | 3.097e+07 | 0.992 |
| stationary | 99 | 0.474 | 4.583e+08 | 1.000 |
| stationary | 100 | 0.012 | 2.479e+06 | 1.000 |
| death | 91 | 0.864 | 9.341e+08 | 1.000 |
| death | 92 | 0.139 | 1.801e+07 | 1.000 |
| death | 93 | 0.847 | 5.924e+08 | 1.000 |
| death | 94 | 0.295 | 1.539e+08 | 1.000 |
| death | 95 | 0.802 | 5.271e+08 | 1.000 |
| death | 96 | 0.228 | 1.018e+08 | 1.000 |
| death | 97 | 0.158 | 6.770e+07 | 1.000 |
| death | 98 | 0.097 | 4.388e+06 | 1.000 |
| death | 99 | 0.840 | 7.586e+08 | 1.000 |
| death | 100 | 0.136 | 3.832e+07 | 1.000 |

---

## Per-soft-sensor coefficient of variation (CV) per fermentation phase

Companion table to Table 2 of the paper. The Coefficient of Variation is
defined here as `CV = RMSE(y, y_hat) / mean(y)`, a unit-free standard
error measure expressed as a fraction of the mean target. It requires the
ground-truth penicillin trace and is therefore a research-time diagnostic,
not a deployment signal. Reported here to let the reader cross-check that
the MDM ranking in Table 2 of the paper matches the actual prediction
quality from the ground-truth side.

| Phase | Batch | CV CART | CV M5 | CV CUBIST | CV RF |
|---|---|---|---|---|---|
| lag | 91 | 2.027 | 1.208 | 17.150 | 14.726 |
| lag | 92 | 0.674 | 1.356 | 80.356 | 25.228 |
| lag | 93 | 1.145 | 1.461 | 57.214 | 20.956 |
| lag | 94 | 23.082 | 59.123 | 77.110 | 48.324 |
| lag | 95 | 2.008 | 1.373 | 13.420 | 8.802 |
| lag | 96 | 0.780 | 0.970 | 69.978 | 23.969 |
| lag | 97 | 2.022 | 1.336 | 15.524 | 9.508 |
| lag | 98 | 0.450 | 0.985 | 75.244 | 24.379 |
| lag | 99 | 20.530 | 50.038 | 73.150 | 44.339 |
| lag | 100 | 20.367 | 48.575 | 70.089 | 43.816 |
| log | 91 | 0.981 | 1.433 | 1.267 | 1.223 |
| log | 92 | 0.081 | 0.212 | 0.216 | 0.209 |
| log | 93 | 0.141 | 0.174 | 0.225 | 0.182 |
| log | 94 | 0.420 | 0.585 | 0.424 | 0.372 |
| log | 95 | 0.616 | 0.697 | 0.556 | 0.707 |
| log | 96 | 0.274 | 0.325 | 0.305 | 0.294 |
| log | 97 | 0.467 | 0.619 | 0.697 | 0.759 |
| log | 98 | 0.109 | 0.227 | 0.198 | 0.173 |
| log | 99 | 0.177 | 0.832 | 0.484 | 0.386 |
| log | 100 | 0.492 | 1.144 | 0.669 | 0.554 |
| stationary | 91 | 1.071 | 1.274 | 0.970 | 1.056 |
| stationary | 92 | 0.114 | 0.229 | 0.135 | 0.119 |
| stationary | 93 | 0.185 | 0.200 | 0.277 | 0.222 |
| stationary | 94 | 0.745 | 0.788 | 0.539 | 0.547 |
| stationary | 95 | 0.561 | 0.652 | 0.403 | 0.475 |
| stationary | 96 | 0.120 | 0.131 | 0.172 | 0.146 |
| stationary | 97 | 0.115 | 0.115 | 0.075 | 0.085 |
| stationary | 98 | 0.152 | 0.204 | 0.245 | 0.182 |
| stationary | 99 | 0.564 | 0.789 | 0.559 | 0.635 |
| stationary | 100 | 1.561 | 1.442 | 1.233 | 1.377 |
| death | 91 | 1.492 | 2.488 | 1.234 | 0.815 |
| death | 92 | 0.139 | 0.083 | 0.095 | 0.041 |
| death | 93 | 0.263 | 0.341 | 0.398 | 0.333 |
| death | 94 | 0.858 | 0.784 | 0.599 | 0.772 |
| death | 95 | 0.652 | 2.661 | 1.176 | 0.214 |
| death | 96 | 0.184 | 0.331 | 0.313 | 0.256 |
| death | 97 | 0.303 | 0.269 | 0.075 | 0.141 |
| death | 98 | 0.258 | 0.250 | 0.390 | 0.298 |
| death | 99 | 0.599 | 0.600 | 0.295 | 0.467 |
| death | 100 | 1.554 | 1.264 | 1.526 | 1.659 |

In the log phase (the operationally relevant window per the paper),
the four soft sensors predict the near-in-control batches (92, 93, 96, 98)
with CV in the range 0.08-0.23. On batches with documented faults the same
learners disagree by a factor of two to ten in CV, with the worst-fitting
sensor (most often M5) more than tripling its in-control CV on aeration
faults 99 and 100.

Source JSON: `expB_per_phase.json`, fields `perf.{CART,M5,CUBIST,RF}.CV`.
