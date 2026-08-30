# -*- coding: utf-8 -*-
"""诊断3（快速）：SiC 单角度，sigma_d=0，网格初值 + L-M，比较双光束 vs Airy。"""
import sys
import numpy as np
sys.path.insert(0, ".")

from preprocess import load_spectra
from q3 import preprocess_sic, detrend_poly, fft_init_d
from models import two_beam_reflectance_avg, airy_reflectance_avg, n_cauchy
from scipy.optimize import least_squares

wn1, R1 = load_spectra("data/raw/附件1.xlsx")
w1, r1, c1 = preprocess_sic(wn1, R1)
d0 = fft_init_d(w1, detrend_poly(w1, r1), n_avg=2.6, theta0_deg=10, search_lo=1500)
print("cutoff", c1, "d0", round(d0, 4), "N", len(w1))
N0 = 1.0


def lin(T, R):
    A = np.column_stack([T, np.ones_like(T)])
    s, *_ = np.linalg.lstsq(A, R, rcond=None)
    return s


def resid_two(th):
    d, A, B, C, n2 = th
    T = two_beam_reflectance_avg(w1, d, 0.0, n_cauchy(w1, A, B, C), n2, 10, N0)
    a, b = lin(T, r1)
    return a * T + b - r1


def resid_airy(th):
    d, A, B, C, n2r, k2 = th
    T = airy_reflectance_avg(w1, d, 0.0, n_cauchy(w1, A, B, C), n2r + 1j * k2, 10, N0)
    a, b = lin(T, r1)
    return a * T + b - r1


def aic(res, p):
    s = float(np.sum(res.fun ** 2))
    n = len(res.fun)
    return n * np.log(s / n) + 2 * (p + 2)  # +2 线性 a,b


# 网格初值：A∈{2.5,2.6}, B,C 小值, n2∈{2.3,2.5}
best2 = (1e18, None)
for A0 in [2.5, 2.55, 2.6, 2.65]:
    for n20 in [2.2, 2.3, 2.4, 2.5, 2.6]:
        x0 = [d0, A0, 0.05, 0.01, n20]
        r = least_squares(resid_two, x0, bounds=([d0 - 2, 2.4, -0.3, -0.1, 2.0],
                                                 [d0 + 2, 2.7, 0.3, 0.1, 2.7]),
                          method="trf", xtol=1e-12, ftol=1e-12, gtol=1e-12)
        a = aic(r, 5)
        if a < best2[0]:
            best2 = (a, r)
a2, r2 = best2
print("two:  d=%.4f A=%.4f B=%.4f C=%.4f n2=%.4f  AIC=%.1f RMSE=%.4f" %
      (r2.x[0], r2.x[1], r2.x[2], r2.x[3], r2.x[4], a2, np.sqrt(np.mean(r2.fun ** 2))))

best3 = (1e18, None)
for A0 in [2.5, 2.55, 2.6, 2.65]:
    for n20 in [2.2, 2.3, 2.4, 2.5, 2.6]:
        x0 = [d0, A0, 0.05, 0.01, n20, 0.01]
        r = least_squares(resid_airy, x0, bounds=([d0 - 2, 2.4, -0.3, -0.1, 2.0, 0.0],
                                                  [d0 + 2, 2.7, 0.3, 0.1, 2.7, 0.2]),
                          method="trf", xtol=1e-12, ftol=1e-12, gtol=1e-12)
        a = aic(r, 6)
        if a < best3[0]:
            best3 = (a, r)
a3, r3 = best3
print("airy: d=%.4f A=%.4f B=%.4f C=%.4f n2r=%.4f k2=%.4f  AIC=%.1f RMSE=%.4f" %
      (r3.x[0], r3.x[1], r3.x[2], r3.x[3], r3.x[4], r3.x[5], a3, np.sqrt(np.mean(r3.fun ** 2))))
print("dAIC(two-airy) = %.1f" % (a2 - a3))
n1m2 = float(np.mean(n_cauchy(w1, r2.x[1], r2.x[2], r2.x[3])))
n1m3 = float(np.mean(n_cauchy(w1, r3.x[1], r3.x[2], r3.x[3])))
rho2 = abs((N0 - n1m2) / (N0 + n1m2) * (n1m2 - r2.x[4]) / (n1m2 + r2.x[4]))
rho3 = abs((N0 - n1m3) / (N0 + n1m3) * (n1m3 - (r3.x[4] + 1j * r3.x[5])) / (n1m3 + (r3.x[4] + 1j * r3.x[5])))
print("rho two=%.5f  airy=%.5f" % (rho2, rho3))
