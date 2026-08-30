# -*- coding: utf-8 -*-
"""诊断4：去掉 sigma_d + 固定增益 a=1（仅 DC 偏移 b 自由）+ n2 可调/复。
检查 SiC 是否得到物理解（n1≈2.6, n2≈2.4）且 Airy 显著优于双光束。"""
import sys
import numpy as np
sys.path.insert(0, ".")

from preprocess import load_spectra
from q3 import preprocess_sic, detrend_poly, fft_init_d
from models import two_beam_reflectance_avg, airy_reflectance_avg, n_cauchy
from scipy.optimize import least_squares

wn1, R1 = load_spectra("data/raw/附件1.xlsx")
w1, r1, c1 = preprocess_sic(wn1, R1)
wn2, R2 = load_spectra("data/raw/附件2.xlsx")
w2, r2, _ = preprocess_sic(wn2, R2)
d0 = fft_init_d(w1, detrend_poly(w1, r1), n_avg=2.6, theta0_deg=10, search_lo=1500)
print("cutoff", c1, "d0", round(d0, 4))
N0 = 1.0


# 固定 a=1，仅 b 自由（DC 偏移）。双角度。
def offb(T, R):
    b = float(np.mean(R - T))  # a=1 时 b 的闭式最优
    return b


def resid_two(th):
    d, A, B, C, n2 = th
    n1 = n_cauchy(w1, A, B, C)
    T1 = two_beam_reflectance_avg(w1, d, 0.0, n1, n2, 10, N0)
    e1 = T1 + offb(T1, r1) - r1
    n1 = n_cauchy(w2, A, B, C)
    T2 = two_beam_reflectance_avg(w2, d, 0.0, n1, n2, 15, N0)
    e2 = T2 + offb(T2, r2) - r2
    return np.concatenate([e1, e2])


def resid_airy(th):
    d, A, B, C, n2r, k2 = th
    n1 = n_cauchy(w1, A, B, C)
    T1 = airy_reflectance_avg(w1, d, 0.0, n1, n2r + 1j * k2, 10, N0)
    e1 = T1 + offb(T1, r1) - r1
    n1 = n_cauchy(w2, A, B, C)
    T2 = airy_reflectance_avg(w2, d, 0.0, n1, n2r + 1j * k2, 15, N0)
    e2 = T2 + offb(T2, r2) - r2
    return np.concatenate([e1, e2])


def aic(res, p):
    s = float(np.sum(res.fun ** 2))
    n = len(res.fun)
    return n * np.log(s / n) + 2 * (p + 2)  # +2 每角度 b（两角度）


cb = [(2.4, 2.7), (-0.3, 0.3), (-0.1, 0.1)]
b2 = [(d0 - 2, d0 + 2)] + cb + [(2.0, 2.7)]
b3 = [(d0 - 2, d0 + 2)] + cb + [(2.0, 2.7), (0.0, 0.2)]

best2 = (1e18, None)
for A0 in [2.5, 2.55, 2.6, 2.65]:
    for n20 in [2.2, 2.3, 2.4, 2.5, 2.6]:
        x0 = [d0, A0, 0.05, 0.01, n20]
        r = least_squares(resid_two, x0, bounds=(np.array([b[0] for b in b2]), np.array([b[1] for b in b2])),
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
        r = least_squares(resid_airy, x0, bounds=(np.array([b[0] for b in b3]), np.array([b[1] for b in b3])),
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
