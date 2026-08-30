# -*- coding: utf-8 -*-
"""诊断：SiC 双光束 vs Airy（无 sigma_d），物理约束 n1 > n2（衬底重掺杂降折射率）。"""
import sys
import numpy as np
sys.path.insert(0, "src/model")

from preprocess import load_spectra
from q3 import preprocess_sic, detrend_poly, fft_init_d
from models import two_beam_reflectance_avg, airy_reflectance_avg, n_cauchy
from scipy.optimize import least_squares

wn1, R1 = load_spectra("data/raw/附件1.xlsx")
w1, data1, c1 = preprocess_sic(wn1, R1)
wn2, R2 = load_spectra("data/raw/附件2.xlsx")
w2, data2, _ = preprocess_sic(wn2, R2)
d0 = fft_init_d(w1, detrend_poly(w1, data1), n_avg=2.6, theta0_deg=10, search_lo=1500)
print(f"cutoff={c1:.0f}  d0={d0:.4f}  N={len(w1)}")
N0 = 1.0
obs = np.concatenate([data1, data2])


def lin(T, R):
    A = np.column_stack([T, np.ones_like(T)])
    s, *_ = np.linalg.lstsq(A, R, rcond=None)
    return s


def fit_two(x0, bounds):
    def resid(th):
        d, A, B, C, n2 = th
        T1 = two_beam_reflectance_avg(w1, d, 0.0, n_cauchy(w1, A, B, C), n2, 10, N0)
        a, b = lin(T1, data1)
        e1 = a * T1 + b - data1
        T2 = two_beam_reflectance_avg(w2, d, 0.0, n_cauchy(w2, A, B, C), n2, 15, N0)
        a, b = lin(T2, data2)
        e2 = a * T2 + b - data2
        return np.concatenate([e1, e2])
    return least_squares(resid, x0, bounds=bounds, method="trf",
                         xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=3000)


def fit_airy(x0, bounds):
    def resid(th):
        d, A, B, C, n2r, k2 = th
        T1 = airy_reflectance_avg(w1, d, 0.0, n_cauchy(w1, A, B, C), n2r + 1j * k2, 10, N0)
        a, b = lin(T1, data1)
        e1 = a * T1 + b - data1
        T2 = airy_reflectance_avg(w2, d, 0.0, n_cauchy(w2, A, B, C), n2r + 1j * k2, 15, N0)
        a, b = lin(T2, data2)
        e2 = a * T2 + b - data2
        return np.concatenate([e1, e2])
    return least_squares(resid, x0, bounds=bounds, method="trf",
                         xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=3000)


def stats(res, p):
    s = float(np.sum(res.fun ** 2))
    n = len(res.fun)
    rmse = np.sqrt(s / n)
    r2v = 1 - s / np.sum((obs - obs.mean()) ** 2)
    aic = n * np.log(s / n) + 2 * (p + 4)
    return rmse, r2v, aic


def report(res, p, name):
    rmse, r2v, aic = stats(res, p)
    tail = f" k2={res.x[5]:.4f}" if len(res.x) > 5 else ""
    print(f"  [{name}] d={res.x[0]:.4f} A={res.x[1]:.4f} B={res.x[2]:.4f} "
          f"C={res.x[3]:.4f} n2r={res.x[4]:.4f}{tail}  "
          f"RMSE={rmse:.4f} R2={r2v:.4f} AIC={aic:.1f}")
    return aic


# ---- 场景2：物理约束 n1 > n2，多组初值网格 ----
print("\n[场景2] 物理约束 n1 > n2（A∈[2.45,2.6], n2∈[2.0,2.45]）")
lb2 = [d0 - 2, 2.45, -0.1, -0.05, 2.0]
ub2 = [d0 + 2, 2.6, 0.1, 0.05, 2.45]
lb3 = [d0 - 2, 2.45, -0.1, -0.05, 2.0, 0.0]
ub3 = [d0 + 2, 2.6, 0.1, 0.05, 2.45, 0.2]
best2 = (1e18, None)
best3 = (1e18, None)
for A0 in [2.45, 2.5, 2.55, 2.6]:
    for n20 in [2.2, 2.3, 2.35, 2.4, 2.45]:
        res2 = fit_two([d0, A0, 0.02, 0.005, n20], (lb2, ub2))
        a2 = stats(res2, 5)[2]
        if a2 < best2[0]:
            best2 = (a2, res2)
        res3 = fit_airy([d0, A0, 0.02, 0.005, n20, 0.01], (lb3, ub3))
        a3 = stats(res3, 6)[2]
        if a3 < best3[0]:
            best3 = (a3, res3)
a2c, res2c = best2
a3c, res3c = best3
print("  --- 最优 ---")
report(res2c, 5, "two")
report(res3c, 6, "airy")
print(f"  dAIC(two-airy) = {a2c - a3c:.1f}")
n1m2 = float(np.mean(n_cauchy(w1, res2c.x[1], res2c.x[2], res2c.x[3])))
n1m3 = float(np.mean(n_cauchy(w1, res3c.x[1], res3c.x[2], res3c.x[3])))
rho2 = abs((N0 - n1m2) / (N0 + n1m2) * (n1m2 - res2c.x[4]) / (n1m2 + res2c.x[4]))
rho3 = abs((N0 - n1m3) / (N0 + n1m3) * (n1m3 - (res3c.x[4] + 1j * res3c.x[5])) / (n1m3 + (res3c.x[4] + 1j * res3c.x[5])))
print(f"  n1_mean two={n1m2:.4f} airy={n1m3:.4f}  rho two={rho2:.5f} airy={rho3:.5f}")
