# -*- coding: utf-8 -*-
"""临时诊断：SiC 是否有多光束。测试 n2 可调 + Airy 含衬底吸收 κ2 的情形。"""
import sys
import numpy as np
sys.path.insert(0, ".")

from preprocess import load_spectra
from q3 import preprocess_sic, detrend_poly, fft_init_d
from models import two_beam_reflectance_avg, airy_reflectance_avg, n_cauchy
from scipy.optimize import differential_evolution, least_squares

wn1, R1 = load_spectra("data/raw/附件1.xlsx")
w1, r1, c1 = preprocess_sic(wn1, R1)
wn2, R2 = load_spectra("data/raw/附件2.xlsx")
w2, r2, _ = preprocess_sic(wn2, R2)
d0 = fft_init_d(w1, detrend_poly(w1, r1), n_avg=2.6, theta0_deg=10, search_lo=1500)
print("d0 =", round(d0, 4))

N0 = 1.0


def lin(T, R):
    A = np.column_stack([T, np.ones_like(T)])
    s, *_ = np.linalg.lstsq(A, R, rcond=None)
    return s


def make_resid(model_func, n_lin_params):
    def resid(th):
        T1 = model_func(w1, 10, th)
        a, b = lin(T1, r1)
        e1 = a * T1 + b - r1
        T2 = model_func(w2, 15, th)
        a, b = lin(T2, r2)
        e2 = a * T2 + b - r2
        return np.concatenate([e1, e2])
    return resid


def fit(resid, bounds, de_bounds):
    def sse(x):
        return float(np.sum(resid(np.r_[d0, x]) ** 2))
    de = differential_evolution(sse, de_bounds, seed=2025, tol=1e-8,
                                popsize=20, maxiter=200, polish=True,
                                updating="immediate")
    x0 = np.r_[d0, de.x]
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    return least_squares(resid, x0, bounds=(lb, ub), method="trf",
                         xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=3000)


def stats(res):
    s = float(np.sum(res.fun ** 2))
    n = len(res.fun)
    obs = np.concatenate([r1, r2])
    r2v = 1 - s / np.sum((obs - obs.mean()) ** 2)
    rmse = np.sqrt(s / n)
    aic = n * np.log(s / n) + 2 * len(res.x)
    return rmse, r2v, aic


# 方案A：双光束，n2 可调（实数），θ=[d,A,B,C,n2,sd]
def two_AB(nu, td, th):
    d, A, B, C, n2, sd = th
    n1 = n_cauchy(nu, A, B, C)
    return two_beam_reflectance_avg(nu, d, sd, n1, n2, td, N0)

# 方案B：Airy，n2 复数，θ=[d,A,B,C,n2r,k2,sd]
def airy_AB(nu, td, th):
    d, A, B, C, n2r, k2, sd = th
    n1 = n_cauchy(nu, A, B, C)
    n2 = n2r + 1j * k2
    return airy_reflectance_avg(nu, d, sd, n1, n2, td, N0)


bounds_two = [(d0 - 2, d0 + 2), (2.3, 2.8), (-0.3, 0.3), (-0.1, 0.1), (2.0, 3.0), (0, 0.6)]
de_two = [(2.3, 2.8), (-0.3, 0.3), (-0.1, 0.1), (2.0, 3.0), (0, 0.6)]
bounds_airy = [(d0 - 2, d0 + 2), (2.3, 2.8), (-0.3, 0.3), (-0.1, 0.1), (2.0, 3.0), (0, 1.0), (0, 0.6)]
de_airy = [(2.3, 2.8), (-0.3, 0.3), (-0.1, 0.1), (2.0, 3.0), (0, 1.0), (0, 0.6)]

print("\n[方案A] 双光束 n2 可调")
rt = fit(make_resid(two_AB, 6), bounds_two, de_two)
rmse, r2v, aic = stats(rt)
print("  theta=", np.round(rt.x, 4))
print("  d=%.4f  n2=%.4f  RMSE=%.4f  R2=%.4f  AIC=%.1f" % (rt.x[0], rt.x[4], rmse, r2v, aic))
n1m = float(np.mean(n_cauchy(w1, rt.x[1], rt.x[2], rt.x[3])))
rho = abs((N0 - n1m) / (N0 + n1m) * (n1m - rt.x[4]) / (n1m + rt.x[4]))
print("  n1_mean=%.4f  rho=%.5f" % (n1m, rho))

print("\n[方案B] Airy n2 复数(含衬底吸收 kappa2)")
ra = fit(make_resid(airy_AB, 7), bounds_airy, de_airy)
rmse, r2v, aic = stats(ra)
print("  theta=", np.round(ra.x, 4))
print("  d=%.4f  n2r=%.4f  k2=%.4f  RMSE=%.4f  R2=%.4f  AIC=%.1f" % (ra.x[0], ra.x[4], ra.x[5], rmse, r2v, aic))
n1m = float(np.mean(n_cauchy(w1, ra.x[1], ra.x[2], ra.x[3])))
rho = abs((N0 - n1m) / (N0 + n1m) * (n1m - (ra.x[4] + 1j * ra.x[5])) / (n1m + (ra.x[4] + 1j * ra.x[5])))
print("  n1_mean=%.4f  rho=%.5f" % (n1m, rho))

print("\n[对比] 双光束 AIC - Airy AIC =", round(stats(rt)[2] - stats(ra)[2], 2))
