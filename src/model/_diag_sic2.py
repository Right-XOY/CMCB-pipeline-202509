# -*- coding: utf-8 -*-
"""诊断2：SiC 多光束，物理正确边界（n1~2.55-2.65，衬底 n2 可调且可复）。
测试不同 n2 范围与是否含吸收，比较双光束 vs Airy 的 AIC。"""
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
print("cutoff =", c1, " d0 =", round(d0, 4))
print("R1 mean/std/min/max = %.3f %.3f %.3f %.3f" % (r1.mean(), r1.std(), r1.min(), r1.max()))

N0 = 1.0


def lin(T, R):
    A = np.column_stack([T, np.ones_like(T)])
    s, *_ = np.linalg.lstsq(A, R, rcond=None)
    return s


def make_resid(model_func):
    def resid(th):
        T1 = model_func(w1, 10, th)
        a, b = lin(T1, r1)
        e1 = a * T1 + b - r1
        T2 = model_func(w2, 15, th)
        a, b = lin(T2, r2)
        e2 = a * T2 + b - r2
        return np.concatenate([e1, e2])
    return resid


def fit(resid, bounds, de_bounds, x0_full):
    def sse(x):
        return float(np.sum(resid(np.r_[x0_full[0], x]) ** 2))
    de = differential_evolution(sse, de_bounds, seed=2025, tol=1e-8,
                                popsize=20, maxiter=200, polish=True,
                                updating="immediate")
    x0 = np.r_[x0_full[0], de.x]
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    return least_squares(resid, x0, bounds=(lb, ub), method="trf",
                         xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=3000)


def stats(res, k):
    s = float(np.sum(res.fun ** 2))
    n = len(res.fun)
    obs = np.concatenate([r1, r2])
    r2v = 1 - s / np.sum((obs - obs.mean()) ** 2)
    rmse = np.sqrt(s / n)
    aic = n * np.log(s / n) + 2 * (len(res.x) + k)
    return rmse, r2v, aic


# n1 Cauchy A ∈ [2.5,2.7], B∈[-0.3,0.3], C∈[-0.1,0.1]
# 双光束 θ=[d,A,B,C,n2,sd]；Airy θ=[d,A,B,C,n2r,k2,sd]
def two(nu, td, th):
    d, A, B, C, n2, sd = th
    return two_beam_reflectance_avg(nu, d, sd, n_cauchy(nu, A, B, C), n2, td, N0)


def airy(nu, td, th):
    d, A, B, C, n2r, k2, sd = th
    return airy_reflectance_avg(nu, d, sd, n_cauchy(nu, A, B, C), n2r + 1j * k2, td, N0)


def run(n2_lo, n2_hi, kappa_hi, label):
    cb = [(2.5, 2.7), (-0.3, 0.3), (-0.1, 0.1)]
    b_two = [(d0 - 2, d0 + 2)] + cb + [(n2_lo, n2_hi), (0, 0.6)]
    de_two = cb + [(n2_lo, n2_hi), (0, 0.6)]
    khi = max(kappa_hi, 1e-6)
    b_airy = [(d0 - 2, d0 + 2)] + cb + [(n2_lo, n2_hi), (0, khi), (0, 0.6)]
    de_airy = cb + [(n2_lo, n2_hi), (0, khi), (0, 0.6)]

    rt = fit(make_resid(two), b_two, de_two, [d0])
    ra = fit(make_resid(airy), b_airy, de_airy, [d0])
    rm2, r22, a2 = stats(rt, 4)   # 每角度 (a,b) 共4线性参数
    rma, r2a, aa = stats(ra, 4)
    n1m2 = float(np.mean(n_cauchy(w1, rt.x[1], rt.x[2], rt.x[3])))
    n1ma = float(np.mean(n_cauchy(w1, ra.x[1], ra.x[2], ra.x[3])))
    rho2 = abs((N0 - n1m2) / (N0 + n1m2) * (n1m2 - rt.x[4]) / (n1m2 + rt.x[4]))
    rhoa = abs((N0 - n1ma) / (N0 + n1ma) * (n1ma - (ra.x[4] + 1j * ra.x[5])) / (n1ma + (ra.x[4] + 1j * ra.x[5])))
    print("\n[%s] n2∈[%.2f,%.2f] kappa≤%.2f" % (label, n2_lo, n2_hi, kappa_hi))
    print("  two: d=%.4f A=%.4f B=%.4f C=%.4f n2=%.4f sd=%.4f" % tuple(rt.x))
    print("       n1m=%.4f rho=%.5f RMSE=%.4f R2=%.4f AIC=%.1f" % (n1m2, rho2, rm2, r22, a2))
    print("  airy: d=%.4f A=%.4f B=%.4f C=%.4f n2r=%.4f k2=%.5f sd=%.4f" % tuple(ra.x))
    print("       n1m=%.4f rho=%.5f RMSE=%.4f R2=%.4f AIC=%.1f" % (n1ma, rhoa, rma, r2a, aa))
    print("  dAIC(two-airy)=%.1f" % (a2 - aa))


run(2.4, 2.7, 0.0, "n2实数可调(与n1同域)")
run(2.2, 2.6, 0.0, "n2可调偏低(重掺杂)")
run(2.2, 2.6, 0.2, "n2复(含衬底吸收)")
run(2.0, 2.6, 0.5, "n2复大范围")
