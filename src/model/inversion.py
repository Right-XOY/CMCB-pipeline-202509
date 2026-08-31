from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg
from scipy.optimize import differential_evolution, least_squares
from scipy.stats import t as t_dist

from optics import theory_R


@dataclass
class FitResult:
    model: str
    theta: np.ndarray
    d: float
    beta: np.ndarray
    linear: dict
    n_lin: int = 0
    res: object = None
    residual1: np.ndarray = field(default_factory=lambda: np.array([]))
    residual2: np.ndarray = field(default_factory=lambda: np.array([]))
    fit1: np.ndarray = field(default_factory=lambda: np.array([]))
    fit2: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def n_params(self) -> int:
        return len(self.theta)


def fft_init_d(wn, R, n_avg=2.6, theta0_deg=10.0, search_lo=1000.0):
    # d0 = f0·1e4/(2·n̄·cosθ̄₁)
    wn = np.asarray(wn, float)
    R = np.asarray(R, float)
    sel = wn >= search_lo
    wn, R = wn[sel], R[sel]
    dnu = float(np.median(np.diff(wn)))
    n_pts = len(R)

    sig = (R - R.mean()) * np.hanning(n_pts)
    F = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(n_pts, d=dnu)
    idx = 1 + int(np.argmax(np.abs(F[1:])))

    if 1 <= idx < len(freqs) - 1:
        y0, y1, y2 = np.abs(F[idx - 1]), np.abs(F[idx]), np.abs(F[idx + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            idx = idx + 0.5 * (y0 - y2) / denom
    f0 = float(freqs[int(idx)]) if idx == int(idx) else float(
        np.interp(idx, np.arange(len(freqs)), freqs))

    theta1 = np.arcsin(np.sin(np.deg2rad(theta0_deg)) / n_avg)
    return float(f0 * 1e4 / (2.0 * n_avg * np.cos(theta1)))


def _lin_solve(T, R):
    A = np.column_stack([T, np.ones_like(T)])
    sol, *_ = np.linalg.lstsq(A, R, rcond=None)
    return float(sol[0]), float(sol[1])


def de_init_beta(wn1, R1, wn2, R2, d0, model, n2, angles, bounds_beta, seed=0):
    def objective(beta):
        T1 = theory_R(wn1, d0, beta, angles[0], n2, model)
        a1, b1 = _lin_solve(T1, R1)
        e1 = a1 * T1 + b1 - R1
        T2 = theory_R(wn2, d0, beta, angles[1], n2, model)
        a2, b2 = _lin_solve(T2, R2)
        e2 = a2 * T2 + b2 - R2
        return float(np.sum(e1 ** 2) + np.sum(e2 ** 2))

    res = differential_evolution(
        objective, bounds_beta, seed=seed, tol=1e-8,
        popsize=20, maxiter=200, polish=True, updating="immediate",
    )
    return np.asarray(res.x, float)


def lm_fit(wn1, R1, wn2, R2, d0, beta0, model, n2, angles, bounds, seed=0, fix_d=False):
    use2 = R2 is not None
    if fix_d:
        theta0 = np.asarray(beta0, float)
    else:
        theta0 = np.r_[d0, np.asarray(beta0, float)]

    def residuals(th):
        if fix_d:
            d, beta = d0, th
        else:
            d, beta = th[0], th[1:]
        T1 = theory_R(wn1, d, beta, angles[0], n2, model)
        a1, b1 = _lin_solve(T1, R1)
        e1 = a1 * T1 + b1 - R1
        if not use2:
            return e1
        T2 = theory_R(wn2, d, beta, angles[1], n2, model)
        a2, b2 = _lin_solve(T2, R2)
        e2 = a2 * T2 + b2 - R2
        return np.concatenate([e1, e2])

    bd = bounds[1:] if fix_d else bounds
    lb = np.array([b[0] for b in bd], float)
    ub = np.array([b[1] for b in bd], float)
    res = least_squares(residuals, theta0, bounds=(lb, ub), method="trf",
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=1000)

    if fix_d:
        theta = np.r_[d0, res.x]
        d, beta = d0, res.x
    else:
        theta = res.x
        d, beta = theta[0], theta[1:]
    T1 = theory_R(wn1, d, beta, angles[0], n2, model)
    a1, b1 = _lin_solve(T1, R1)
    fit1 = a1 * T1 + b1
    e1 = fit1 - R1
    linear = {angles[0]: (a1, b1)}
    if use2:
        T2 = theory_R(wn2, d, beta, angles[1], n2, model)
        a2, b2 = _lin_solve(T2, R2)
        fit2 = a2 * T2 + b2
        e2 = fit2 - R2
        linear[angles[1]] = (a2, b2)
    else:
        fit2 = np.array([])
        e2 = np.array([])
    return FitResult(model=model, theta=theta, d=float(d), beta=beta,
                     linear=linear, n_lin=2 * (2 if use2 else 1), res=res,
                     residual1=e1, residual2=e2, fit1=fit1, fit2=fit2)


def fit_statistics(res, n_params, R1, R2=None):
    fun = res.fun
    n_data = len(fun)
    SSE = float(np.sum(fun ** 2))
    rmse = float(np.sqrt(SSE / n_data))
    obs = np.concatenate([R1, R2]) if R2 is not None else R1
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    r2 = float(1.0 - SSE / ss_tot) if ss_tot > 0 else float("nan")
    aic = float(n_data * np.log(SSE / n_data) + 2 * n_params)
    bic = float(n_data * np.log(SSE / n_data) + n_params * np.log(n_data))
    return {"n_data": n_data, "SSE": SSE, "RMSE": rmse, "R2": r2, "AIC": aic, "BIC": bic}


def parameter_uncertainty(res, param_names, n_lin=0):
    J = res.jac
    m, n = J.shape
    dof = m - n - n_lin
    s2 = float(np.sum(res.fun ** 2) / max(dof, 1))
    cov = s2 * linalg.pinv(J.T @ J)
    sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
    tval = float(t_dist.ppf(0.975, max(dof, 1)))
    ci = np.column_stack([res.x - tval * sd, res.x + tval * sd])
    return sd, ci, {"sigma2": s2, "t_value": tval, "dof": dof, "cov": cov}


def noise_robustness_test(wn1, R1, wn2, R2, model, n2, angles,
                          bounds, de_bounds,
                          levels=(0.01, 0.02, 0.05), nrep=5,
                          seed=0, search_lo=1500.0,
                          profile_step=0.05, d_half=0.5):
    rng = np.random.default_rng(seed)
    out = {}
    for eta in levels:
        ds = []
        for rep in range(nrep):
            eps1 = rng.normal(0.0, eta * R1.std(), R1.shape)
            eps2 = rng.normal(0.0, eta * R2.std(), R2.shape)
            R1n, R2n = R1 + eps1, R2 + eps2
            d0n = fft_init_d(wn1, R1n, n_avg=2.6, theta0_deg=angles[0], search_lo=search_lo)
            beta0n = de_init_beta(wn1, R1n, wn2, R2n, d0n, model, n2, angles,
                                  de_bounds, seed=seed + rep)
            fit, _ = profile_scan_d(wn1, R1n, wn2, R2n, d0n, beta0n, model, n2,
                                    angles, bounds, d_half=d_half,
                                    step=profile_step, seed=seed + rep)
            ds.append(fit.d)
        out[str(eta)] = {"d_mean": float(np.mean(ds)), "d_std": float(np.std(ds)),
                         "d_min": float(np.min(ds)), "d_max": float(np.max(ds))}
    return out


def single_angle_cross_check(wn1, R1, wn2, R2, d0, beta0, model, n2, angles, bounds, seed=0):
    fit1 = lm_fit(wn1, R1, None, None, d0, beta0, model, n2, angles, bounds, seed=seed)
    fit2 = lm_fit(wn2, R2, None, None, d0, beta0, model, n2, angles, bounds, seed=seed)
    return {"d_angle1": float(fit1.d), "d_angle2": float(fit2.d)}


def profile_scan_d(wn1, R1, wn2, R2, d0, beta0, model, n2, angles, bounds,
                   d_half=0.5, step=0.01, seed=0):
    d_grid = np.arange(d0 - d_half, d0 + d_half + step * 0.5, step)
    sse = np.full_like(d_grid, np.inf)
    best_fit = None
    best_sse = np.inf
    cur_beta = beta0
    for i, d in enumerate(d_grid):
        fit = lm_fit(wn1, R1, wn2, R2, float(d), cur_beta, model, n2, angles,
                     bounds, seed=seed, fix_d=True)
        if any(v[0] <= 0.0 for v in fit.linear.values()):  # 正增益约束，排除反相分支
            continue
        sse[i] = float(np.sum(fit.res.fun ** 2))
        if sse[i] < best_sse:
            best_sse = sse[i]
            best_fit = fit
            cur_beta = fit.beta

    if best_fit is None:
        best_fit = lm_fit(wn1, R1, wn2, R2, float(d0), beta0, model, n2,
                          angles, bounds, seed=seed, fix_d=True)
        sse[:] = float(np.sum(best_fit.res.fun ** 2))
        best_sse = sse[0]
        d_opt = float(d0)
    else:
        valid = np.isfinite(sse)
        k = int(np.argmin(np.where(valid, sse, np.inf)))
        if 0 < k < len(d_grid) - 1 and np.isfinite(sse[k - 1]) and np.isfinite(sse[k + 1]):
            y0, y1, y2 = sse[k - 1], sse[k], sse[k + 1]
            denom = y0 - 2.0 * y1 + y2
            if abs(denom) > 1e-15:
                d_opt = d_grid[k] + 0.5 * step * (y0 - y2) / denom
            else:
                d_opt = d_grid[k]
        else:
            d_opt = d_grid[k]
        final_fit = lm_fit(wn1, R1, wn2, R2, float(d_opt), cur_beta, model, n2,
                           angles, bounds, seed=seed, fix_d=True)
        if all(v[0] > 0.0 for v in final_fit.linear.values()):
            best_fit = final_fit

    sse_min = float(np.sum(best_fit.res.fun ** 2))

    m = len(best_fit.res.fun)
    n_beta = best_fit.res.x.size
    n_lin = best_fit.n_lin
    dof = m - n_beta - n_lin - 1
    s2 = sse_min / max(dof, 1)
    valid = np.isfinite(sse)
    valid_idx = np.where(valid)[0]
    k = int(np.argmin(np.where(valid, sse, np.inf)))
    lo_k = max(valid_idx.min(), k - 3)
    hi_k = min(valid_idx.max() + 1, k + 4)
    dloc = d_grid[lo_k:hi_k]
    sloc = sse[lo_k:hi_k]
    dloc = dloc[np.isfinite(sloc)]
    sloc = sloc[np.isfinite(sloc)]
    a2 = 0.0
    if len(dloc) >= 3:
        c = np.polyfit(dloc, sloc, 2)
        a2 = float(c[0])
    if a2 > 1e-12:
        std_d = float(np.sqrt(s2 / a2))
    else:
        std_d = float(step)
    tval = float(t_dist.ppf(0.975, max(dof, 1)))
    d_ci = (float(d_opt - tval * std_d), float(d_opt + tval * std_d))

    profile = {
        "d_grid": d_grid.tolist(), "sse": sse.tolist(),
        "sse_min": sse_min, "d_opt": float(d_opt),
        "std_d": std_d, "d_ci": list(d_ci),
    }
    return best_fit, profile
