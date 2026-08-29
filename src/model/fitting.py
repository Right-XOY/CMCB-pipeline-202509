"""反演与可靠性模块：FFT 初值、DE 初值、L-M 拟合、统计检验。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg
from scipy.optimize import differential_evolution, least_squares
from scipy.stats import t as t_dist

from models import theory_R


@dataclass
class FitResult:
    """一次拟合的全部结果。"""
    model: str
    theta: np.ndarray          # [d, 色散系数...]（线性参数 a,b 已消元）
    d: float
    beta: np.ndarray           # 色散系数
    linear: dict               # 每角度线性校正 {角度: (a, b)}，未进优化器
    n_lin: int = 0             # 线性参数个数（每角度 a,b 各 1）
    res: object = None         # scipy least_squares 结果
    residual1: np.ndarray = field(default_factory=lambda: np.array([]))
    residual2: np.ndarray = field(default_factory=lambda: np.array([]))
    fit1: np.ndarray = field(default_factory=lambda: np.array([]))
    fit2: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def n_params(self) -> int:
        return len(self.theta)


# FFT 求厚度初值
def fft_init_d(wn: np.ndarray, R: np.ndarray, n_avg: float = 2.6,
               theta0_deg: float = 10.0, search_lo: float = 1000.0) -> float:
    """由反射谱 FFT 主频反推厚度初值 d0（μm）。

    反射谱 R(ν) ≈ DC + AC·cos(2π f0 ν)，f0 = 2 n̄ d cosθ̄₁ / 1e4。
    故 d0 = f0·1e4 / (2 n̄ cosθ̄₁)。n̄ 用平均折射率近似（色散使峰展宽，仅作初值）。
    """
    wn = np.asarray(wn, float)
    R = np.asarray(R, float)
    sel = wn >= search_lo
    wn, R = wn[sel], R[sel]
    dnu = float(np.median(np.diff(wn)))
    n_pts = len(R)

    sig = (R - R.mean()) * np.hanning(n_pts)
    F = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(n_pts, d=dnu)  # 单位：cm
    idx = 1 + int(np.argmax(np.abs(F[1:])))  # 跳过 DC 项

    # 抛物线峰值细化
    if 1 <= idx < len(freqs) - 1:
        y0, y1, y2 = np.abs(F[idx - 1]), np.abs(F[idx]), np.abs(F[idx + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            df = freqs[1] - freqs[0]
            idx = idx + 0.5 * (y0 - y2) / denom
    f0 = float(freqs[int(idx)]) if idx == int(idx) else float(np.interp(idx, np.arange(len(freqs)), freqs))

    theta1 = np.arcsin(np.sin(np.deg2rad(theta0_deg)) / n_avg)
    d0 = f0 * 1e4 / (2.0 * n_avg * np.cos(theta1))
    return float(d0)


# 线性校正闭式解：R ≈ a·T + b 的最小二乘 (a, b)
def _lin_solve(T: np.ndarray, R: np.ndarray) -> tuple[float, float]:
    """对固定理论值 T，用最小二乘求 (a, b)（不进入优化器）。"""
    A = np.column_stack([T, np.ones_like(T)])
    sol, *_ = np.linalg.lstsq(A, R, rcond=None)
    return float(sol[0]), float(sol[1])


# DE 求色散系数初值（d 固定，线性校正闭式消元）
def de_init_beta(wn1, R1, wn2, R2, d0: float, model: str, n2: float,
                 angles, bounds_beta, seed: int = 0) -> np.ndarray:
    """差分进化全局搜索色散系数初值（厚度固定为 d0，与 L-M 同口径）。"""
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


# L-M 非线性最小二乘拟合（线性参数 a,b 每角度闭式消元）
def lm_fit(wn1, R1, wn2, R2, d0: float, beta0: np.ndarray, model: str, n2: float,
           angles, bounds, seed: int = 0) -> FitResult:
    """L-M 双角度联合拟合（若 R2 为 None 则仅拟合角度 1）。

    优化变量 θ = [d, 色散系数...]。每个角度独立引入线性校正
    R_obs,i ≈ a_i·T_i + b_i（a 增益、b DC 偏移），(a_i, b_i) 不进优化器，
    每次残差评估时由线性最小二乘闭式求解（profile 消元）。
    """
    theta0 = np.r_[d0, np.asarray(beta0, float)]
    use2 = R2 is not None

    def residuals(th):
        d = th[0]
        beta = th[1:]
        T1 = theory_R(wn1, d, beta, angles[0], n2, model)
        a1, b1 = _lin_solve(T1, R1)
        e1 = a1 * T1 + b1 - R1
        if not use2:
            return e1
        T2 = theory_R(wn2, d, beta, angles[1], n2, model)
        a2, b2 = _lin_solve(T2, R2)
        e2 = a2 * T2 + b2 - R2
        return np.concatenate([e1, e2])

    # bounds: list[(lo, hi)] -> (lb, ub) 两个数组（least_squares 要求的格式）
    lb = np.array([b[0] for b in bounds], float)
    ub = np.array([b[1] for b in bounds], float)
    res = least_squares(residuals, theta0, bounds=(lb, ub), method="trf",
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=1000)

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


# 可靠性统计
def fit_statistics(res, n_params: int, R1, R2=None) -> dict:
    """拟合优度统计：SSE / RMSE / R² / AIC / BIC。"""
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


def parameter_uncertainty(res, param_names: list[str], n_lin: int = 0
                          ) -> tuple[np.ndarray, np.ndarray, dict]:
    """由 L-M 雅可比矩阵估计参数标准差与 95% 置信区间。

    协方差 Σ = σ̂² (JᵀJ)⁻¹，σ̂² = SSE/(m−n−n_lin)，置信区间用 t 分布。
    n_lin 为闭式消元的线性参数个数（自由度需一并扣除）。
    返回 (sd, ci 矩阵[lo, hi]，以及原始协方差)。
    """
    J = res.jac                       # (m, n)
    m, n = J.shape
    dof = m - n - n_lin
    s2 = float(np.sum(res.fun ** 2) / max(dof, 1))
    cov = s2 * linalg.pinv(J.T @ J)
    sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
    tval = float(t_dist.ppf(0.975, max(dof, 1)))
    ci = np.column_stack([res.x - tval * sd, res.x + tval * sd])
    info = {"sigma2": s2, "t_value": tval, "dof": dof, "cov": cov}
    return sd, ci, info


# 抗噪声能力测试
def noise_robustness_test(wn1, R1, wn2, R2, d_fit: float, beta_fit: np.ndarray,
                          model: str, n2: float, angles, bounds,
                          levels=(0.01, 0.02, 0.05), nrep: int = 5, seed: int = 0) -> dict:
    """向实测谱加入不同等级高斯噪声，重复拟合，统计厚度 d 的漂移。

    噪声 σ_n = eta · std(R_obs)。以最终拟合参数 (d_fit, beta_fit) 为初值，
    反映对最优解的扰动鲁棒性（避免加噪后跳入其它周期分支）。
    """
    rng = np.random.default_rng(seed)
    out = {}
    for eta in levels:
        ds = []
        for _ in range(nrep):
            n1 = rng.normal(0.0, eta * R1.std(), R1.shape)
            n2 = rng.normal(0.0, eta * R2.std(), R2.shape)
            fit = lm_fit(wn1, R1 + n1, wn2, R2 + n2, d_fit, beta_fit,
                         model, n2, angles, bounds)
            ds.append(fit.d)
        out[str(eta)] = {"d_mean": float(np.mean(ds)), "d_std": float(np.std(ds)),
                         "d_min": float(np.min(ds)), "d_max": float(np.max(ds))}
    return out


# 双角度一致性：两角度独立拟合 vs 联合拟合
def single_angle_cross_check(wn1, R1, wn2, R2, d0: float, beta0: np.ndarray,
                             model: str, n2: float, angles, bounds,
                             seed: int = 0) -> dict:
    """附件1、附件2 各自独立拟合的 d，与联合拟合对比，检验"同一厚度"假设。"""
    fit1 = lm_fit(wn1, R1, None, None, d0, beta0, model, n2, angles, bounds, seed=seed)
    fit2 = lm_fit(wn2, R2, None, None, d0, beta0, model, n2, angles, bounds, seed=seed)
    return {"d_angle1": float(fit1.d), "d_angle2": float(fit2.d)}
