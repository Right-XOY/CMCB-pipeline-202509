# -*- coding: utf-8 -*-
"""问题三主流程：多光束干涉（Airy）模型与硅外延层厚度反演。

运行方式（在项目根目录）：
    python src/model/q3.py

任务：
1. 用 Airy 公式（单层薄膜精确解，含复折射率/吸收，s/p 偏振平均）建模多光束；
2. 对附件3、附件4（硅片）预处理 → FFT 估 d0 → DE 色散/吸收初值 → L-M Airy 拟合；
3. 判定多光束干涉：双光束残差谐波检验、界面反射率 ρ、Airy vs 双光束 AIC/BIC；
4. 对附件1、附件2（SiC）用同一套判据检验是否需要 Airy 修正；
5. 可靠性检验（残差统计、参数置信区间、双角度一致性）。

数值结果输出到 outputs/result/，不绘图（绘图交给 src/visualize/ 生图阶段）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import differential_evolution, least_squares

from preprocess import (find_cutoff_stft, hampel_filter, load_spectra,
                        resample, sg_smooth)
from models import airy_reflectance_avg, n_cauchy, two_beam_reflectance_avg
from fitting import fft_init_d, fit_statistics, parameter_uncertainty

# 物理常量与入射角
N0 = 1.0
ANG1, ANG2 = 10.0, 15.0
ANGLES = (ANG1, ANG2)

# 硅片 FFT 初值用平均折射率（红外区 n ≈ 3.42）
N_SI_AVG = 3.42
# 硅外延层折射率（红外区，固定为文献值；色散极弱，取常数近似）
N1_SI = 3.42
# 硅片预处理裁剪点（自由载流子 Drude 反射区之上，2000 cm⁻¹ 起条纹干净）
SI_CUTOFF = 2000.0
PREPROC = dict(hampel_win=11, hampel_t=3.0, sg_win=15, sg_poly=3)

# Airy 模型参数 θ = [d, n2_real, kappa2, sigma_d]
#   n1 = N1_SI（外延层折射率固定，轻掺杂近似透明）
#   n2 = n2_real + i·κ2（衬底，重掺杂自由载流子吸收）
#   σ_d：光斑内厚度不均匀标准差，解释条纹对比度随波数的高斯衰减
AIRY_NAMES = ["d(um)", "n2_real", "kappa2", "sigma_d(um)"]
AIRY_BOUNDS = [(0.5, 200.0), (2.5, 3.6), (0.0, 1.0), (0.0, 0.5)]
AIRY_DE_BOUNDS = [(2.5, 3.6), (0.0, 1.0), (0.0, 0.5)]
AIRY_X0 = [3.1, 0.1, 0.1]                # DE 搜索前参考初值（衬底/不均匀）

# 双光束模型参数 θ = [d, n2, sigma_d]（实数折射率，一阶近似 + 厚度平均）
#   与 Airy 保持相同自由度（线性校正、σ_d），仅差"多光束高阶项"，供 AIC/BIC 公平对比
TWO_NAMES = ["d(um)", "n2", "sigma_d(um)"]
TWO_BOUNDS = [(0.5, 200.0), (2.5, 3.6), (0.0, 0.5)]
TWO_DE_BOUNDS = [(2.5, 3.6), (0.0, 0.5)]
TWO_X0 = [3.1, 0.1]

# 碳化硅（附件 1/2）：n1 用 Cauchy 色散（SiC 红外色散显著），衬底 n2 固定
N2_SIC = 2.65                # SiC 衬底折射率（问题二固定，红外区近似平稳）
SIC_CUTOFF_FALLBACK = 1000.0  # STFT 定界失效时的回退裁剪点（cm⁻¹）

# SiC 双光束 / Airy 参数 θ = [d, A, B, C, sigma_d]
#   n1(ν) = A + B/λ² + C/λ⁴（Cauchy，λ 单位 μm）
#   n2 = N2_SIC（衬底半绝缘、近似无吸收，实数）
#   双光束与 Airy 参数完全相同，仅差多光束高阶项（谐波），供 AIC/BIC 公平对比
SIC_NAMES = ["d(um)", "A", "B", "C", "sigma_d(um)"]
SIC_CAUCHY_BOUNDS = [(1.5, 3.5), (-1.0, 1.0), (-1.0, 1.0)]
SIC_SD_BOUND = (0.0, 0.5)

SEED = 2025


def load_config() -> tuple[Path, dict]:
    root = Path(__file__).resolve().parents[2]
    with open(root / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return root, cfg


def save_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  [output] {path.name}")


def save_csv(path: Path, header: str, cols: list[np.ndarray]) -> None:
    arr = np.column_stack(cols)
    fmt = ",".join(["%.6f"] * arr.shape[1])
    np.savetxt(path, arr, fmt=fmt, header=header, comments="")
    print(f"  [output] {path.name}  ({arr.shape[0]} rows)")


# 硅片预处理：裁剪低波数 → 重采样 → Hampel → SG 平滑（保留 DC，供完整反射率拟合）
def preprocess_si(wn: np.ndarray, R: np.ndarray, cutoff: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    mask = wn >= cutoff
    wn, R = wn[mask], R[mask]
    wn, R = resample(wn, R)
    R = hampel_filter(R, win=PREPROC["hampel_win"], t=PREPROC["hampel_t"])
    R = sg_smooth(R, win=PREPROC["sg_win"], polyorder=PREPROC["sg_poly"])
    return wn, R


# SiC 预处理：STFT 自动定界裁剪 Reststrahlen 低波数带，其余同硅片（保留 DC）
def preprocess_sic(wn: np.ndarray, R: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, float]:
    cutoff = find_cutoff_stft(wn, R)
    if not (700.0 <= cutoff <= 1600.0):
        cutoff = SIC_CUTOFF_FALLBACK
    wn_p, R_p = preprocess_si(wn, R, cutoff)
    return wn_p, R_p, cutoff


def detrend_poly(wn: np.ndarray, R: np.ndarray, deg: int = 2) -> np.ndarray:
    """多项式去慢变背景，仅用于 FFT 厚度初值估计。

    SiC 完整反射率的慢变 DC 漂移（色散 + 自由载流子尾）幅度大于干涉条纹，
    直接对完整反射率做 FFT 会让主频被背景漂移主导。这里用 deg 次多项式拟合
    背景并扣除，保留干涉振荡后交给 fft_init_d，拟合仍用完整反射率。
    """
    wn = np.asarray(wn, float)
    R = np.asarray(R, float)
    p = np.polyfit(wn, R, deg)
    return R - np.polyval(p, wn)


# Airy 模型（固定 n1 + 衬底吸收 + 厚度不均匀）：θ = [d, n2_real, kappa2, sigma_d]
def airy_R(nu, theta_deg, theta) -> np.ndarray:
    d, n2r, k2, sd = theta
    n2 = n2r + 1j * k2
    return airy_reflectance_avg(nu, d, sd, N1_SI, n2, theta_deg, N0)


# 双光束模型（固定 n1 + 厚度平均，无吸收）：θ = [d, n2, sigma_d]
def twobeam_R(nu, theta_deg, theta) -> np.ndarray:
    d, n2, sd = theta
    return two_beam_reflectance_avg(nu, d, sd, N1_SI, n2, theta_deg, N0)


# SiC 双光束模型（Cauchy 色散 n1 + 厚度平均，衬底 n2 固定）：θ = [d, A, B, C, sigma_d]
def twobeam_R_sic(nu, theta_deg, theta) -> np.ndarray:
    d, A, B, C, sd = theta
    n1 = n_cauchy(nu, A, B, C)
    return two_beam_reflectance_avg(nu, d, sd, n1, N2_SIC, theta_deg, N0)


# SiC Airy 模型（Cauchy 色散 n1 + 厚度平均，衬底 n2 固定）：θ = [d, A, B, C, sigma_d]
def airy_R_sic(nu, theta_deg, theta) -> np.ndarray:
    d, A, B, C, sd = theta
    n1 = n_cauchy(nu, A, B, C)
    return airy_reflectance_avg(nu, d, sd, n1, N2_SIC, theta_deg, N0)


def _lin_solve(T: np.ndarray, R: np.ndarray) -> tuple[float, float]:
    """对固定理论值 T 做最小二乘求线性校正 R ≈ a·T + b（闭式消元）。"""
    A = np.column_stack([T, np.ones_like(T)])
    sol, *_ = np.linalg.lstsq(A, R, rcond=None)
    return float(sol[0]), float(sol[1])


def make_residual(wn1, R1, wn2, R2, model_func):
    """拼接双角度残差，每角度独立线性校正 R_obs ≈ a·T + b（闭式消元）。

    两角度反射率的绝对 DC 水平存在仪器/角度校准偏差（实测相差约 2.7%），
    用每角度独立的 (a, b) 校正吸收，不进入优化器，与问题二口径一致。
    """
    def resid(theta):
        T1 = model_func(wn1, ANG1, theta)
        a1, b1 = _lin_solve(T1, R1)
        e1 = a1 * T1 + b1 - R1
        T2 = model_func(wn2, ANG2, theta)
        a2, b2 = _lin_solve(T2, R2)
        e2 = a2 * T2 + b2 - R2
        return np.concatenate([e1, e2])
    return resid


def fit_model(wn1, R1, wn2, R2, d0, model_func, bounds, de_bounds, seed=SEED):
    """通用拟合：DE 固定 d0 搜其余参数初值，再 L-M 联合拟合全部参数。"""
    resid = make_residual(wn1, R1, wn2, R2, model_func)

    def sse_other(x):
        return float(np.sum(resid(np.r_[d0, x]) ** 2))

    de = differential_evolution(sse_other, de_bounds, seed=seed,
                                tol=1e-8, popsize=20, maxiter=200,
                                polish=True, updating="immediate")
    x0 = np.r_[d0, de.x]
    lb = np.array([b[0] for b in bounds], float)
    ub = np.array([b[1] for b in bounds], float)
    res = least_squares(resid, x0, bounds=(lb, ub), method="trf",
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=3000)
    return res, de.x


def residual_harmonics(wn, resid, f0, nharm=3, half_peak=3.0):
    """残差频谱在 k·f0 (k=2,3) 处的局部谐波信噪比 H_k。

    H_k = 谐波峰幅值 / 峰两侧局部邻带噪声标准差（与推导过程 §4.2 一致）。
    """
    dnu = float(np.median(np.diff(wn)))
    F = np.abs(np.fft.rfft(resid * np.hanning(len(resid))))
    freqs = np.fft.rfftfreq(len(resid), d=dnu)
    # freqs 单位为 cm（1/dnu 量纲），频率分辨率 dfreq = freqs[1]-freqs[0]
    dfreq = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0
    out = {}
    for k in range(2, nharm + 1):
        target = k * f0
        half = half_peak * dfreq        # 峰半宽（half_peak 个频率 bin）
        sel_peak = np.abs(freqs - target) <= half
        if not np.any(sel_peak):
            out[k] = 0.0
            continue
        peak = float(F[sel_peak].max())
        sel_noise = (np.abs(freqs - target) > half) & \
                    (np.abs(freqs - target) <= 3.0 * half)
        sigma = float(F[sel_noise].std()) if np.sum(sel_noise) > 2 else 1.0
        out[k] = peak / max(sigma, 1e-12)
    return out


def rho_estimate(n1, n2):
    """界面反射率 ρ = |r01 · r12|（垂直入射近似，n1、n2 可为复数）。"""
    r01 = (N0 - n1) / (N0 + n1)
    r12 = (n1 - n2) / (n1 + n2)
    return float(abs(r01 * r12))


def solve_silicon(wn1, R1, wn2, R2, d0, seed=SEED) -> dict:
    """对硅片做双光束 + Airy 双模型反演与多光束判定。"""
    n_lin = 4  # 双角度各 (a, b) 线性校正参数，闭式消元
    # 1) 双光束拟合（判定基准 + 残差谐波）
    res_two, _ = fit_model(wn1, R1, wn2, R2, d0, twobeam_R,
                           TWO_BOUNDS, TWO_DE_BOUNDS, seed)
    theta_two = res_two.x
    d_two, n2_two, sd_two = theta_two
    n1_ = len(wn1)
    r_two1 = res_two.fun[:n1_]
    stats_two = fit_statistics(res_two, len(theta_two) + n_lin, R1, R2)
    # 残差谐波（用 10° 残差；f0 = 2 n1 d cosθ1 / 1e4，n1 固定为 N1_SI）
    th1 = np.arcsin(np.sin(np.deg2rad(ANG1)) / N1_SI)
    f0 = 2.0 * N1_SI * d_two * np.cos(th1) / 1e4
    harm_two = residual_harmonics(wn1, r_two1, f0)
    rho_two = rho_estimate(N1_SI, n2_two)

    # 2) Airy 拟合
    res_airy, _ = fit_model(wn1, R1, wn2, R2, d0, airy_R,
                            AIRY_BOUNDS, AIRY_DE_BOUNDS, seed)
    theta_airy = res_airy.x
    d_airy, n2r_airy, k2_airy, sd_airy = theta_airy
    r_airy1 = res_airy.fun[:n1_]
    stats_airy = fit_statistics(res_airy, len(theta_airy) + n_lin, R1, R2)
    sd_a, ci_a, _ = parameter_uncertainty(res_airy, AIRY_NAMES, n_lin=n_lin)
    # Airy 主频 f0_airy 与 ρ 用固定外延层折射率 N1_SI
    th1_airy = np.arcsin(np.sin(np.deg2rad(ANG1)) / N1_SI)
    f0_airy = 2.0 * N1_SI * d_airy * np.cos(th1_airy) / 1e4
    harm_airy = residual_harmonics(wn1, r_airy1, f0_airy)
    rho_airy = rho_estimate(N1_SI, n2r_airy + 1j * k2_airy)

    return {
        "d0": d0,
        "two_beam": {
            "theta": theta_two.tolist(), "names": TWO_NAMES,
            "stats": stats_two, "rho": rho_two,
            "harmonics": {str(k): v for k, v in harm_two.items()},
        },
        "airy": {
            "theta": theta_airy.tolist(), "names": AIRY_NAMES,
            "stats": stats_airy, "rho": rho_airy,
            "harmonics": {str(k): v for k, v in harm_airy.items()},
            "sd": sd_a.tolist(), "ci": ci_a.tolist(),
        },
        "f0": f0,
        "delta_aic": stats_two["AIC"] - stats_airy["AIC"],
        "delta_bic": stats_two["BIC"] - stats_airy["BIC"],
        "n_data": stats_airy["n_data"],
    }


def solve_sic(wn1, R1, wn2, R2, d0, seed=SEED) -> dict:
    """对 SiC 做双光束 + Airy 双模型反演与多光束判定（与硅片同一套判据）。

    SiC 外延层 n1 用 Cauchy 色散（λ∈[2.5,6.4] μm 内 n1≈2.5~2.7），衬底 n2 固定；
    双光束与 Airy 参数完全相同，仅差多光束高阶项，供 AIC/BIC 公平对比。
    厚度边界按 FFT 初值 d0 收窄到 [d0-2, d0+2]，避免多周期分支跳变。
    """
    n_lin = 4  # 双角度各 (a, b) 线性校正参数，闭式消元
    d_lo, d_hi = max(0.5, d0 - 2.0), d0 + 2.0
    sic_bounds = [(d_lo, d_hi)] + list(SIC_CAUCHY_BOUNDS) + [SIC_SD_BOUND]
    sic_de_bounds = list(SIC_CAUCHY_BOUNDS) + [SIC_SD_BOUND]

    # 1) 双光束拟合（判定基准 + 残差谐波）
    res_two, _ = fit_model(wn1, R1, wn2, R2, d0, twobeam_R_sic,
                           sic_bounds, sic_de_bounds, seed)
    theta_two = res_two.x
    d_two, A2, B2, C2, sd_two = theta_two
    n1_ = len(wn1)
    r_two1 = res_two.fun[:n1_]
    stats_two = fit_statistics(res_two, len(theta_two) + n_lin, R1, R2)
    # 主频用带内平均相位折射率近似（色散使 f0 微展宽，仅作谐波定位）
    n1m = float(n_cauchy(wn1, A2, B2, C2).mean())
    th1 = np.arcsin(np.sin(np.deg2rad(ANG1)) / n1m)
    f0 = 2.0 * n1m * d_two * np.cos(th1) / 1e4
    harm_two = residual_harmonics(wn1, r_two1, f0)
    rho_two = rho_estimate(n1m, N2_SIC)

    # 2) Airy 拟合（同参数，仅多光束高阶项不同）
    res_airy, _ = fit_model(wn1, R1, wn2, R2, d0, airy_R_sic,
                            sic_bounds, sic_de_bounds, seed)
    theta_airy = res_airy.x
    d_airy, Aa, Ba, Ca, sd_airy = theta_airy
    r_airy1 = res_airy.fun[:n1_]
    stats_airy = fit_statistics(res_airy, len(theta_airy) + n_lin, R1, R2)
    sd_a, ci_a, _ = parameter_uncertainty(res_airy, SIC_NAMES, n_lin=n_lin)
    n1ma = float(n_cauchy(wn1, Aa, Ba, Ca).mean())
    th1a = np.arcsin(np.sin(np.deg2rad(ANG1)) / n1ma)
    f0_airy = 2.0 * n1ma * d_airy * np.cos(th1a) / 1e4
    harm_airy = residual_harmonics(wn1, r_airy1, f0_airy)
    rho_airy = rho_estimate(n1ma, N2_SIC)

    return {
        "d0": d0,
        "two_beam": {
            "theta": theta_two.tolist(), "names": SIC_NAMES,
            "stats": stats_two, "rho": rho_two,
            "harmonics": {str(k): v for k, v in harm_two.items()},
        },
        "airy": {
            "theta": theta_airy.tolist(), "names": SIC_NAMES,
            "stats": stats_airy, "rho": rho_airy,
            "harmonics": {str(k): v for k, v in harm_airy.items()},
            "sd": sd_a.tolist(), "ci": ci_a.tolist(),
        },
        "f0": f0,
        "delta_aic": stats_two["AIC"] - stats_airy["AIC"],
        "delta_bic": stats_two["BIC"] - stats_airy["BIC"],
        "n_data": stats_airy["n_data"],
    }


def main() -> None:
    root, cfg = load_config()
    raw = cfg["data"]["raw_files"]
    result_dir = root / cfg["result"]["dir"]
    result_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Question 3: silicon epi thickness via multi-beam (Airy) model")
    print("=" * 60)

    # 1. 读取与预处理（附件3/4 硅片）
    print("\n[1] read & preprocess (Si wafer: f3=10deg, f4=15deg)")
    spec = {}
    for key, path in [("f3", raw["f3"]), ("f4", raw["f4"])]:
        wn, R = load_spectra(root / path)
        wn_p, R_p = preprocess_si(wn, R, SI_CUTOFF)
        spec[key] = (wn_p, R_p)
        print(f"  {key}: {len(wn)} pts -> cutoff {SI_CUTOFF:.1f} cm-1 -> {len(wn_p)} pts")
        save_csv(result_dir / f"q3_processed_{key}.csv",
                 "wavenumber_cm-1,reflectance_%", [wn_p, R_p])

    wn1, R1 = spec["f3"]
    wn2, R2 = spec["f4"]

    # 2. FFT 厚度初值
    print("\n[2] FFT thickness init")
    d0 = fft_init_d(wn1, R1, n_avg=N_SI_AVG, theta0_deg=ANG1, search_lo=SI_CUTOFF)
    print(f"  d0 = {d0:.3f} um")

    # 3. 硅片双模型反演与判定
    print("\n[3] Si wafer: two-beam vs Airy inversion & decision")
    si = solve_silicon(wn1, R1, wn2, R2, d0)
    tb, ay = si["two_beam"], si["airy"]
    print(f"  two-beam: d={tb['theta'][0]:.4f} um  RMSE={tb['stats']['RMSE']:.4f}  "
          f"R2={tb['stats']['R2']:.4f}  AIC={tb['stats']['AIC']:.2f}  rho={tb['rho']:.4f}")
    print(f"  two-beam harmonics: H2={tb['harmonics'].get('2',0):.2f} "
          f"H3={tb['harmonics'].get('3',0):.2f}")
    print(f"  airy:      d={ay['theta'][0]:.4f} um  RMSE={ay['stats']['RMSE']:.4f}  "
          f"R2={ay['stats']['R2']:.4f}  AIC={ay['stats']['AIC']:.2f}  rho={ay['rho']:.4f}")
    print(f"  airy harmonics: H2={ay['harmonics'].get('2',0):.2f} "
          f"H3={ay['harmonics'].get('3',0):.2f}")
    print(f"  delta_AIC(two-airy)={si['delta_aic']:.2f}  delta_BIC={si['delta_bic']:.2f}")
    print(f"  airy theta = {np.round(ay['theta'], 5)}")
    print(f"  airy d 95%CI = [{ay['ci'][0][0]:.4f}, {ay['ci'][0][1]:.4f}]")

    save_json(result_dir / "q3_silicon.json", si)

    # 4. SiC 双模型反演与判定（附件1/2，同一套判据）
    print("\n[4] SiC wafer (f1/f2): two-beam vs Airy inversion & decision")
    spec_sic = {}
    for key, path in [("f1", raw["f1"]), ("f2", raw["f2"])]:
        wn, R = load_spectra(root / path)
        wn_p, R_p, cutoff = preprocess_sic(wn, R)
        spec_sic[key] = (wn_p, R_p)
        print(f"  {key}: {len(wn)} pts -> cutoff {cutoff:.1f} cm-1 -> {len(wn_p)} pts")
        save_csv(result_dir / f"q3_processed_{key}.csv",
                 "wavenumber_cm-1,reflectance_%", [wn_p, R_p])

    wn1s, R1s = spec_sic["f1"]
    wn2s, R2s = spec_sic["f2"]
    # FFT 厚度初值：先去慢变背景再取主频（SiC 背景漂移幅度大于条纹，见 detrend_poly）
    d0s = fft_init_d(wn1s, detrend_poly(wn1s, R1s), n_avg=2.6,
                     theta0_deg=ANG1, search_lo=1500.0)
    print(f"  d0 = {d0s:.3f} um")

    sic = solve_sic(wn1s, R1s, wn2s, R2s, d0s)
    tb_s, ay_s = sic["two_beam"], sic["airy"]
    print(f"  two-beam: d={tb_s['theta'][0]:.4f} um  RMSE={tb_s['stats']['RMSE']:.4f}  "
          f"R2={tb_s['stats']['R2']:.4f}  AIC={tb_s['stats']['AIC']:.2f}  rho={tb_s['rho']:.4f}")
    print(f"  two-beam harmonics: H2={tb_s['harmonics'].get('2',0):.2f} "
          f"H3={tb_s['harmonics'].get('3',0):.2f}")
    print(f"  airy:      d={ay_s['theta'][0]:.4f} um  RMSE={ay_s['stats']['RMSE']:.4f}  "
          f"R2={ay_s['stats']['R2']:.4f}  AIC={ay_s['stats']['AIC']:.2f}  rho={ay_s['rho']:.4f}")
    print(f"  airy harmonics: H2={ay_s['harmonics'].get('2',0):.2f} "
          f"H3={ay_s['harmonics'].get('3',0):.2f}")
    print(f"  delta_AIC(two-airy)={sic['delta_aic']:.2f}  delta_BIC={sic['delta_bic']:.2f}")
    print(f"  airy theta = {np.round(ay_s['theta'], 5)}")
    print(f"  airy d 95%CI = [{ay_s['ci'][0][0]:.4f}, {ay_s['ci'][0][1]:.4f}]")

    save_json(result_dir / "q3_sic.json", sic)

    print("\nDone. results -> outputs/result/.")


if __name__ == "__main__":
    main()
