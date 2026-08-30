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
                        resample, sg_smooth, preprocess)
from models import airy_reflectance_avg, n_drude, n_sellmeier1, two_beam_reflectance_avg
from fitting import (de_init_beta, fft_init_d, fit_statistics,
                     parameter_uncertainty, profile_scan_d)

# 物理常量与入射角
N0 = 1.0
ANG1, ANG2 = 10.0, 15.0
ANGLES = (ANG1, ANG2)

# 硅片 FFT 初值用平均折射率（红外区 n ≈ 3.42）
N_SI_AVG = 3.42
# 硅片预处理裁剪点（自由载流子 Drude 反射区之上，2000 cm⁻¹ 起条纹干净）
SI_CUTOFF = 2000.0
PREPROC = dict(hampel_win=11, hampel_t=3.0, sg_win=15, sg_poly=3)

# 硅片外延层折射率 n1：轻掺杂硅在红外波段几乎无色散，取已知常数 n1≈3.42。
#   用 Sellmeier 单振子 n1²=1+Bλ²/(λ²−C) 的极限 B=10.7、C=0（n1²=1+B=11.7），
#   即 Q2 比选出的最优色散模型在"无色散硅"下的退化形式。
#   固定 n1 是关键：若把 B 也作为自由参数，会与每角度线性增益 a 耦合，产生
#   "n1→n2（两界面折射率相等→r12→0→条纹消失）"的退化解——增益 a 把近似平坦的
#   理论值放大成虚假低 SSE。n1 固定后，衬底折射率 n2 由条纹幅度唯一确定。
# 衬底 n2 用 Drude 色散 n2²(ν)=n_inf²−wp²/(ν²+iγν) 描述（重掺杂硅自由载流子
#   响应）：ν→∞ 时 n2→n_inf（接近本征硅折射率），低波数端自由载流子使 n2 实部
#   下降、引入吸收 κ2。Drude 使 r12(ν) 与条纹包络随波数衰减，且均匀作用于全部
#   谐波——既不破坏多光束高阶项，又解释"条纹对比度随波数衰减约 6 倍"的包络。
# σ_d（光斑内厚度不均匀）已取消：多光束干涉的必要条件之一是厚度均匀，故取
#   理想平行平面 σ_d=0 作为判定多光束的基准口径。
SI_B = 10.7
SI_C = 0.0
SI_N1 = float(np.sqrt(1.0 + SI_B))          # ≈ 3.4205

# 双光束模型 θ = [d, n_inf, wp, gamma]（衬底 n2 用 Drude 色散，一阶近似，理想平面 σ_d=0）
#   与 Airy 参数完全一致（仅差多光束高阶项），供 AIC/BIC 公平对比
TWO_NAMES = ["d(um)", "n_inf", "wp", "gamma"]
TWO_BOUNDS = [(0.5, 200.0), (2.5, 3.4), (300.0, 8000.0), (5.0, 500.0)]
TWO_DE_BOUNDS = [(2.5, 3.4), (300.0, 8000.0), (5.0, 500.0)]
TWO_X0 = [3.0, 2500.0, 100.0]

# Airy 模型参数 θ = [d, n_inf, wp, gamma]（Drude 色散 n2 含自由载流子吸收，多光束高阶项，σ_d=0）
AIRY_NAMES = ["d(um)", "n_inf", "wp", "gamma"]
AIRY_BOUNDS = [(0.5, 200.0), (2.5, 3.4), (300.0, 8000.0), (5.0, 500.0)]
AIRY_DE_BOUNDS = [(2.5, 3.4), (300.0, 8000.0), (5.0, 500.0)]
AIRY_X0 = [3.0, 2500.0, 100.0]

# 碳化硅（附件 1/2）：与问题二同口径 —— AsLS 去基线拟合振荡项，衬底 n2 固定
SIC_CUTOFF_FALLBACK = 1000.0  # STFT 定界失效时的回退裁剪点（cm⁻¹）
N2_SIC = 2.65                 # SiC 衬底折射率（问题二固定值，选定带宽内近似平稳）
SIC_PREPROC = dict(hampel_win=11, hampel_t=3.0, asls_lam=1e5, asls_p=0.01,
                   sg_win=15, sg_poly=3)

# SiC 双光束 / Airy 参数（与问题二同口径：单振子色散，n2 固定 2.65，θ 均为 [d, B, C]）
#   双光束 = theory_osc（一阶振荡项）；Airy = 完整 Airy 振荡项（含多光束高阶项）
#   两者参数相同、n2 相同，仅差"多光束高阶项"，供 AIC/BIC 公平对比
#   n1²(λ) = 1 + Bλ²/(λ²−C)（Sellmeier 单振子；边界同问题二 sellmeier1）
SIC_NAMES_TWO = ["d(um)", "B", "C"]
SIC_NAMES_AIRY = ["d(um)", "B", "C"]
SIC_SELLMEIER1_BOUNDS = [(0.0, 8.0), (0.0, 1.0)]

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


# Airy 模型（n1 固定常数 + Drude 色散衬底 + 理想平行平面 σ_d=0）：θ = [d, n_inf, wp, gamma]
def airy_R(nu, theta_deg, theta) -> np.ndarray:
    d, n_inf, wp, gamma = theta
    n1 = n_sellmeier1(nu, SI_B, SI_C)
    n2 = n_drude(nu, n_inf, wp, gamma)
    return airy_reflectance_avg(nu, d, 0.0, n1, n2, theta_deg, N0)


# 双光束模型（n1 固定常数 + Drude 色散衬底 + 理想平行平面 σ_d=0）：θ = [d, n_inf, wp, gamma]
def twobeam_R(nu, theta_deg, theta) -> np.ndarray:
    d, n_inf, wp, gamma = theta
    n1 = n_sellmeier1(nu, SI_B, SI_C)
    n2 = n_drude(nu, n_inf, wp, gamma)
    return two_beam_reflectance_avg(nu, d, 0.0, n1, n2, theta_deg, N0)


def make_residual(wn1, R1, wn2, R2, model_func):
    """拼接双角度残差，每角度独立 DC 偏置校正 R_obs ≈ T + b（闭式消元，无增益）。

    硅片反射率是绝对反射率（~30%），DC 水平由已知 n1≈3.42 唯一确定，故只校正
    每角度独立的仪器 DC 偏置 b（两角度实测 DC 相差约 2.7% 的校准偏差），不再引入
    增益 a。增益 a 会与折射率/幅度参数耦合，产生"把近似平坦理论值放大成虚假低
    SSE"的退化解，因此这里显式剔除。
    """
    def resid(theta):
        T1 = model_func(wn1, ANG1, theta)
        b1 = float(np.mean(R1 - T1))
        e1 = T1 + b1 - R1
        T2 = model_func(wn2, ANG2, theta)
        b2 = float(np.mean(R2 - T2))
        e2 = T2 + b2 - R2
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


def rho_drude(nu, n_inf, wp, gamma):
    """Drude 衬底下的界面反射率 ρ = mean|r01·r12|（带内平均，n1 固定为 SI_N1）。"""
    n2 = n_drude(nu, n_inf, wp, gamma)
    r01 = (N0 - SI_N1) / (N0 + SI_N1)
    r12 = (SI_N1 - n2) / (SI_N1 + n2)
    return float(np.mean(np.abs(r01 * r12)))


def solve_silicon(wn1, R1, wn2, R2, d0, seed=SEED) -> dict:
    """对硅片做双光束 + Airy 双模型反演与多光束判定。"""
    n_lin = 2  # 双角度各 1 个 DC 偏置 b，闭式消元
    # 1) 双光束拟合（判定基准 + 残差谐波）
    res_two, _ = fit_model(wn1, R1, wn2, R2, d0, twobeam_R,
                           TWO_BOUNDS, TWO_DE_BOUNDS, seed)
    theta_two = res_two.x
    d_two, n_inf_two, wp_two, gamma_two = theta_two
    n1_ = len(wn1)
    r_two1 = res_two.fun[:n1_]
    stats_two = fit_statistics(res_two, len(theta_two) + n_lin, R1, R2)
    # 残差谐波（用 10° 残差；f0 = 2 n1 d cosθ1 / 1e4，n1 为固定常数）
    th1 = np.arcsin(np.sin(np.deg2rad(ANG1)) / SI_N1)
    f0 = 2.0 * SI_N1 * d_two * np.cos(th1) / 1e4
    harm_two = residual_harmonics(wn1, r_two1, f0)
    rho_two = rho_drude(wn1, n_inf_two, wp_two, gamma_two)

    # 2) Airy 拟合
    res_airy, _ = fit_model(wn1, R1, wn2, R2, d0, airy_R,
                            AIRY_BOUNDS, AIRY_DE_BOUNDS, seed)
    theta_airy = res_airy.x
    d_airy, n_inf_airy, wp_airy, gamma_airy = theta_airy
    r_airy1 = res_airy.fun[:n1_]
    stats_airy = fit_statistics(res_airy, len(theta_airy) + n_lin, R1, R2)
    sd_a, ci_a, _ = parameter_uncertainty(res_airy, AIRY_NAMES, n_lin=n_lin)
    # Airy 主频 f0_airy 与 ρ 用固定 n1
    th1_airy = np.arcsin(np.sin(np.deg2rad(ANG1)) / SI_N1)
    f0_airy = 2.0 * SI_N1 * d_airy * np.cos(th1_airy) / 1e4
    harm_airy = residual_harmonics(wn1, r_airy1, f0_airy)
    rho_airy = rho_drude(wn1, n_inf_airy, wp_airy, gamma_airy)

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
    """对 SiC 做双光束 + Airy 双模型反演与多光束判定（与问题二同口径）。

    SiC 外延层 n1 用 Sellmeier 单振子色散（问题二比选出的最优模型），衬底
    n2 固定 2.65（问题二值）；双光束用一阶振荡项 theory_osc，Airy 用完整
    多光束振荡项，两者参数均为 [d,B,C]、n2 相同，仅差多光束高阶项，供
    AIC/BIC 公平对比。厚度边界按 FFT 初值 d0 收窄到 [d0-0.5, d0+0.5]，
    避免分支跳变（同问题二剖面扫描口径）。
    """
    d_half, step = 0.5, 0.01
    bounds = [(max(0.5, d0 - d_half), d0 + d_half)] + list(SIC_SELLMEIER1_BOUNDS)

    # 1) 双光束（Q2 sellmeier1 口径：theory_osc + 单振子 n1，n2 固定 2.65）
    beta0_two = de_init_beta(wn1, R1, wn2, R2, d0, "sellmeier1", N2_SIC, ANGLES,
                             SIC_SELLMEIER1_BOUNDS, seed=seed)
    fit_two, prof_two = profile_scan_d(wn1, R1, wn2, R2, d0, beta0_two, "sellmeier1",
                                       N2_SIC, ANGLES, bounds, d_half=d_half,
                                       step=step, seed=seed)
    theta_two = fit_two.theta.tolist()
    d_two = float(fit_two.d)
    B2, C2 = [float(v) for v in fit_two.beta]
    stats_two = fit_statistics(fit_two.res, fit_two.res.x.size + fit_two.n_lin, R1, R2)
    r_two1 = fit_two.residual1
    # 主频用带内平均相位折射率近似（色散使 f0 微展宽，仅作谐波定位）
    n1m = float(n_sellmeier1(wn1, B2, C2).mean())
    th1 = np.arcsin(np.sin(np.deg2rad(ANG1)) / n1m)
    f0 = 2.0 * n1m * d_two * np.cos(th1) / 1e4
    harm_two = residual_harmonics(wn1, r_two1, f0)
    rho_two = rho_estimate(n1m, N2_SIC)

    # 2) Airy（Q2 同口径：airy_reflectance_osc + 单振子 n1，仅多光束高阶项不同）
    beta0_airy = de_init_beta(wn1, R1, wn2, R2, d0, "airy_sellmeier1", N2_SIC, ANGLES,
                              SIC_SELLMEIER1_BOUNDS, seed=seed)
    fit_airy, prof_airy = profile_scan_d(wn1, R1, wn2, R2, d0, beta0_airy, "airy_sellmeier1",
                                         N2_SIC, ANGLES, bounds, d_half=d_half,
                                         step=step, seed=seed)
    theta_airy = fit_airy.theta.tolist()
    d_airy = float(fit_airy.d)
    Ba, Ca = [float(v) for v in fit_airy.beta]
    stats_airy = fit_statistics(fit_airy.res, fit_airy.res.x.size + fit_airy.n_lin, R1, R2)
    r_airy1 = fit_airy.residual1
    sd_beta, ci_beta, _ = parameter_uncertainty(fit_airy.res, ["B", "C"],
                                                n_lin=fit_airy.n_lin)
    n1ma = float(n_sellmeier1(wn1, Ba, Ca).mean())
    th1a = np.arcsin(np.sin(np.deg2rad(ANG1)) / n1ma)
    f0_airy = 2.0 * n1ma * d_airy * np.cos(th1a) / 1e4
    harm_airy = residual_harmonics(wn1, r_airy1, f0_airy)
    rho_airy = rho_estimate(n1ma, N2_SIC)

    return {
        "d0": d0,
        "two_beam": {
            "theta": theta_two, "names": SIC_NAMES_TWO,
            "stats": stats_two, "rho": rho_two,
            "harmonics": {str(k): v for k, v in harm_two.items()},
            "d_ci": prof_two["d_ci"], "d_std": prof_two["std_d"],
        },
        "airy": {
            "theta": theta_airy, "names": SIC_NAMES_AIRY,
            "stats": stats_airy, "rho": rho_airy,
            "harmonics": {str(k): v for k, v in harm_airy.items()},
            "d_ci": prof_airy["d_ci"], "d_std": prof_airy["std_d"],
            "sd": [prof_airy["std_d"]] + sd_beta.tolist(),
            "ci": [prof_airy["d_ci"]] + ci_beta.tolist(),
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

    # 4. SiC 双模型反演与判定（附件1/2，与问题二同口径）
    print("\n[4] SiC wafer (f1/f2): two-beam vs Airy inversion & decision")
    spec_sic = {}
    for key, path in [("f1", raw["f1"]), ("f2", raw["f2"])]:
        wn, R = load_spectra(root / path)
        wn_p, R_p, cutoff, _, _ = preprocess(wn, R, **SIC_PREPROC)
        # 裁剪分界点合理性防护（同问题二）：异常时回退默认 1000 cm⁻¹
        if not (700.0 <= cutoff <= 1600.0):
            cutoff = 1000.0
            wn_p, R_p, _, _, _ = preprocess(wn, R, cutoff=cutoff, **SIC_PREPROC)
        spec_sic[key] = (wn_p, R_p)
        print(f"  {key}: {len(wn)} pts -> cutoff {cutoff:.1f} cm-1 -> {len(wn_p)} pts")
        save_csv(result_dir / f"q3_processed_{key}.csv",
                 "wavenumber_cm-1,reflectance_osc_%", [wn_p, R_p])

    wn1s, R1s = spec_sic["f1"]
    wn2s, R2s = spec_sic["f2"]
    # FFT 厚度初值（R1s 已是 AsLS 去基线后的振荡项，同问题二，无需再 detrend）
    d0s = fft_init_d(wn1s, R1s, n_avg=2.6, theta0_deg=ANG1, search_lo=1500.0)
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
