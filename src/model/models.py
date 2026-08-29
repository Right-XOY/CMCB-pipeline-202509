"""物理模型模块：折射率色散、菲涅尔系数、双光束干涉理论反射率（振荡项）。"""
from __future__ import annotations

import numpy as np


# 波长/波数换算
def wavenumber_to_wavelength_um(nu: np.ndarray) -> np.ndarray:
    """波数(cm⁻¹) → 波长(μm)：λ[μm] = 1e4 / ν[cm⁻¹]。"""
    return 1e4 / np.asarray(nu, float)


# 折射率色散模型
def n_cauchy(nu: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
    """Cauchy 色散：n(λ) = A + B/λ² + C/λ⁴，λ 单位 μm。"""
    lam = wavenumber_to_wavelength_um(nu)
    return A + B / lam ** 2 + C / lam ** 4


def n_sellmeier(nu: np.ndarray, B1: float, C1: float, B2: float, C2: float) -> np.ndarray:
    """Sellmeier 双振子：n²(λ) = 1 + B1λ²/(λ²-C1) + B2λ²/(λ²-C2)，λ 单位 μm。

    数值保护：分母 clamp 到 1e-4，避免接近谐振点 λ²≈C 时除零；
    根号内 clamp 非负，防止折射率平方出现负值。
    """
    lam = wavenumber_to_wavelength_um(nu)
    t1 = B1 * lam ** 2 / np.maximum(lam ** 2 - C1, 1e-4)
    t2 = B2 * lam ** 2 / np.maximum(lam ** 2 - C2, 1e-4)
    return np.sqrt(np.maximum(1.0 + t1 + t2, 1e-8))


# 菲涅尔振幅反射系数
def fresnel_r(n_inc, n_exit, theta_inc: float):
    """两介质界面的振幅反射系数（s/p 偏振）。

    参数
    ----
    n_inc / n_exit : 入射侧 / 出射侧折射率（可为数组）
    theta_inc      : 入射角（弧度，标量）

    返回 (rs, rp)。
    """
    n_inc = np.asarray(n_inc, float)
    n_exit = np.asarray(n_exit, float)
    ratio = np.minimum(n_inc * np.sin(theta_inc) / np.maximum(n_exit, 1e-12), 1.0)
    theta_exit = np.arcsin(ratio)
    cos_i = np.cos(theta_inc)
    cos_e = np.cos(theta_exit)
    rs = (n_inc * cos_i - n_exit * cos_e) / (n_inc * cos_i + n_exit * cos_e)
    rp = (n_exit * cos_i - n_inc * cos_e) / (n_exit * cos_i + n_inc * cos_e)
    return rs, rp


# 双光束干涉理论反射率（振荡项）
def theory_osc(nu: np.ndarray, d_um: float, n1: np.ndarray, theta0_deg: float,
               n2: float, n0: float = 1.0) -> np.ndarray:
    """双光束干涉反射率的振荡部分（单位 %）。

    口径说明：预处理后的实测谱已用 AsLS 去除慢变基线（DC 项），
    因此理论模型也只保留干涉振荡项：
        R_osc = [2 r01 (1-r01²) r12 cos(δ)]  （s/p 平均后）
    其中 δ = 4π n1 d cosθ1 · ν/1e4（d 单位 μm，ν 单位 cm⁻¹）。

    半波损失已自动包含在 r01、r12 的符号中（复数叠加取模）。
    """
    nu = np.asarray(nu, float)
    theta0 = np.deg2rad(theta0_deg)
    theta1 = np.arcsin(np.minimum(n0 * np.sin(theta0) / n1, 1.0))

    # 相位差：d[μm]×1e-4→cm，δ = 4π n1 d cosθ1 / λ[cm] = 4π n1 d cosθ1 ν/1e4
    delta = 4.0 * np.pi * n1 * d_um * np.cos(theta1) * nu / 1e4
    e = np.exp(1j * delta)

    # 上表面（空气 n0 → 外延层 n1）与下表面（外延层 n1 → 衬底 n2）
    rs01, rp01 = fresnel_r(n0, n1, theta0)
    rs12, rp12 = fresnel_r(n1, n2, theta1)

    # 全反射率 = |r01 + (1-r01²) r12 e^{iδ}|²，振荡项 = 全反射率 - DC 项
    Rs = np.abs(rs01 + (1 - rs01 ** 2) * rs12 * e) ** 2 - (rs01 ** 2 + (1 - rs01 ** 2) ** 2 * rs12 ** 2)
    Rp = np.abs(rp01 + (1 - rp01 ** 2) * rp12 * e) ** 2 - (rp01 ** 2 + (1 - rp01 ** 2) ** 2 * rp12 ** 2)

    Rosc = (Rs + Rp) / 2.0
    return Rosc * 100.0


def theory_R(nu: np.ndarray, d_um: float, beta: np.ndarray, theta0_deg: float,
             n2: float, model: str, n0: float = 1.0) -> np.ndarray:
    """按指定色散模型计算振荡反射率（%）。

    beta : Cauchy → (A, B, C)；Sellmeier → (B1, C1, B2, C2)
    """
    beta = np.asarray(beta, float)
    if model == "cauchy":
        n1 = n_cauchy(nu, *beta)
    elif model == "sellmeier":
        n1 = n_sellmeier(nu, *beta)
    else:
        raise ValueError(f"未知模型: {model}")
    return theory_osc(nu, d_um, n1, theta0_deg, n2, n0)
