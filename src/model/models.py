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

def n_sellmeier1(nu: np.ndarray, B: float, C: float) -> np.ndarray:
    """Sellmeier 单振子（UV 吸收边）：n²(λ) = 1 + Bλ²/(λ²-C)，λ 单位 μm。

    物理设定：C 为紫外吸收边 λ_res²（λ_res≈0~1 μm），远小于测量带宽
    （λ∈[2.5,6.4] μm），故 λ²-C 恒正、n²≈1+B>1，色散正常且单调。
    """
    lam = wavenumber_to_wavelength_um(nu)
    lam2 = lam ** 2
    denom = np.maximum(lam2 - C, 1e-4)
    return np.sqrt(np.maximum(1.0 + B * lam2 / denom, 1e-8))


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


# 双光束干涉完整反射率（含 DC 项，一阶近似，用于与 Airy 同口径对比）
def two_beam_reflectance(nu, d_um, n1, n2, theta0_deg, n0=1.0):
    """双光束干涉**完整反射率**（%），含 DC 项。

    与 theory_osc 的差别：theory_osc 已减去 DC 项（只留振荡），本函数保留
    DC 项，用于问题三中与 Airy 模型在"完整反射率"口径下公平对比（AIC/BIC）。
    无吸收时 n1、n2 为实数，r01、r12 由菲涅尔公式给出。
    """
    nu = np.asarray(nu, float)
    theta0 = np.deg2rad(theta0_deg)
    theta1 = np.arcsin(np.minimum(n0 * np.sin(theta0) / n1, 1.0))
    delta = 4.0 * np.pi * n1 * d_um * np.cos(theta1) * nu / 1e4
    e = np.exp(1j * delta)

    rs01, rp01 = fresnel_r(n0, n1, theta0)
    rs12, rp12 = fresnel_r(n1, n2, theta1)

    Rs = np.abs(rs01 + (1 - rs01 ** 2) * rs12 * e) ** 2
    Rp = np.abs(rp01 + (1 - rp01 ** 2) * rp12 * e) ** 2
    return (Rs + Rp) / 2.0 * 100.0


def theory_R(nu: np.ndarray, d_um: float, beta: np.ndarray, theta0_deg: float,
             n2: float, model: str, n0: float = 1.0) -> np.ndarray:
    """按指定色散模型计算振荡反射率（%）。

    beta : Cauchy → (A, B, C)；Sellmeier 双振子 → (B1, C1, B2, C2)；
           Sellmeier 单振子 → (B, C)
    """
    beta = np.asarray(beta, float)
    if model == "cauchy":
        n1 = n_cauchy(nu, *beta)
    elif model == "sellmeier":
        n1 = n_sellmeier(nu, *beta)
    elif model == "sellmeier1":
        n1 = n_sellmeier1(nu, *beta)
    else:
        raise ValueError(f"未知模型: {model}")
    return theory_osc(nu, d_um, n1, theta0_deg, n2, n0)


# 多光束干涉：Airy 公式单层薄膜反射率（s/p 偏振平均，支持复折射率）
def airy_reflectance(nu, d_um, n1, n2, theta0_deg, n0=1.0):
    """多光束干涉完整反射率 R = |r|²（%），Airy 公式（单层薄膜精确解）。

    与双光束的"振荡项"不同，本函数返回**完整反射率**（含 DC 项 r01²），
    因此可直接拟合原始反射率。对单层薄膜，Airy 公式
        r = (r01 + r12 e^{2iδ}) / (1 + r01 r12 e^{2iδ})
    就是传输矩阵法（Maxwell 方程精确解）的解析形式，其中 r01、r12 为
    复菲涅尔系数、δ = 2π ν n1 d cosθ1 为单程复相位，故有吸收（复折射率）
    时仍精确，不依赖 Stokes 关系。

    参数
    ----
    nu         : 波数数组（cm⁻¹）
    d_um       : 外延层厚度（μm）
    n1         : 外延层复折射率（标量或数组，随 nu 可变化）
    n2         : 衬底复折射率（标量或数组）
    theta0_deg : 入射角（°）
    n0         : 入射介质折射率（空气，实数）

    返回
    ----
    R : 反射率（%），s/p 偏振平均，取值 [0, 100]
    """
    nu = np.asarray(nu, float)
    n1 = np.asarray(n1, complex)
    n2 = np.asarray(n2, complex)
    theta0 = np.deg2rad(theta0_deg)
    cos0 = np.cos(theta0)
    sin0 = n0 * np.sin(theta0)

    # Snell 定律复推广：n0 sinθ0 = n_j sinθ_j，cosθ_j 取实部非负分支
    sin1 = sin0 / n1
    sin2 = sin0 / n2
    cos1 = np.sqrt(1.0 - sin1 ** 2)
    cos2 = np.sqrt(1.0 - sin2 ** 2)

    # 单程相位：δ = 2π ν n1 d cosθ1，d[cm] = d_um × 1e-4；往返因子 e^{2iδ}
    delta = 2.0 * np.pi * nu * n1 * (d_um * 1e-4) * cos1
    e2 = np.exp(2j * delta)

    def _r_pol(eta0, eta1, eta2):
        """由光学导纳求 Airy 反射系数 r = (r01 + r12 e^{2iδ})/(1 + r01 r12 e^{2iδ})。"""
        r01 = (eta0 - eta1) / (eta0 + eta1)
        r12 = (eta1 - eta2) / (eta1 + eta2)
        return (r01 + r12 * e2) / (1.0 + r01 * r12 * e2)

    # s 偏振导纳 η = n cosθ；p 偏振导纳 η = n / cosθ
    rs = _r_pol(n0 * cos0, n1 * cos1, n2 * cos2)
    rp = _r_pol(n0 / cos0, n1 / cos1, n2 / cos2)

    R = (np.abs(rs) ** 2 + np.abs(rp) ** 2) / 2.0
    return R * 100.0


def airy_reflectance_avg(nu, d_um, sigma_d, n1, n2, theta0_deg, n0=1.0,
                         n_gh=9):
    """厚度系综平均的多光束反射率（%），建模光斑内厚度不均匀。

    光斑内厚度 d 近似服从高斯分布 N(d_um, sigma_d²)（表面粗糙/楔角/外延
    均匀性），实测反射率是不同厚度反射率的系综平均。用 Gauss-Hermite 积分
    对厚度做平均：
        R_avg(ν) = (1/√π) Σ w_k · R(ν; d_um + √2·σ_d·x_k)
    其中 (x_k, w_k) 为 n_gh 点 Gauss-Hermite 节点与权重。

    该平均使干涉条纹对比度随波数近似高斯衰减（远快于吸收的指数衰减）：
        visibility(ν) ≈ exp[ -(4π n1 σ_d cosθ1 ν / 1e4)² / 2 ]
    sigma_d=0 时退化为单厚度 Airy 解。
    """
    nu = np.asarray(nu, float)
    if sigma_d <= 0.0:
        return airy_reflectance(nu, d_um, n1, n2, theta0_deg, n0)
    x, w = np.polynomial.hermite.hermgauss(n_gh)
    R = np.zeros_like(nu)
    for xi, wi in zip(x, w):
        d_i = d_um + np.sqrt(2.0) * sigma_d * xi
        R = R + wi * airy_reflectance(nu, d_i, n1, n2, theta0_deg, n0)
    return R / np.sqrt(np.pi)


def two_beam_reflectance_avg(nu, d_um, sigma_d, n1, n2, theta0_deg, n0=1.0,
                             n_gh=9):
    """双光束完整反射率在厚度高斯分布下的系综平均（%）。

    与 airy_reflectance_avg 同口径：建模光斑内厚度不均匀 σ_d，使双光束与
    Airy 在"是否含多光束高阶项"之外保持相同自由度，用于 AIC/BIC 公平对比。
    """
    nu = np.asarray(nu, float)
    if sigma_d <= 0.0:
        return two_beam_reflectance(nu, d_um, n1, n2, theta0_deg, n0)
    x, w = np.polynomial.hermite.hermgauss(n_gh)
    R = np.zeros_like(nu)
    for xi, wi in zip(x, w):
        d_i = d_um + np.sqrt(2.0) * sigma_d * xi
        R = R + wi * two_beam_reflectance(nu, d_i, n1, n2, theta0_deg, n0)
    return R / np.sqrt(np.pi)
