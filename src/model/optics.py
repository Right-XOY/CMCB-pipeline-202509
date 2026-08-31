from __future__ import annotations

import numpy as np


def wavenumber_to_wavelength_um(nu):
    # λ[μm] = 1e4 / ν[cm⁻¹]
    return 1e4 / np.asarray(nu, float)


def n_cauchy(nu, A, B, C):
    # n(λ) = A + B/λ² + C/λ⁴
    lam = wavenumber_to_wavelength_um(nu)
    return A + B / lam ** 2 + C / lam ** 4


def n_sellmeier(nu, B1, C1, B2, C2):
    # n²(λ) = 1 + B1λ²/(λ²-C1) + B2λ²/(λ²-C2)
    lam = wavenumber_to_wavelength_um(nu)
    t1 = B1 * lam ** 2 / np.maximum(lam ** 2 - C1, 1e-4)
    t2 = B2 * lam ** 2 / np.maximum(lam ** 2 - C2, 1e-4)
    return np.sqrt(np.maximum(1.0 + t1 + t2, 1e-8))


def n_sellmeier1(nu, B, C):
    # n²(λ) = 1 + Bλ²/(λ²-C)
    lam = wavenumber_to_wavelength_um(nu)
    lam2 = lam ** 2
    denom = np.maximum(lam2 - C, 1e-4)
    return np.sqrt(np.maximum(1.0 + B * lam2 / denom, 1e-8))


def n_drude(nu, n_inf, wp, gamma):
    # n²(ν) = n_inf² − wp²/(ν² + iγν)
    nu = np.asarray(nu, float)
    eps = n_inf ** 2 - wp ** 2 / (nu ** 2 + 1j * gamma * nu)
    return np.sqrt(eps)


def fresnel_r(n_inc, n_exit, theta_inc):
    # s/p 振幅反射系数，光学导纳 η_s=n·cosθ、η_p=n/cosθ
    n_inc = np.asarray(n_inc)
    n_exit = np.asarray(n_exit)
    is_cplx = np.iscomplexobj(n_inc) or np.iscomplexobj(n_exit)
    if is_cplx:
        n_inc = n_inc.astype(complex)
        n_exit = n_exit.astype(complex)
        sin_e = n_inc * np.sin(theta_inc) / n_exit
        cos_e = np.sqrt(1.0 - sin_e ** 2)
    else:
        n_inc = n_inc.astype(float)
        n_exit = n_exit.astype(float)
        ratio = np.minimum(n_inc * np.sin(theta_inc) / np.maximum(n_exit, 1e-12), 1.0)
        sin_e = ratio
        cos_e = np.sqrt(1.0 - sin_e ** 2)
    cos_i = np.cos(theta_inc)
    eta_inc_s = n_inc * cos_i
    eta_exit_s = n_exit * cos_e
    eta_inc_p = n_inc / cos_i
    eta_exit_p = n_exit / cos_e
    rs = (eta_inc_s - eta_exit_s) / (eta_inc_s + eta_exit_s)
    rp = (eta_exit_p - eta_inc_p) / (eta_exit_p + eta_inc_p)
    return rs, rp


def theory_osc(nu, d_um, n1, theta0_deg, n2, n0=1.0):
    # 双光束振荡项：R_osc = 2 r01(1-r01²)r12 cos(δ)，δ = 4π n1 d cosθ1 · ν/1e4
    nu = np.asarray(nu, float)
    theta0 = np.deg2rad(theta0_deg)
    theta1 = np.arcsin(np.minimum(n0 * np.sin(theta0) / n1, 1.0))
    delta = 4.0 * np.pi * n1 * d_um * np.cos(theta1) * nu / 1e4
    e = np.exp(1j * delta)
    rs01, rp01 = fresnel_r(n0, n1, theta0)
    rs12, rp12 = fresnel_r(n1, n2, theta1)
    Rs = np.abs(rs01 + (1 - rs01 ** 2) * rs12 * e) ** 2 - (rs01 ** 2 + (1 - rs01 ** 2) ** 2 * rs12 ** 2)
    Rp = np.abs(rp01 + (1 - rp01 ** 2) * rp12 * e) ** 2 - (rp01 ** 2 + (1 - rp01 ** 2) ** 2 * rp12 ** 2)
    return (Rs + Rp) / 2.0 * 100.0


def two_beam_reflectance(nu, d_um, n1, n2, theta0_deg, n0=1.0):
    # 双光束完整反射率（含 DC），用于与 Airy 同口径对比
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


def theory_R(nu, d_um, beta, theta0_deg, n2, model, n0=1.0):
    beta = np.asarray(beta, float)
    if model == "cauchy":
        n1 = n_cauchy(nu, *beta)
    elif model == "airy":
        n1 = n_cauchy(nu, *beta)
        return airy_reflectance_osc(nu, d_um, n1, n2, theta0_deg, n0)
    elif model == "sellmeier":
        n1 = n_sellmeier(nu, *beta)
    elif model == "sellmeier1":
        n1 = n_sellmeier1(nu, *beta)
    elif model == "airy_sellmeier1":
        n1 = n_sellmeier1(nu, *beta)
        return airy_reflectance_osc(nu, d_um, n1, n2, theta0_deg, n0)
    else:
        raise ValueError(f"未知模型: {model}")
    return theory_osc(nu, d_um, n1, theta0_deg, n2, n0)


def airy_reflectance(nu, d_um, n1, n2, theta0_deg, n0=1.0):
    # 多光束 Airy 完整反射率：r = (r01 + r12 e^{2iδ}) / (1 + r01 r12 e^{2iδ})
    nu = np.asarray(nu, float)
    n1 = np.asarray(n1, complex)
    n2 = np.asarray(n2, complex)
    theta0 = np.deg2rad(theta0_deg)
    cos0 = np.cos(theta0)
    sin0 = n0 * np.sin(theta0)
    sin1 = sin0 / n1
    sin2 = sin0 / n2
    cos1 = np.sqrt(1.0 - sin1 ** 2)
    cos2 = np.sqrt(1.0 - sin2 ** 2)
    delta = 2.0 * np.pi * nu * n1 * (d_um * 1e-4) * cos1
    e2 = np.exp(2j * delta)

    def _r_pol(eta0, eta1, eta2):
        r01 = (eta0 - eta1) / (eta0 + eta1)
        r12 = (eta1 - eta2) / (eta1 + eta2)
        return (r01 + r12 * e2) / (1.0 + r01 * r12 * e2)

    rs = _r_pol(n0 * cos0, n1 * cos1, n2 * cos2)
    rp = _r_pol(n0 / cos0, n1 / cos1, n2 / cos2)
    R = (np.abs(rs) ** 2 + np.abs(rp) ** 2) / 2.0
    return R * 100.0


def airy_reflectance_avg(nu, d_um, sigma_d, n1, n2, theta0_deg, n0=1.0, n_gh=9):
    # 光斑内厚度高斯分布系综平均（Gauss-Hermite）
    nu = np.asarray(nu, float)
    if sigma_d <= 0.0:
        return airy_reflectance(nu, d_um, n1, n2, theta0_deg, n0)
    x, w = np.polynomial.hermite.hermgauss(n_gh)
    R = np.zeros_like(nu)
    for xi, wi in zip(x, w):
        d_i = d_um + np.sqrt(2.0) * sigma_d * xi
        R = R + wi * airy_reflectance(nu, d_i, n1, n2, theta0_deg, n0)
    return R / np.sqrt(np.pi)


def airy_reflectance_osc(nu, d_um, n1, n2, theta0_deg, n0=1.0):
    # Airy 振荡项 = 完整反射率 − 解析 DC 项（与 theory_osc 同口径）
    R = airy_reflectance(nu, d_um, n1, n2, theta0_deg, n0)
    theta0 = np.deg2rad(theta0_deg)
    theta1 = np.arcsin(np.minimum(n0 * np.sin(theta0) / np.asarray(n1, float), 1.0))
    rs01, rp01 = fresnel_r(n0, n1, theta0)
    rs12, rp12 = fresnel_r(n1, n2, theta1)
    dc = ((rs01 ** 2 + (1 - rs01 ** 2) ** 2 * rs12 ** 2)
          + (rp01 ** 2 + (1 - rp01 ** 2) ** 2 * rp12 ** 2)) / 2.0
    return R - dc * 100.0


def two_beam_reflectance_avg(nu, d_um, sigma_d, n1, n2, theta0_deg, n0=1.0, n_gh=9):
    nu = np.asarray(nu, float)
    if sigma_d <= 0.0:
        return two_beam_reflectance(nu, d_um, n1, n2, theta0_deg, n0)
    x, w = np.polynomial.hermite.hermgauss(n_gh)
    R = np.zeros_like(nu)
    for xi, wi in zip(x, w):
        d_i = d_um + np.sqrt(2.0) * sigma_d * xi
        R = R + wi * two_beam_reflectance(nu, d_i, n1, n2, theta0_deg, n0)
    return R / np.sqrt(np.pi)
