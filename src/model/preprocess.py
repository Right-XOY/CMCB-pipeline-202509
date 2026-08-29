"""数据预处理模块：读取原始光谱并完成裁剪、重采样、去异常、基线校正、平滑。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage import median_filter
from scipy.sparse.linalg import spsolve


# 数据读取
def load_spectra(path: str) -> tuple[np.ndarray, np.ndarray]:
    """读取附件 xlsx，返回 (波数 cm⁻¹, 反射率 %)。

    附件格式：第 1 列波数(cm⁻¹)，第 2 列反射率(%)，第 0 行为表头。
    """
    df = pd.read_excel(path, header=0)
    wn = df.iloc[:, 0].to_numpy(dtype=float)
    R = df.iloc[:, 1].to_numpy(dtype=float)
    return wn, R


# 低波数异常裁剪（滑动窗口 FFT 定量定界）
def find_cutoff_stft(wn: np.ndarray, R: np.ndarray,
                     window: int = 1200, hop: int = 200, eta: float = 0.3,
                     f0_est=(1500.0, 4000.0), dnu_band: int = 4,
                     detrend_win: int = 1000) -> float:
    """用滑动窗口 FFT 定量确定裁剪分界点。

    原理：干涉条纹在主频 f0（对应光程 2nd·cosθ₁）处有集中能量；
    低波数异常区（如 Reststrahlen 带）低频背景能量占主导。
    判据采用 主频带能量/低频残留能量 比值，取比值首次达到其峰值
    eta 倍的位置作为分界点。窗口长度需覆盖 ≥2 个条纹周期。
    """
    wn = np.asarray(wn, float)
    R = np.asarray(R, float)
    dnu = float(np.median(np.diff(wn)))

    # 1) 去趋势（移动平均去除慢变背景，保留条纹）
    bg = np.convolve(R, np.ones(detrend_win) / detrend_win, mode="same")
    Rd = R - bg

    # 2) 在高波数段（去趋势后）估计干涉主频 f0
    sel = (wn >= f0_est[0]) & (wn <= f0_est[1])
    if np.sum(sel) < 8:
        return float(wn[0])
    sig = Rd[sel] * np.hanning(np.sum(sel))
    F = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(len(sig), d=dnu)
    f0 = float(freqs[1 + int(np.argmax(np.abs(F[1:])))])  # 跳过 DC 项

    # 3) 滑动窗口：主频带能量 / 低频残留能量
    n = len(wn)
    fw = np.fft.rfftfreq(window, d=dnu)
    f0_idx = int(np.argmin(np.abs(fw - f0)))
    band = slice(max(1, f0_idx - dnu_band), min(len(fw), f0_idx + dnu_band + 1))
    low = slice(1, max(2, f0_idx // 2))  # f < f0/2 视为低频残留

    centers, ratios = [], []
    for start in range(0, n - window + 1, hop):
        wsig = Rd[start:start + window]
        wsig = (wsig - np.mean(wsig)) * np.hanning(window)
        Fw = np.abs(np.fft.rfft(wsig)) ** 2
        Ef = float(np.sum(Fw[band]))
        El = float(np.sum(Fw[low]))
        ratios.append(Ef / max(El, 1e-12))
        centers.append(float(wn[start + window // 2]))

    ratios = np.asarray(ratios)
    peak = ratios.max()
    if peak <= 0:
        return float(wn[0])
    ok = np.where(ratios >= eta * peak)[0]
    return float(centers[ok[0]]) if len(ok) else float(wn[0])


# 等间距重采样（线性插值，保证 FFT 有效）
def resample(wn: np.ndarray, R: np.ndarray, step: float | None = None
             ) -> tuple[np.ndarray, np.ndarray]:
    """在 [wn.min(), wn.max()] 上按固定步长线性插值重采样。"""
    wn = np.asarray(wn, float)
    R = np.asarray(R, float)
    if step is None:
        step = float(np.median(np.diff(wn)))
    wn_new = np.arange(wn.min(), wn.max() + step, step)
    R_new = np.interp(wn_new, wn, R)
    return wn_new, R_new


# Hampel 滤波：去孤立异常点
def hampel_filter(R: np.ndarray, win: int = 11, t: float = 3.0) -> np.ndarray:
    """滑动窗口中位数滤波变体：偏离中位数超过 t·σ_MAD 的点用中位数替换。"""
    R = np.asarray(R, float)
    med = median_filter(R, size=win, mode="reflect")
    resid = R - med
    mad = median_filter(np.abs(resid), size=win, mode="reflect")
    sigma = 1.4826 * mad
    sigma[sigma < 1e-12] = 1e-12
    mask = np.abs(resid) > t * sigma
    out = R.copy()
    out[mask] = med[mask]
    return out


# AsLS 非对称最小二乘基线校正
def asls_baseline(R: np.ndarray, lam: float = 1e5, p: float = 0.01,
                  niter: int = 10) -> np.ndarray:
    """非对称最小二乘拟合法提取基线（贴附谱线下包络），返回基线 z。"""
    R = np.asarray(R, float)
    L = len(R)
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(L - 2, L))
    DtD = (D.T @ D).tocsc()
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.diags(w, 0, format="csc")
        Z = W + lam * DtD
        z = spsolve(Z, w * R)
        w = np.where(R > z, p, 1.0 - p)
    return z


# Savitzky-Golay 平滑
def sg_smooth(R: np.ndarray, win: int = 15, polyorder: int = 3) -> np.ndarray:
    """多项式最小二乘卷积平滑，保留峰形与振荡细节。"""
    from scipy.signal import savgol_filter
    return savgol_filter(np.asarray(R, float), window_length=win, polyorder=polyorder)


# 预处理总流程
def preprocess(wn: np.ndarray, R: np.ndarray, cutoff: float | None = None,
               step: float | None = None,
               hampel_win: int = 11, hampel_t: float = 3.0,
               asls_lam: float = 1e5, asls_p: float = 0.01,
               sg_win: int = 15, sg_poly: int = 3,
               ) -> tuple[np.ndarray, np.ndarray, float]:
    """完整预处理流水线：裁剪 → 重采样 → Hampel → AsLS 去基线 → SG 平滑。

    返回 (重采样波数, 去基线后的振荡反射率, 裁剪分界点)。
    去基线后的谱线围绕 0 振荡，与理论模型振荡项口径一致。
    """
    wn = np.asarray(wn, float)
    R = np.asarray(R, float)

    if cutoff is None:
        cutoff = find_cutoff_stft(wn, R)

    mask = wn >= cutoff
    wn, R = wn[mask], R[mask]

    wn, R = resample(wn, R, step)

    R = hampel_filter(R, win=hampel_win, t=hampel_t)
    baseline = asls_baseline(R, lam=asls_lam, p=asls_p)
    R = R - baseline
    R = sg_smooth(R, win=sg_win, polyorder=sg_poly)
    R = R - np.mean(R)  # 去 DC 偏移，保证围绕 0 振荡（与理论振荡项口径一致）

    return wn, R, float(cutoff)
