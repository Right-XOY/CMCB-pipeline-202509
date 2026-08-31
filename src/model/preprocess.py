from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage import median_filter
from scipy.sparse.linalg import spsolve


def load_spectra(path):
    # 附件格式：第1列波数(cm⁻¹)、第2列反射率(%)
    df = pd.read_excel(path, header=0)
    wn = df.iloc[:, 0].to_numpy(dtype=float)
    R = df.iloc[:, 1].to_numpy(dtype=float)
    return wn, R


def find_cutoff_stft(wn, R, window=1200, hop=200, eta=0.3,
                     f0_est=(1500.0, 4000.0), dnu_band=4, detrend_win=1000):
    # 滑动窗口 FFT：主频带能量/低频残留能量 达峰值 eta 倍处为裁剪分界点
    wn = np.asarray(wn, float)
    R = np.asarray(R, float)
    dnu = float(np.median(np.diff(wn)))

    bg = np.convolve(R, np.ones(detrend_win) / detrend_win, mode="same")
    Rd = R - bg

    sel = (wn >= f0_est[0]) & (wn <= f0_est[1])
    if np.sum(sel) < 8:
        return float(wn[0])
    sig = Rd[sel] * np.hanning(np.sum(sel))
    F = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(len(sig), d=dnu)
    f0 = float(freqs[1 + int(np.argmax(np.abs(F[1:])))])

    n = len(wn)
    fw = np.fft.rfftfreq(window, d=dnu)
    f0_idx = int(np.argmin(np.abs(fw - f0)))
    band = slice(max(1, f0_idx - dnu_band), min(len(fw), f0_idx + dnu_band + 1))
    low = slice(1, max(2, f0_idx // 2))

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


def resample(wn, R, step=None):
    wn = np.asarray(wn, float)
    R = np.asarray(R, float)
    if step is None:
        step = float(np.median(np.diff(wn)))
    wn_new = np.arange(wn.min(), wn.max() + step, step)
    R_new = np.interp(wn_new, wn, R)
    return wn_new, R_new


def hampel_filter(R, win=11, t=3.0):
    # 偏离滑动中位数超过 t·σ_MAD 的点用中位数替换
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


def asls_baseline(R, lam=1e5, p=0.01, niter=10):
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


def sg_smooth(R, win=15, polyorder=3):
    from scipy.signal import savgol_filter
    return savgol_filter(np.asarray(R, float), window_length=win, polyorder=polyorder)


def preprocess(wn, R, cutoff=None, step=None,
               hampel_win=11, hampel_t=3.0,
               asls_lam=1e5, asls_p=0.01,
               sg_win=15, sg_poly=3):
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
    dc = float(np.mean(R))
    R = R - dc

    return wn, R, float(cutoff), baseline, dc
