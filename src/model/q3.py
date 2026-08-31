from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import yaml
from scipy.optimize import differential_evolution, least_squares
from preprocess import (find_cutoff_stft, hampel_filter, load_spectra,
                        resample, sg_smooth, preprocess)
from optics import (airy_reflectance_avg, n_drude, n_sellmeier1,
                    two_beam_reflectance_avg)
from inversion import (de_init_beta, fft_init_d, fit_statistics,
                       parameter_uncertainty, profile_scan_d)

N0 = 1.0
ANG1, ANG2 = 10.0, 15.0
ANGLES = (ANG1, ANG2)

N_SI_AVG = 3.42
SI_CUTOFF = 2000.0
PREPROC = dict(hampel_win=11, hampel_t=3.0, sg_win=15, sg_poly=3)

# 硅晶圆片外延层 n1 固定（无色散硅）：n1²=1+B, B=10.7, C=0 → n1≈3.4205
# 衬底 n2 用 Drude 色散；σ_d=0 取理想平行平面（多光束判定基准）
SI_B = 10.7
SI_C = 0.0
SI_N1 = float(np.sqrt(1.0 + SI_B))

TWO_NAMES = ["d(um)", "n_inf", "wp", "gamma"]
TWO_BOUNDS = [(0.5, 200.0), (2.5, 3.4), (300.0, 8000.0), (5.0, 500.0)]
TWO_DE_BOUNDS = [(2.5, 3.4), (300.0, 8000.0), (5.0, 500.0)]

AIRY_NAMES = ["d(um)", "n_inf", "wp", "gamma"]
AIRY_BOUNDS = [(0.5, 200.0), (2.5, 3.4), (300.0, 8000.0), (5.0, 500.0)]
AIRY_DE_BOUNDS = [(2.5, 3.4), (300.0, 8000.0), (5.0, 500.0)]

SIC_CUTOFF_FALLBACK = 1000.0
N2_SIC = 2.65
SIC_PREPROC = dict(hampel_win=11, hampel_t=3.0, asls_lam=1e5, asls_p=0.01,
                   sg_win=15, sg_poly=3)

SIC_NAMES_TWO = ["d(um)", "B", "C"]
SIC_NAMES_AIRY = ["d(um)", "B", "C"]
SIC_SELLMEIER1_BOUNDS = [(0.0, 8.0), (0.0, 1.0)]

SEED = 2025

def load_config():
    root = Path(__file__).resolve().parents[2]
    with open(root / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return root, cfg

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  [output] {path.name}")

def save_csv(path, header, cols):
    arr = np.column_stack(cols)
    fmt = ",".join(["%.6f"] * arr.shape[1])
    np.savetxt(path, arr, fmt=fmt, header=header, comments="")
    print(f"  [output] {path.name}  ({arr.shape[0]} rows)")

def preprocess_si(wn, R, cutoff):
    # 硅晶圆片：裁剪 → 重采样 → Hampel → SG（保留 DC）
    mask = wn >= cutoff
    wn, R = wn[mask], R[mask]
    wn, R = resample(wn, R)
    R = hampel_filter(R, win=PREPROC["hampel_win"], t=PREPROC["hampel_t"])
    R = sg_smooth(R, win=PREPROC["sg_win"], polyorder=PREPROC["sg_poly"])
    return wn, R

def preprocess_sic(wn, R):
    cutoff = find_cutoff_stft(wn, R)
    if not (700.0 <= cutoff <= 1600.0):
        cutoff = SIC_CUTOFF_FALLBACK
    wn_p, R_p = preprocess_si(wn, R, cutoff)
    return wn_p, R_p, cutoff

def airy_R(nu, theta_deg, theta):
    d, n_inf, wp, gamma = theta
    n1 = n_sellmeier1(nu, SI_B, SI_C)
    n2 = n_drude(nu, n_inf, wp, gamma)
    return airy_reflectance_avg(nu, d, 0.0, n1, n2, theta_deg, N0)

def twobeam_R(nu, theta_deg, theta):
    d, n_inf, wp, gamma = theta
    n1 = n_sellmeier1(nu, SI_B, SI_C)
    n2 = n_drude(nu, n_inf, wp, gamma)
    return two_beam_reflectance_avg(nu, d, 0.0, n1, n2, theta_deg, N0)

def make_residual(wn1, R1, wn2, R2, model_func):
    # 每角度独立 DC 偏置 b（闭式消元，无增益），避免增益与折射率耦合的退化解
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
    # H_k = k·f0 谐波峰幅值 / 峰两侧局部邻带噪声标准差
    dnu = float(np.median(np.diff(wn)))
    F = np.abs(np.fft.rfft(resid * np.hanning(len(resid))))
    freqs = np.fft.rfftfreq(len(resid), d=dnu)
    dfreq = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0
    out = {}
    for k in range(2, nharm + 1):
        target = k * f0
        half = half_peak * dfreq
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
    # ρ = |r01·r12|（垂直入射近似）
    r01 = (N0 - n1) / (N0 + n1)
    r12 = (n1 - n2) / (n1 + n2)
    return float(abs(r01 * r12))

def rho_drude(nu, n_inf, wp, gamma):
    n2 = n_drude(nu, n_inf, wp, gamma)
    r01 = (N0 - SI_N1) / (N0 + SI_N1)
    r12 = (SI_N1 - n2) / (SI_N1 + n2)
    return float(np.mean(np.abs(r01 * r12)))

def solve_silicon(wn1, R1, wn2, R2, d0, seed=SEED):
    n_lin = 2
    res_two, _ = fit_model(wn1, R1, wn2, R2, d0, twobeam_R,
                           TWO_BOUNDS, TWO_DE_BOUNDS, seed)
    theta_two = res_two.x
    d_two, n_inf_two, wp_two, gamma_two = theta_two
    n1_ = len(wn1)
    r_two1 = res_two.fun[:n1_]
    stats_two = fit_statistics(res_two, len(theta_two) + n_lin, R1, R2)
    th1 = np.arcsin(np.sin(np.deg2rad(ANG1)) / SI_N1)
    f0 = 2.0 * SI_N1 * d_two * np.cos(th1) / 1e4
    harm_two = residual_harmonics(wn1, r_two1, f0)
    rho_two = rho_drude(wn1, n_inf_two, wp_two, gamma_two)

    res_airy, _ = fit_model(wn1, R1, wn2, R2, d0, airy_R,
                            AIRY_BOUNDS, AIRY_DE_BOUNDS, seed)
    theta_airy = res_airy.x
    d_airy, n_inf_airy, wp_airy, gamma_airy = theta_airy
    r_airy1 = res_airy.fun[:n1_]
    stats_airy = fit_statistics(res_airy, len(theta_airy) + n_lin, R1, R2)
    sd_a, ci_a, _ = parameter_uncertainty(res_airy, AIRY_NAMES, n_lin=n_lin)
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


def solve_sic(wn1, R1, wn2, R2, d0, seed=SEED):
    # 与问题二同口径：Sellmeier 单振子 n1、n2 固定 2.65，双光束 vs Airy 仅差多光束高阶项
    d_half, step = 0.5, 0.01
    bounds = [(max(0.5, d0 - d_half), d0 + d_half)] + list(SIC_SELLMEIER1_BOUNDS)

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
    n1m = float(n_sellmeier1(wn1, B2, C2).mean())
    th1 = np.arcsin(np.sin(np.deg2rad(ANG1)) / n1m)
    f0 = 2.0 * n1m * d_two * np.cos(th1) / 1e4
    harm_two = residual_harmonics(wn1, r_two1, f0)
    rho_two = rho_estimate(n1m, N2_SIC)

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

def main():
    root, cfg = load_config()
    raw = cfg["data"]["raw_files"]
    result_dir = root / cfg["result"]["dir"]
    result_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Question 3: silicon epi thickness via multi-beam (Airy) model")
    print("=" * 60)

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

    print("\n[2] FFT thickness init")
    d0 = fft_init_d(wn1, R1, n_avg=N_SI_AVG, theta0_deg=ANG1, search_lo=SI_CUTOFF)
    print(f"  d0 = {d0:.3f} um")

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

    print("\n[4] SiC wafer (f1/f2): two-beam vs Airy inversion & decision")
    spec_sic = {}
    for key, path in [("f1", raw["f1"]), ("f2", raw["f2"])]:
        wn, R = load_spectra(root / path)
        wn_p, R_p, cutoff, _, _ = preprocess(wn, R, **SIC_PREPROC)
        if not (700.0 <= cutoff <= 1600.0):
            cutoff = 1000.0
            wn_p, R_p, _, _, _ = preprocess(wn, R, cutoff=cutoff, **SIC_PREPROC)
        spec_sic[key] = (wn_p, R_p)
        print(f"  {key}: {len(wn)} pts -> cutoff {cutoff:.1f} cm-1 -> {len(wn_p)} pts")
        save_csv(result_dir / f"q3_processed_{key}.csv",
                 "wavenumber_cm-1,reflectance_osc_%", [wn_p, R_p])

    wn1s, R1s = spec_sic["f1"]
    wn2s, R2s = spec_sic["f2"]
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
