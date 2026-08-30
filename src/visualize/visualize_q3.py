"""问题三生图脚本：多光束判定 + 硅片/SiC 反演的全部数据驱动图。

运行方式（在项目根目录）：
    python src/visualize/visualize_q3.py

数据来源：
  - 原始光谱：data/raw/附件3.xlsx、附件4.xlsx（复用 preprocess.load_spectra）
  - 预处理结果：outputs/result/q3_processed_f3/f4.csv（硅片，保留 DC）、
                 outputs/result/q3_processed_f1/f2.csv（SiC，振荡项）
  - 判定/拟合结果：outputs/result/q3_silicon.json、q3_sic.json

输出：outputs/figures/q3/
  fig01_raw_f3/f4                原始干涉光谱（硅片）
  fig02_cropped_f3/f4            裁剪后光谱（标注 2000 cm-1 分界点）
  fig03_fit_silicon              硅片 双光束 vs Airy 拟合对比
  fig04_residuals_silicon        硅片 双光束 vs Airy 残差
  fig05_residual_fft_silicon     硅片 双光束残差频谱（谐波检验）
  fig06_fit_sic                  SiC  双光束 vs Airy 拟合对比（振荡口径）
  fig07_residual_fft_sic         SiC  双光束残差频谱（谐波检验）
  fig08_sim_bias                 仿真：双光束反演厚度偏差随衬底吸收变化
  fig09_aic_compare              双光束 vs Airy 的 AIC/BIC 对比
  fig10_thickness                双光束 vs Airy 厚度修正对比
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "model"))
from preprocess import load_spectra, resample  # noqa: E402
from models import (airy_reflectance_avg, airy_reflectance_osc, n_cauchy,  # noqa: E402
                    theory_osc, two_beam_reflectance_avg)

# matplotlib 全局样式：与 q2 生图脚本保持一致（PNG + 彩色 + 中文）
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "PingFang SC"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.35,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
C_F3 = "#d62728"     # 附件3（10°）红
C_F4 = "#1f77b4"     # 附件4（15°）蓝
C_TWO = "#7f7f7f"    # 双光束 灰
C_AIRY = "#2ca02c"   # Airy 绿

# 与 q3.py 保持一致的物理常量
N0 = 1.0
N1_SI = 3.42         # 硅外延层折射率（固定）
N2_SIC = 2.65        # SiC 衬底折射率（固定）
SI_CUTOFF = 2000.0   # 硅片预处理裁剪点（cm-1）
ANG1, ANG2 = 10.0, 15.0


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------- 模型重建
def _lin_fit(T: np.ndarray, R: np.ndarray) -> tuple[float, float]:
    """复现 q3.py 的每角度线性校正 R ≈ a·T + b（闭式最小二乘）。"""
    A = np.column_stack([T, np.ones_like(T)])
    sol, *_ = np.linalg.lstsq(A, R, rcond=None)
    return float(sol[0]), float(sol[1])


def si_two_beam(nu, ang, theta):
    """硅片双光束完整反射率（含 σ_d 厚度平均）。theta=[d, n2, sigma_d]。"""
    d, n2, sd = theta
    return two_beam_reflectance_avg(nu, d, sd, N1_SI, n2, ang, N0)


def si_airy(nu, ang, theta):
    """硅片 Airy 完整反射率（复衬底 + σ_d 厚度平均）。theta=[d, n2r, k2, sigma_d]。"""
    d, n2r, k2, sd = theta
    n2 = n2r + 1j * k2
    return airy_reflectance_avg(nu, d, sd, N1_SI, n2, ang, N0)


def sic_two_beam(nu, ang, theta):
    """SiC 双光束振荡项（Cauchy n1，n2 固定 2.65）。theta=[d, A, B, C]。"""
    d, A, B, C = theta
    n1 = n_cauchy(nu, A, B, C)
    return theory_osc(nu, d, n1, ang, N2_SIC, N0)


def sic_airy(nu, ang, theta):
    """SiC Airy 多光束振荡项（Cauchy n1，n2 固定 2.65）。theta=[d, A, B, C]。"""
    d, A, B, C = theta
    n1 = n_cauchy(nu, A, B, C)
    return airy_reflectance_osc(nu, d, n1, N2_SIC, ang, N0)


def _fit_curve(wn, R, theta, model_func, ang):
    """重建某角度拟合曲线 R_fit = a·T + b，返回 (R_fit, a, b)。"""
    T = model_func(wn, ang, theta)
    a, b = _lin_fit(T, R)
    return a * T + b, a, b


# ---------------------------------------------------------------- 图
def fig01_raw(out_dir: Path, wn3, R3, wn4, R4) -> None:
    """原始干涉光谱（未预处理），每个附件一张图。"""
    for tag, wn, R, color, label in [
            ("f3", wn3, R3, C_F3, "附件3（10°）"),
            ("f4", wn4, R4, C_F4, "附件4（15°）")]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(wn, R, color=color, lw=0.8, label=label)
        ax.set_xlabel("波数（cm-1）")
        ax.set_ylabel("反射率（%）")
        ax.set_title("原始干涉光谱：" + label)
        ax.legend(loc="best")
        fig.savefig(out_dir / f"q3_fig01_raw_{tag}.png")
        plt.close(fig)
        print(f"save -> {out_dir.name}/q3_fig01_raw_{tag}.png")


def fig02_cropped(out_dir: Path, wn3, R3, wn4, R4) -> None:
    """硅片裁剪并等间距重采样后的光谱（标注 2000 cm-1 分界点）。"""
    for tag, wn, R, color, label in [
            ("f3", wn3, R3, C_F3, "附件3（10°）"),
            ("f4", wn4, R4, C_F4, "附件4（15°）")]:
        mask = wn >= SI_CUTOFF
        wnc, Rc = resample(wn[mask], R[mask])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(wnc, Rc, color=color, lw=0.9,
                label=f"{label}（裁剪点 {SI_CUTOFF:.0f} cm-1）")
        ax.axvline(SI_CUTOFF, color=color, ls="--", lw=0.9, alpha=0.55)
        ax.set_xlabel("波数（cm-1）")
        ax.set_ylabel("反射率（%）")
        ax.set_title("裁剪并等间距重采样后的光谱：" + label)
        ax.legend(loc="best")
        fig.savefig(out_dir / f"q3_fig02_cropped_{tag}.png")
        plt.close(fig)
        print(f"save -> {out_dir.name}/q3_fig02_cropped_{tag}.png")


def fig03_fit_silicon(out_dir: Path, p3, p4, si: dict) -> None:
    """硅片：双光束 vs Airy 拟合对比（完整反射率口径，两角度）。"""
    theta_two = si["two_beam"]["theta"]
    theta_airy = si["airy"]["theta"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True)
    for ax, p, color, angle in [
            (axes[0], p3, C_F3, ANG1), (axes[1], p4, C_F4, ANG2)]:
        wn, R = p[:, 0], p[:, 1]
        fit_two, _, _ = _fit_curve(wn, R, theta_two, si_two_beam, angle)
        fit_airy, _, _ = _fit_curve(wn, R, theta_airy, si_airy, angle)
        ax.plot(wn, R, color=color, lw=0.7, alpha=0.8, label="实测")
        ax.plot(wn, fit_two, color=C_TWO, lw=1.0, ls="--", label="双光束")
        ax.plot(wn, fit_airy, color=C_AIRY, lw=1.0, label="Airy")
        ax.set_ylabel("反射率（%）")
        ax.set_title(f"硅片 入射角 {angle:.0f}°：双光束 vs Airy 拟合")
        ax.legend(loc="lower right", ncol=3)
    axes[1].set_xlabel("波数（cm-1）")
    fig.tight_layout()
    fig.savefig(out_dir / "q3_fig03_fit_silicon.png")
    plt.close(fig)
    print(f"save -> {out_dir.name}/q3_fig03_fit_silicon.png")


def fig04_residuals_silicon(out_dir: Path, p3, p4, si: dict) -> None:
    """硅片：双光束 vs Airy 残差（按入射角拆成两张，上下子图为两种模型）。"""
    theta_two = si["two_beam"]["theta"]
    theta_airy = si["airy"]["theta"]
    for tag, p, angle, color in [
            ("f3", p3, ANG1, C_F3), ("f4", p4, ANG2, C_F4)]:
        wn, R = p[:, 0], p[:, 1]
        fit_two, _, _ = _fit_curve(wn, R, theta_two, si_two_beam, angle)
        fit_airy, _, _ = _fit_curve(wn, R, theta_airy, si_airy, angle)
        res_two = R - fit_two
        res_airy = R - fit_airy
        rmse_two = float(np.sqrt(np.mean(res_two ** 2)))
        rmse_airy = float(np.sqrt(np.mean(res_airy ** 2)))
        fig, axes = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True)
        axes[0].plot(wn, res_two, color=C_TWO, lw=0.7,
                     label=f"双光束（RMSE={rmse_two:.4f}）")
        axes[0].axhline(0, color="k", lw=0.8, ls="--")
        axes[0].set_ylabel("残差（%）")
        axes[0].set_title(f"硅片 入射角 {angle:.0f}°：双光束残差")
        axes[0].legend(loc="upper right")
        axes[1].plot(wn, res_airy, color=C_AIRY, lw=0.7,
                     label=f"Airy（RMSE={rmse_airy:.4f}）")
        axes[1].axhline(0, color="k", lw=0.8, ls="--")
        axes[1].set_ylabel("残差（%）")
        axes[1].set_title(f"硅片 入射角 {angle:.0f}°：Airy 残差")
        axes[1].legend(loc="upper right")
        axes[1].set_xlabel("波数（cm-1）")
        fig.tight_layout()
        fig.savefig(out_dir / f"q3_fig04_residuals_silicon_{tag}.png")
        plt.close(fig)
        print(f"save -> {out_dir.name}/q3_fig04_residuals_silicon_{tag}.png")


def _fig_residual_fft(out_dir: Path, p, theta_two, model_func, f0: float,
                      harm: dict, tag: str, title: str) -> None:
    """双光束残差频谱（谐波检验）：标注 f0/2f0/3f0 位置。"""
    wn, R = p[:, 0], p[:, 1]
    fit_two, _, _ = _fit_curve(wn, R, theta_two, model_func, ANG1)
    resid = R - fit_two
    dnu = float(np.median(np.diff(wn)))
    F = np.abs(np.fft.rfft(resid * np.hanning(len(resid)))) ** 2
    f = np.fft.rfftfreq(len(resid), d=dnu)
    h2 = harm.get("2", 0)
    h3 = harm.get("3", 0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(f, F, color=C_TWO, lw=0.9, label="双光束残差 |FFT|²")
    for k, name in [(1, "f0"), (2, "2f0"), (3, "3f0")]:
        ax.axvline(k * f0, color="gray", ls=":", lw=0.9, alpha=0.7)
        ax.text(k * f0, float(F.max()) * 0.92, name, rotation=90, fontsize=8,
                va="top", ha="right", color="gray")
    ax.set_xlim(0.0, 3.2 * f0)
    ax.set_xlabel("频率 f（周期/cm）")
    ax.set_ylabel("|FFT|²（a.u.）")
    ax.set_title(f"{title}（H2={h2:.2f}, H3={h3:.2f}）")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}.png")
    plt.close(fig)
    print(f"save -> {out_dir.name}/{tag}.png")


def fig05_residual_fft_silicon(out_dir: Path, p3, si: dict) -> None:
    """硅片双光束残差频谱（谐波检验）。"""
    _fig_residual_fft(out_dir, p3, si["two_beam"]["theta"], si_two_beam,
                      si["f0"], si["two_beam"]["harmonics"],
                      "q3_fig05_residual_fft_silicon", "硅片 双光束残差频谱")


def fig06_fit_sic(out_dir: Path, p1, p2, sic: dict) -> None:
    """SiC：双光束 vs Airy 拟合对比（振荡口径，两角度）。"""
    theta_two = sic["two_beam"]["theta"]
    theta_airy = sic["airy"]["theta"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True)
    for ax, p, color, angle in [
            (axes[0], p1, C_F3, ANG1), (axes[1], p2, C_F4, ANG2)]:
        wn, R = p[:, 0], p[:, 1]
        fit_two, _, _ = _fit_curve(wn, R, theta_two, sic_two_beam, angle)
        fit_airy, _, _ = _fit_curve(wn, R, theta_airy, sic_airy, angle)
        ax.plot(wn, R, color=color, lw=0.7, alpha=0.8, label="实测（振荡）")
        ax.plot(wn, fit_two, color=C_TWO, lw=1.0, ls="--", label="双光束")
        ax.plot(wn, fit_airy, color=C_AIRY, lw=1.0, label="Airy")
        ax.set_ylabel("反射率振荡（%）")
        ax.set_title(f"SiC 入射角 {angle:.0f}°：双光束 vs Airy 拟合（振荡口径）")
        ax.legend(loc="lower right", ncol=3)
    axes[1].set_xlabel("波数（cm-1）")
    fig.tight_layout()
    fig.savefig(out_dir / "q3_fig06_fit_sic.png")
    plt.close(fig)
    print(f"save -> {out_dir.name}/q3_fig06_fit_sic.png")


def fig07_residual_fft_sic(out_dir: Path, p1, sic: dict) -> None:
    """SiC 双光束残差频谱（谐波检验）。"""
    _fig_residual_fft(out_dir, p1, sic["two_beam"]["theta"], sic_two_beam,
                      sic["f0"], sic["two_beam"]["harmonics"],
                      "q3_fig07_residual_fft_sic", "SiC 双光束残差频谱")


def fig08_sim_bias(out_dir: Path) -> None:
    """仿真：双光束反演厚度偏差随衬底吸收 κ2 的变化（推导过程 §3.5，d=4 μm）。"""
    kappa = np.array([0.0, 0.05, 0.2, 0.5])
    bias = np.array([-0.091, -13.584, -49.119, -90.427])
    std = np.array([0.063, 0.060, 0.093, 0.022])
    kappa_w = np.array([0.000, 0.005, 0.010, 0.020])
    bias_w = np.array([-0.063, -1.389, -2.761, -5.504])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.errorbar(kappa, bias, yerr=std, fmt="-o", color=C_AIRY, capsize=4)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("衬底吸收 κ2")
    ax.set_ylabel("双光束反演厚度偏差（nm）")
    ax.set_title("系统偏差随衬底吸收变化（d=4 μm）")

    ax = axes[1]
    ax.plot(kappa_w, bias_w, "-o", color=C_TWO)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("衬底吸收 κ2（弱吸收放大）")
    ax.set_ylabel("厚度偏差（nm）")
    ax.set_title("弱吸收区偏差（d=4 μm）")

    fig.tight_layout()
    fig.savefig(out_dir / "q3_fig08_sim_bias.png")
    plt.close(fig)
    print(f"save -> {out_dir.name}/q3_fig08_sim_bias.png")


def fig09_aic_compare(out_dir: Path, si: dict, sic: dict) -> None:
    """双光束 vs Airy 的 AIC 对比（ΔAIC>10 判定多光束）。"""
    labels = ["硅片(附件3/4)", "SiC(附件1/2)"]
    two_aic = [si["two_beam"]["stats"]["AIC"], sic["two_beam"]["stats"]["AIC"]]
    airy_aic = [si["airy"]["stats"]["AIC"], sic["airy"]["stats"]["AIC"]]
    delta = [si["delta_aic"], sic["delta_aic"]]
    x = np.arange(2)
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - w / 2, two_aic, w, color=C_TWO, label="双光束")
    ax.bar(x + w / 2, airy_aic, w, color=C_AIRY, label="Airy")
    for xi in x:
        lo = min(two_aic[xi], airy_aic[xi])
        ax.text(xi, lo, f"ΔAIC={delta[xi]:.0f}", ha="center", va="top", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("AIC")
    ax.set_title("双光束 vs Airy 的 AIC 对比（ΔAIC>10 判定多光束）")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "q3_fig09_aic_compare.png")
    plt.close(fig)
    print(f"save -> {out_dir.name}/q3_fig09_aic_compare.png")


def fig10_thickness(out_dir: Path, si: dict, sic: dict) -> None:
    """双光束 vs Airy 的厚度对比与修正量。"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, d, label in [(axes[0], si, "硅片（附件3/4）"),
                         (axes[1], sic, "SiC（附件1/2）")]:
        d2 = d["two_beam"]["theta"][0]
        da = d["airy"]["theta"][0]
        dd = da - d2
        names = ["双光束", "Airy"]
        vals = [d2, da]
        bars = ax.bar(names, vals, color=[C_TWO, C_AIRY], width=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("厚度 d（μm）")
        ax.set_title(f"{label}\n修正量 Δd={dd:+.4f} μm")
    fig.tight_layout()
    fig.savefig(out_dir / "q3_fig10_thickness.png")
    plt.close(fig)
    print(f"save -> {out_dir.name}/q3_fig10_thickness.png")


def main() -> None:
    cfg = load_config()
    raw = cfg["data"]["raw_files"]
    result_dir = ROOT / cfg["result"]["dir"]
    out_dir = ROOT / cfg["visualize"]["output"] / "q3"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 56)
    print("Q3 生图：outputs/figures/q3/（多光束判定 + 硅片/SiC 反演）")
    print("=" * 56)

    # 原始光谱
    wn3, R3 = load_spectra(ROOT / raw["f3"])
    wn4, R4 = load_spectra(ROOT / raw["f4"])
    print(f"  附件3：{len(wn3)} 点，波数 {wn3.min():.2f}-{wn3.max():.2f} cm-1")
    print(f"  附件4：{len(wn4)} 点，波数 {wn4.min():.2f}-{wn4.max():.2f} cm-1")

    # 预处理后光谱（q3.py 输出）
    p3 = np.loadtxt(result_dir / "q3_processed_f3.csv", delimiter=",", skiprows=1)
    p4 = np.loadtxt(result_dir / "q3_processed_f4.csv", delimiter=",", skiprows=1)
    p1 = np.loadtxt(result_dir / "q3_processed_f1.csv", delimiter=",", skiprows=1)
    p2 = np.loadtxt(result_dir / "q3_processed_f2.csv", delimiter=",", skiprows=1)

    # 判定/拟合结果
    with open(result_dir / "q3_silicon.json", encoding="utf-8") as f:
        si = json.load(f)
    with open(result_dir / "q3_sic.json", encoding="utf-8") as f:
        sic = json.load(f)

    fig01_raw(out_dir, wn3, R3, wn4, R4)
    fig02_cropped(out_dir, wn3, R3, wn4, R4)
    fig03_fit_silicon(out_dir, p3, p4, si)
    fig04_residuals_silicon(out_dir, p3, p4, si)
    fig05_residual_fft_silicon(out_dir, p3, si)
    fig06_fit_sic(out_dir, p1, p2, sic)
    fig07_residual_fft_sic(out_dir, p1, sic)
    fig08_sim_bias(out_dir)
    fig09_aic_compare(out_dir, si, sic)
    fig10_thickness(out_dir, si, sic)

    print("完成。12 张图已输出到 outputs/figures/q3/。")


if __name__ == "__main__":
    main()