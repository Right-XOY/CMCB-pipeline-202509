# -*- coding: utf-8 -*-
"""问题二生图脚本：生成论文所需的 7 张数据驱动图（PNG，彩色，中文标签）。

运行方式（在项目根目录）：
    python src/visualize/visualize_q2.py

数据来源：
  - 原始光谱：data/raw/附件1.xlsx、附件2.xlsx（复用 src/model/preprocess.load_spectra）
  - 预处理结果：outputs/result/q2_processed_f1/f2.csv（主代码 q2.py 输出）
  - 拟合结果：outputs/result/q2_fit_cauchy/sellmeier.csv
  - 可靠性结果：outputs/result/q2_results.json、q2_noise_test.json

输出：outputs/figures/q2/q2_fig01_raw_f1/f2.png、q2_fig02_cropped_f1/f2.png、
      q2_fig03_prepost_f1/f2.png、q2_fig04_fit_cauchy.png、q2_fig05_fit_sellmeier.png、
      q2_fig06_residuals_cauchy/sellmeier.png、q2_fig07_reliability.png、
      q2_fig08_cross_check.png、q2_fig09_residual_fft_cauchy/sellmeier.png、
      q2_fig10_dispersion.png、q2_fig11_residual_norm_cauchy/sellmeier.png
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
from preprocess import find_cutoff_stft, load_spectra, resample  # noqa: E402
from scipy.stats import norm, probplot  # noqa: E402
from models import n_cauchy, n_sellmeier  # noqa: E402

# matplotlib 全局样式：PNG + 彩色 + 中文
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
C_F1 = "#d62728"     # 附件1（10°）红
C_F2 = "#1f77b4"     # 附件2（15°）蓝
C_CAU = "#2ca02c"    # Cauchy 绿
C_SEL = "#9467bd"    # Sellmeier 紫


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cutoff_safe(wn: np.ndarray, R: np.ndarray) -> float:
    """复现主代码 q2.py 的裁剪点与回退逻辑（保证图②标注与结果数据一致）。"""
    cut = float(find_cutoff_stft(wn, R))
    if not (700.0 <= cut <= 1600.0):
        cut = 1000.0
    return cut


# 图1：原始干涉光谱（每个附件一张图，一蓝一红）
def fig01_raw(out_dir: Path, wn1, R1, wn2, R2) -> None:
    for tag, wn, R, color, label in [
            ("f1", wn1, R1, C_F1, "附件1（10°）"),
            ("f2", wn2, R2, C_F2, "附件2（15°）")]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(wn, R, color=color, lw=0.8, label=label)
        ax.set_xlabel("波数（cm-1）")
        ax.set_ylabel("反射率（%）")
        ax.set_title("原始干涉光谱：" + label)
        ax.legend(loc="best")
        fig.savefig(out_dir / f"q2_fig01_raw_{tag}.png")
        plt.close(fig)
        print(f"save -> {out_dir.name}/q2_fig01_raw_{tag}.png")


# 图2：裁剪并等间距重采样后的光谱（每个附件一张图，标注裁剪分界点）
def fig02_crop(out_dir: Path, wn1, R1, wn2, R2) -> None:
    for tag, wn, R, color, label in [
            ("f1", wn1, R1, C_F1, "附件1（10°）"),
            ("f2", wn2, R2, C_F2, "附件2（15°）")]:
        cut = cutoff_safe(wn, R)
        mask = wn >= cut
        wnc, Rc = resample(wn[mask], R[mask])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(wnc, Rc, color=color, lw=0.9,
                label=f"{label}（裁剪点 {cut:.1f} cm-1）")
        ax.axvline(cut, color=color, ls="--", lw=0.9, alpha=0.55)
        ax.set_xlabel("波数（cm-1）")
        ax.set_ylabel("反射率（%）")
        ax.set_title("裁剪并等间距重采样后的光谱：" + label)
        ax.legend(loc="best")
        fig.savefig(out_dir / f"q2_fig02_cropped_{tag}.png")
        plt.close(fig)
        print(f"save -> {out_dir.name}/q2_fig02_cropped_{tag}.png")


# 图3：裁剪后原始光谱 vs 预处理后光谱（每个附件一个文件，两子图独立缩放）
def fig03_prepost(out_dir: Path, wn1, R1, wn2, R2, p1, p2) -> None:
    for tag, wn, R, p, color, label in [
            ("f1", wn1, R1, p1, C_F1, "附件1（10°）"),
            ("f2", wn2, R2, p2, C_F2, "附件2（15°）")]:
        cut = cutoff_safe(wn, R)
        mask = wn >= cut
        fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
        axes[0].plot(wn[mask], R[mask], color=color, lw=0.8,
                     label=f"裁剪点 {cut:.1f} cm-1")
        axes[0].axvline(cut, color=color, ls="--", lw=0.9, alpha=0.55)
        axes[0].set_title(f"{label} 裁剪后原始光谱")
        axes[0].set_ylabel("反射率（%）")
        axes[0].legend(loc="best")
        axes[1].plot(p[:, 0], p[:, 1], color=color, lw=1.0)
        axes[1].axvline(cut, color=color, ls="--", lw=0.9, alpha=0.55)
        axes[1].set_title(f"{label} 预处理后（去基线、去 DC）")
        axes[1].set_ylabel("反射率（%，去 DC）")
        axes[1].set_xlabel("波数（cm-1）")
        fig.tight_layout()
        fig.savefig(out_dir / f"q2_fig03_prepost_{tag}.png")
        plt.close(fig)
        print(f"save -> {out_dir.name}/q2_fig03_prepost_{tag}.png")


# 图4/5：模型拟合对比（实测=裁剪后原始反射率，模型=AsLS基线+振荡+DC 重建）
def fig_fit(out_dir: Path, result_dir: Path, model: str,
            wn1, R1, wn2, R2, p1, p2) -> None:
    data = np.loadtxt(result_dir / f"q2_fit_{model}.csv",
                      delimiter=",", skiprows=1)
    w1, _, f1_, w2, _, f2_ = data.T
    name = "Cauchy" if model == "cauchy" else "Sellmeier"
    tag = "q2_fig04_fit_cauchy.png" if model == "cauchy" else "q2_fig05_fit_sellmeier.png"
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True)
    panels = [(axes[0], w1, wn1, R1, f1_, p1, C_F1, "10°"),
              (axes[1], w2, wn2, R2, f2_, p2, C_F2, "15°")]
    for ax, wfit, wn_r, R_r, ffit, p, color, angle in panels:
        # 实测曲线：裁剪后原始反射率（线性插值到拟合网格）
        cut = cutoff_safe(wn_r, R_r)
        wnc, Rc = resample(wn_r[wn_r >= cut], R_r[wn_r >= cut])
        Rc = np.interp(wfit, wnc, Rc)
        # 模型曲线：AsLS 基线 + 拟合振荡 + DC 偏移（重建原始反射率口径）
        Rmodel = p[:, 2] + ffit + p[:, 3]
        ax.plot(wfit, Rc, color=color, lw=0.7, alpha=0.55, label="实测（原始）")
        ax.plot(wfit, Rmodel, color="k", lw=1.2, label="模型拟合")
        ax.set_ylabel("反射率（%）")
        ax.set_title(f"入射角 {angle}：{name} 模型拟合（原始反射率口径）")
        ax.legend(loc="lower right")
    axes[1].set_xlabel("波数（cm-1）")
    fig.tight_layout()
    fig.savefig(out_dir / tag)
    plt.close(fig)
    print("save -> %s/%s" % (out_dir.name, tag))


# 图6：残差分布（每个模型一张图，含两角度子图）
def fig06_residuals(out_dir: Path, result_dir: Path) -> None:
    for model, name in [("cauchy", "Cauchy"), ("sellmeier", "Sellmeier")]:
        d = np.loadtxt(result_dir / f"q2_fit_{model}.csv",
                       delimiter=",", skiprows=1)
        wn1, o1, f1_, wn2, o2, f2_ = d.T
        fig, axes = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True)
        for ax, wn, obs, fitv, color, angle in [
                (axes[0], wn1, o1, f1_, C_F1, "10°"),
                (axes[1], wn2, o2, f2_, C_F2, "15°")]:
            res = obs - fitv
            rmse = float(np.sqrt(np.mean(res ** 2)))
            ax.plot(wn, res, color=color, lw=0.7, alpha=0.85,
                    label=f"{name}（RMSE={rmse:.4f}%）")
            ax.axhline(0, color="k", lw=0.8, ls="--")
            ax.set_ylabel("残差（%）")
            ax.set_title(f"{name} 模型残差：入射角 {angle}（RMSE={rmse:.4f}%）")
            ax.legend(loc="upper right")
        axes[1].set_xlabel("波数（cm-1）")
        fig.tight_layout()
        fig.savefig(out_dir / f"q2_fig06_residuals_{model}.png")
        plt.close(fig)
        print("save -> %s/q2_fig06_residuals_%s.png" % (out_dir.name, model))


# 图7：抗噪声能力（不同噪声等级下的厚度估计漂移）
def fig07_reliability(out_dir: Path, result_dir: Path) -> None:
    with open(result_dir / "q2_results.json", encoding="utf-8") as f:
        res = json.load(f)
    with open(result_dir / "q2_noise_test.json", encoding="utf-8") as f:
        noise = json.load(f)

    models = ["cauchy", "sellmeier"]
    names = ["Cauchy", "Sellmeier"]
    colors = [C_CAU, C_SEL]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, c, name in zip(models, colors, names):
        etas = sorted(noise[m].keys(), key=float)
        x = [float(e) * 100 for e in etas]
        y = [noise[m][e]["d_mean"] for e in etas]
        yerr = [noise[m][e]["d_std"] for e in etas]
        ax.errorbar(x, y, yerr=yerr, fmt="-o", color=c, capsize=4,
                    label=f"{name}（联合 d={res['models'][m]['d']:.3f}）")
        ax.axhline(res["models"][m]["d"], color=c, ls=":", lw=0.9, alpha=0.55)
    ax.set_xlabel("噪声等级 η（%，相对信号 std）")
    ax.set_ylabel("厚度 d（μm）")
    ax.set_title("不同噪声等级下的厚度估计漂移")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "q2_fig07_reliability.png")
    plt.close(fig)
    print("save -> %s/q2_fig07_reliability.png" % out_dir.name)


# 图8：双角度交叉验证（单角度独立拟合 vs 联合拟合，与 FFT 初值对比）
def fig08_cross_check(out_dir: Path, result_dir: Path) -> None:
    with open(result_dir / "q2_results.json", encoding="utf-8") as f:
        res = json.load(f)
    models = ["cauchy", "sellmeier"]
    names = ["Cauchy", "Sellmeier"]
    colors = [C_CAU, C_SEL]
    labels = ["联合拟合", "单角度 10°", "单角度 15°"]
    xs = np.arange(3)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    w = 0.32
    for i, (m, name, color) in enumerate(zip(models, names, colors)):
        r = res["models"][m]
        vals = [r["d"], r["cross_check"]["d_angle1"], r["cross_check"]["d_angle2"]]
        x = xs + (i - 0.5) * w
        ax.bar(x, vals, width=w, color=color, label=name)
        for xx, v in zip(x, vals):
            ax.text(xx, v + 0.05, f"{v:.3f}", ha="center", fontsize=8)
    ax.axhline(res["d0_fft"], color="gray", ls="--", lw=1.2,
               label="FFT 初值 d0 = %.3f μm" % res["d0_fft"])
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("厚度 d（μm）")
    ax.set_title("双角度交叉验证：单角度独立拟合 vs 联合拟合")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "q2_fig08_cross_check.png")
    plt.close(fig)
    print("save -> %s/q2_fig08_cross_check.png" % out_dir.name)


# 图9：残差频谱（残差是否仍含周期结构 → 多光束干涉伏笔）
def fig09_residual_fft(out_dir: Path, result_dir: Path) -> None:
    for model, name in [("cauchy", "Cauchy"), ("sellmeier", "Sellmeier")]:
        d = np.loadtxt(result_dir / f"q2_fit_{model}.csv",
                       delimiter=",", skiprows=1)
        wn1, o1, f1_, wn2, o2, f2_ = d.T
        fig, axes = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True)
        for ax, wn, obs, fitv, color, angle in [
                (axes[0], wn1, o1, f1_, C_F1, "10°"),
                (axes[1], wn2, o2, f2_, C_F2, "15°")]:
            res = obs - fitv
            dnu = float(np.median(np.diff(wn)))
            F = np.abs(np.fft.rfft(res * np.hanning(len(res)))) ** 2
            f = np.fft.rfftfreq(len(res), d=dnu)
            ax.plot(f, F, color=color, lw=0.9)
            # 主峰位置 → 对应厚度 d = f0 / (2 n̄ cosθ1)
            sel = f > 0.0005
            fp = float(f[sel][int(np.argmax(F[sel]))])
            n_bar = 2.6
            th1 = float(np.arcsin(min(np.sin(np.deg2rad(int(angle[:2]))) / n_bar, 1.0)))
            d_from_f = fp / (2.0 * n_bar * np.cos(th1))
            ax.axvline(fp, color=color, ls="--", lw=0.9, alpha=0.6)
            ax.annotate(f"f0={fp:.4f}\n→ d≈{d_from_f:.2f} μm",
                        (fp, F.max()), textcoords="offset points",
                        xytext=(6, 0), va="center", fontsize=8)
            ax.set_ylabel("|FFT|²（a.u.）")
            ax.set_title(f"残差频谱：{name} 模型 入射角 {angle}")
        axes[1].set_xlim(0.0, 0.02)
        axes[1].set_xlabel("频率 f（周期/cm）")
        fig.tight_layout()
        fig.savefig(out_dir / f"q2_fig09_residual_fft_{model}.png")
        plt.close(fig)
        print("save -> %s/q2_fig09_residual_fft_%s.png" % (out_dir.name, model))


# 图10：色散曲线 n(λ)（Cauchy vs Sellmeier 拟合结果）
def fig10_dispersion(out_dir: Path, result_dir: Path) -> None:
    with open(result_dir / "q2_results.json", encoding="utf-8") as f:
        res = json.load(f)
    nu = np.linspace(1550.0, 4000.0, 400)   # cm⁻¹
    lam = 1e4 / nu                          # μm
    n_c = n_cauchy(nu, *res["models"]["cauchy"]["beta"])
    n_s = n_sellmeier(nu, *res["models"]["sellmeier"]["beta"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(lam, n_c, color=C_CAU, lw=1.6, label="Cauchy")
    ax.plot(lam, n_s, color=C_SEL, lw=1.6, label="Sellmeier")
    ax.set_xlabel("波长 λ（μm）")
    ax.set_ylabel("外延层折射率 n")
    ax.set_title("外延层折射率色散曲线 n(λ)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "q2_fig10_dispersion.png")
    plt.close(fig)
    print("save -> %s/q2_fig10_dispersion.png" % out_dir.name)


# 图11：残差正态性检验（直方图 + 正态 QQ 图）
def fig11_residual_norm(out_dir: Path, result_dir: Path) -> None:
    for model, name in [("cauchy", "Cauchy"), ("sellmeier", "Sellmeier")]:
        d = np.loadtxt(result_dir / f"q2_fit_{model}.csv",
                       delimiter=",", skiprows=1)
        wn1, o1, f1_, wn2, o2, f2_ = d.T
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 7))
        panels = [(axes[0, 0], axes[0, 1], o1 - f1_, C_F1, "10°"),
                  (axes[1, 0], axes[1, 1], o2 - f2_, C_F2, "15°")]
        for axh, axq, res, color, angle in panels:
            mu, sigma = float(np.mean(res)), float(np.std(res))
            axh.hist(res, bins=60, density=True, alpha=0.6, color=color)
            xg = np.linspace(float(res.min()), float(res.max()), 200)
            axh.plot(xg, norm.pdf(xg, mu, sigma), "k", lw=1.3, label="正态拟合")
            axh.set_title(f"{name} 残差直方图：入射角 {angle}")
            axh.set_xlabel("残差（%）")
            axh.set_ylabel("密度")
            axh.legend(loc="best")
            probplot(res, dist="norm", plot=axq)
            axq.set_title(f"{name} 正态 QQ 图：入射角 {angle}")
            axq.set_xlabel("理论分位数")
            axq.set_ylabel("样本分位数")
        fig.tight_layout()
        fig.savefig(out_dir / f"q2_fig11_residual_norm_{model}.png")
        plt.close(fig)
        print("save -> %s/q2_fig11_residual_norm_%s.png" % (out_dir.name, model))


def main() -> None:
    cfg = load_config()
    raw = cfg["data"]["raw_files"]
    result_dir = ROOT / cfg["result"]["dir"]
    out_dir = ROOT / cfg["visualize"]["output"] / "q2"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 56)
    print("Q2 生图：outputs/figures/q2/")
    print("=" * 56)

    # 原始光谱
    wn1, R1 = load_spectra(ROOT / raw["f1"])
    wn2, R2 = load_spectra(ROOT / raw["f2"])

    # 主代码输出的预处理后光谱
    p1 = np.loadtxt(result_dir / "q2_processed_f1.csv", delimiter=",", skiprows=1)
    p2 = np.loadtxt(result_dir / "q2_processed_f2.csv", delimiter=",", skiprows=1)

    fig01_raw(out_dir, wn1, R1, wn2, R2)
    fig02_crop(out_dir, wn1, R1, wn2, R2)
    fig03_prepost(out_dir, wn1, R1, wn2, R2, p1, p2)
    fig_fit(out_dir, result_dir, "cauchy", wn1, R1, wn2, R2, p1, p2)
    fig_fit(out_dir, result_dir, "sellmeier", wn1, R1, wn2, R2, p1, p2)
    fig06_residuals(out_dir, result_dir)
    fig07_reliability(out_dir, result_dir)
    fig08_cross_check(out_dir, result_dir)
    fig09_residual_fft(out_dir, result_dir)
    fig10_dispersion(out_dir, result_dir)
    fig11_residual_norm(out_dir, result_dir)

    print("完成。15 张图已输出到 outputs/figures/q2/。")


if __name__ == "__main__":
    main()
