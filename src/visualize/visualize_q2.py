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
      q2_fig06_residuals_cauchy/sellmeier.png、q2_fig07_reliability.png
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


# 图3：预处理前后光谱对比（每个附件一个文件，内含原始/预处理后两子图，独立缩放）
def fig03_prepost(out_dir: Path, wn1, R1, wn2, R2, p1, p2) -> None:
    for tag, wn, R, p, color, label in [
            ("f1", wn1, R1, p1, C_F1, "附件1（10°）"),
            ("f2", wn2, R2, p2, C_F2, "附件2（15°）")]:
        fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
        axes[0].plot(wn, R, color=color, lw=0.8)
        axes[0].set_title(f"{label} 原始光谱")
        axes[0].set_ylabel("反射率（%）")
        axes[1].plot(p[:, 0], p[:, 1], color=color, lw=1.0)
        axes[1].set_title(f"{label} 预处理后（去基线、去 DC）")
        axes[1].set_ylabel("反射率（%，去 DC）")
        axes[1].set_xlabel("波数（cm-1）")
        fig.tight_layout()
        fig.savefig(out_dir / f"q2_fig03_prepost_{tag}.png")
        plt.close(fig)
        print(f"save -> {out_dir.name}/q2_fig03_prepost_{tag}.png")


# 图4/5：模型拟合对比（两角度 实测 vs 理论）
def fig_fit(out_dir: Path, result_dir: Path, model: str) -> None:
    data = np.loadtxt(result_dir / f"q2_fit_{model}.csv",
                      delimiter=",", skiprows=1)
    wn1, o1, f1_, wn2, o2, f2_ = data.T
    name = "Cauchy" if model == "cauchy" else "Sellmeier"
    tag = "q2_fig04_fit_cauchy.png" if model == "cauchy" else "q2_fig05_fit_sellmeier.png"
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True)
    for ax, wn, obs, fitv, color, angle in [
            (axes[0], wn1, o1, f1_, C_F1, "10°"),
            (axes[1], wn2, o2, f2_, C_F2, "15°")]:
        ax.plot(wn, obs, color=color, lw=0.7, alpha=0.55, label="实测")
        ax.plot(wn, fitv, color="k", lw=1.2, label="模型拟合")
        ax.set_ylabel("反射率（%）")
        ax.set_title(f"入射角 {angle}：{name} 模型拟合对比")
        ax.legend(loc="upper right")
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


# 图7：可靠性图（左：d 的 95% CI；右：抗噪漂移）
def fig07_reliability(out_dir: Path, result_dir: Path) -> None:
    with open(result_dir / "q2_results.json", encoding="utf-8") as f:
        res = json.load(f)
    with open(result_dir / "q2_noise_test.json", encoding="utf-8") as f:
        noise = json.load(f)

    models = ["cauchy", "sellmeier"]
    names = ["Cauchy", "Sellmeier"]
    colors = [C_CAU, C_SEL]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # 左：厚度 d 与 95% 置信区间（逐个模型绘制，避免 errorbar 多色限制）
    dvals = [res["models"][m]["d"] for m in models]
    ci = [res["models"][m]["param_ci"][0] for m in models]
    xs = np.arange(len(dvals))
    for x, d, c, color, name in zip(xs, dvals, ci, colors, names):
        axes[0].errorbar([x], [d], yerr=[[d - c[0]], [c[1] - d]], fmt="o",
                         color=color, ecolor=color, markersize=8, capsize=6,
                         label=name)
        axes[0].annotate(f"{d:.4f}", (x, d), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=9)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(names)
    axes[0].axhline(res["d0_fft"], color="gray", ls="--", lw=1,
                    label="FFT 初值 d0 = %.3f μm" % res["d0_fft"])
    axes[0].set_ylabel("厚度 d（μm）")
    axes[0].set_title("两模型厚度估计（含 95% 置信区间）")
    axes[0].legend(loc="best")

    # 右：抗噪能力（d_mean 随噪声等级漂移）
    for m, c, name in zip(models, colors, names):
        etas = sorted(noise[m].keys(), key=float)
        x = [float(e) * 100 for e in etas]
        y = [noise[m][e]["d_mean"] for e in etas]
        yerr = [noise[m][e]["d_std"] for e in etas]
        axes[1].errorbar(x, y, yerr=yerr, fmt="-o", color=c, capsize=4,
                         label=f"{name}（联合 d={res['models'][m]['d']:.3f}）")
        axes[1].axhline(res["models"][m]["d"], color=c, ls=":", lw=0.9, alpha=0.55)
    axes[1].set_xlabel("噪声等级 η（%，相对信号 std）")
    axes[1].set_ylabel("厚度 d（μm）")
    axes[1].set_title("不同噪声等级下的厚度估计漂移")
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_dir / "q2_fig07_reliability.png")
    plt.close(fig)
    print("save -> %s/q2_fig07_reliability.png" % out_dir.name)


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
    fig_fit(out_dir, result_dir, "cauchy")
    fig_fit(out_dir, result_dir, "sellmeier")
    fig06_residuals(out_dir, result_dir)
    fig07_reliability(out_dir, result_dir)

    print("完成。7 张图已输出到 outputs/figures/q2/。")


if __name__ == "__main__":
    main()
