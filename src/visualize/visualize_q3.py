"""问题三生图脚本（附件3/4 原始干涉光谱，复刻 q2 图1 样式）。

运行方式（在项目根目录）：
    python src/visualize/visualize_q3.py

数据来源：
  - 原始光谱：data/raw/附件3.xlsx、附件4.xlsx（复用 src/model/preprocess.load_spectra）

输出：outputs/figures/q3/q3_fig01_raw_f3.png、q3_fig01_raw_f4.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "model"))
from preprocess import load_spectra  # noqa: E402

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
C_F3 = "#d62728"     # 附件3（10°）红，与 q2 附件1 同色
C_F4 = "#1f77b4"     # 附件4（15°）蓝，与 q2 附件2 同色


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fig01_raw(out_dir: Path, wn3, R3, wn4, R4) -> None:
    """原始干涉光谱（未预处理），每个附件一张图，格式与 q2 图1 相同。"""
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


def main() -> None:
    cfg = load_config()
    raw = cfg["data"]["raw_files"]
    out_dir = ROOT / cfg["visualize"]["output"] / "q3"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 56)
    print("Q3 生图：outputs/figures/q3/（附件3/4 原始干涉光谱）")
    print("=" * 56)

    wn3, R3 = load_spectra(ROOT / raw["f3"])
    wn4, R4 = load_spectra(ROOT / raw["f4"])
    print(f"  附件3：{len(wn3)} 点，波数 {wn3.min():.2f}-{wn3.max():.2f} cm-1")
    print(f"  附件4：{len(wn4)} 点，波数 {wn4.min():.2f}-{wn4.max():.2f} cm-1")

    fig01_raw(out_dir, wn3, R3, wn4, R4)

    print("完成。2 张图已输出到 outputs/figures/q3/。")


if __name__ == "__main__":
    main()
