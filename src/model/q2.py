# -*- coding: utf-8 -*-
"""问题二主流程：数据预处理 → 初值估计 → L-M 拟合 → 可靠性检验。

运行方式（在项目根目录）：
    python src/model/q2.py

依赖 config.yaml 中的路径配置；数值结果输出到 outputs/result/，不绘图
（绘图交给 src/visualize/ 生图阶段）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

from preprocess import load_spectra, preprocess
from fitting import (de_init_beta, fft_init_d, fit_statistics, lm_fit,
                     noise_robustness_test, parameter_uncertainty,
                     profile_scan_d, single_angle_cross_check)

# 物理常量与入射角（题目给定）
N0 = 1.0              # 空气折射率
N2 = 2.65             # SiC 衬底折射率（用户确认：选定带宽内近似平稳）
ANG1 = 10.0           # 附件1 入射角（°）
ANG2 = 15.0           # 附件2 入射角（°）
ANGLES = (ANG1, ANG2)

# 预处理默认值（与推导过程文档一致）
PREPROC = dict(
    hampel_win=11, hampel_t=3.0,
    asls_lam=1e5, asls_p=0.01,
    sg_win=15, sg_poly=3,
)

# 参数约束范围：theta = [d(μm), 色散系数...]（线性参数 a,b 不进优化器，闭式消元）
BOUNDS = {
    "cauchy": [(0.5, 200.0), (1.5, 3.5), (-1.0, 1.0), (-1.0, 1.0)],
    "sellmeier": [(0.5, 200.0), (0.0, 3.0), (0.1, 2000.0), (0.0, 3.0), (0.1, 2000.0)],
    "sellmeier1": [(0.5, 200.0), (0.0, 8.0), (0.0, 1.0)],
}
# DE 初值搜索只针对色散系数
DE_BOUNDS = {
    "cauchy": [(1.5, 3.5), (-1.0, 1.0), (-1.0, 1.0)],
    "sellmeier": [(0.0, 3.0), (0.1, 2000.0), (0.0, 3.0), (0.1, 2000.0)],
    "sellmeier1": [(0.0, 8.0), (0.0, 1.0)],
}
PARAM_NAMES = {
    "cauchy": ["d(μm)", "A", "B", "C"],
    "sellmeier": ["d(μm)", "B1", "C1", "B2", "C2"],
    "sellmeier1": ["d(μm)", "B", "C"],
}
SEED = 2025


def load_config() -> tuple[Path, dict]:
    """读取 config.yaml，返回 (项目根目录, 配置字典)。"""
    root = Path(__file__).resolve().parents[2]
    with open(root / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return root, cfg


def save_csv(path: Path, header: str, cols: list[np.ndarray]) -> None:
    """写 CSV（带表头，保留较高精度）。"""
    arr = np.column_stack(cols)
    fmt = ",".join(["%.6f"] * arr.shape[1])
    np.savetxt(path, arr, fmt=fmt, header=header, comments="")
    print(f"  [输出] {path.name}  ({arr.shape[0]} 行)")


def save_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  [输出] {path.name}")


def solve_model(wn1, R1, wn2, R2, d0, model: str,
                d_half: float = 0.5, step: float = 0.01) -> tuple[dict, object]:
    """对单个模型执行 DE 初值 → 剖面扫描精调 d → 可靠性统计。

    返回 (可序列化摘要 dict, FitResult 拟合对象)。
    线性参数 (a, b) 每个角度独立、闭式消元，不进优化器。
    厚度 d 先由 FFT 固定 d0 估计色散初值，再用剖面扫描在 [d0±d_half] 内
    精调：每个候选 d 固定、仅优化色散系数，取 SSE 最小者，锁在正确分支内。
    """
    # 1) DE 初值（色散系数，厚度固定 d0）
    beta0 = de_init_beta(wn1, R1, wn2, R2, d0, model, N2, ANGLES,
                         DE_BOUNDS[model], seed=SEED)

    # 2) 剖面扫描精调厚度 d（每个 d 固定，仅优化色散系数）
    fit, profile = profile_scan_d(wn1, R1, wn2, R2, d0, beta0, model, N2,
                                  ANGLES, BOUNDS[model], d_half=d_half,
                                  step=step, seed=SEED)
    d = fit.d

    # 3) 拟合优度统计（自由度 = 色散系数数 + 线性参数数）
    stats = fit_statistics(fit.res, fit.res.x.size + fit.n_lin, R1, R2)

    # 4) 色散系数不确定性（协方差 → 置信区间）
    pnames = PARAM_NAMES[model][1:]
    sd, ci, _ = parameter_uncertainty(fit.res, pnames, n_lin=fit.n_lin)

    # 5) 双角度一致性交叉验证（独立自由拟合，仅用于检验）
    cross = single_angle_cross_check(wn1, R1, wn2, R2, d, beta0, model,
                                     N2, ANGLES, BOUNDS[model], seed=SEED)

    linear = {f"angle{int(k)}": {"a": v[0], "b": v[1]}
              for k, v in fit.linear.items()}
    summary = {
        "model": model,
        "theta": fit.theta.tolist(),
        "d": d, "d0": d0,
        "d_ci": profile["d_ci"],
        "d_std": profile["std_d"],
        "profile_sse_min": profile["sse_min"],
        "profile_d_grid": profile["d_grid"],
        "profile_sse": profile["sse"],
        "beta": fit.beta.tolist(),
        "linear": linear,
        "param_sd": sd.tolist(),
        "param_ci": ci.tolist(),
        "param_names": pnames,
        "stats": stats,
        "cross_check": cross,
        "beta0": beta0.tolist(),
    }
    return summary, fit


def main() -> None:
    root, cfg = load_config()
    raw = cfg["data"]["raw_files"]
    result_dir = root / cfg["result"]["dir"]
    result_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("问题二：碳化硅外延层厚度反演（双光束干涉模型）")
    print("=" * 60)

    # 1. 读取与预处理
    print("\n[1] 数据读取与预处理")
    spec = {}
    for key, path in [("f1", raw["f1"]), ("f2", raw["f2"])]:
        wn, R = load_spectra(root / path)
        wn_p, R_p, cutoff, baseline, dc = preprocess(wn, R, **PREPROC)
        # 裁剪分界点合理性防护：异常时回退默认 1000 cm⁻¹
        if not (700.0 <= cutoff <= 1600.0):
            cutoff = 1000.0
            wn_p, R_p, _, baseline, dc = preprocess(wn, R, cutoff=cutoff, **PREPROC)
        spec[key] = (wn_p, R_p)
        print(f"  {key}: {len(wn)} 点 -> 裁剪点 {cutoff:.1f} cm-1 -> {len(wn_p)} 点")
        save_csv(result_dir / f"q2_processed_{key}.csv",
                 "wavenumber_cm-1,reflectance_osc_%,baseline_%,dc_%",
                 [wn_p, R_p, baseline, np.full(len(baseline), dc)])

    wn1, R1 = spec["f1"]
    wn2, R2 = spec["f2"]

    # 2. FFT 厚度初值
    print("\n[2] FFT 厚度初值估计")
    d0 = fft_init_d(wn1, R1, n_avg=2.6, theta0_deg=ANG1, search_lo=1500.0)
    print(f"  d0 = {d0:.3f} um")

    # 3. 两模型求解
    print("\n[3] 模型求解（DE 初值 + 剖面扫描精调 d）")
    results = {}
    for model in ["cauchy", "sellmeier1"]:
        print(f"  --- 模型: {model} ---")
        results[model], fit = solve_model(wn1, R1, wn2, R2, d0, model)
        r = results[model]
        lin = r["linear"]
        d_ci = r["d_ci"]
        print(f"    d = {r['d']:.4f} um  95%CI [{d_ci[0]:.4f}, {d_ci[1]:.4f}]  "
              f"RMSE = {r['stats']['RMSE']:.4f}   R^2 = {r['stats']['R2']:.4f}")
        print(f"    线性校正 10deg: a={lin['angle10']['a']:.4f} b={lin['angle10']['b']:.4f}  |  "
              f"15deg: a={lin['angle15']['a']:.4f} b={lin['angle15']['b']:.4f}")
        print(f"    单角度独立拟合(自由d): 10deg -> {r['cross_check']['d_angle1']:.4f} um, "
              f"15deg -> {r['cross_check']['d_angle2']:.4f} um")
        # 拟合曲线落盘（供生图阶段直接读取）
        save_csv(result_dir / f"q2_fit_{model}.csv",
                 "wn1,R_obs1,R_fit1,wn2,R_obs2,R_fit2",
                 [wn1, R1, fit.fit1, wn2, R2, fit.fit2])
        # 参数表落盘：d 用剖面扫描值 + profile CI；色散系数用协方差 sd/ci
        names = PARAM_NAMES[model]
        theta_full = np.concatenate([[r["d"]], np.asarray(r["beta"], float)])
        sd_full = np.concatenate([[r["d_std"]], np.asarray(r["param_sd"], float)])
        ci_lo = np.concatenate([[d_ci[0]], np.asarray(r["param_ci"], float)[:, 0]])
        ci_hi = np.concatenate([[d_ci[1]], np.asarray(r["param_ci"], float)[:, 1]])
        param_csv = np.column_stack([np.array(names, dtype=object),
                                     np.round(theta_full, 6), np.round(sd_full, 6),
                                     np.round(ci_lo, 6), np.round(ci_hi, 6)])
        np.savetxt(result_dir / f"q2_parameters_{model}.csv", param_csv,
                   fmt="%s", delimiter=",",
                   header="param,value,sd,ci_lo,ci_hi", comments="")
        print(f"  [输出] q2_parameters_{model}.csv")

    # 4. 抗噪声能力测试（全局：每次加噪后重新走完整反演）
    print("\n[4] 抗噪声能力测试（全局反演，含 DE，较慢）")
    noise = {}
    for model in ["cauchy", "sellmeier1"]:
        noise[model] = noise_robustness_test(
            wn1, R1, wn2, R2, model, N2, ANGLES,
            BOUNDS[model], DE_BOUNDS[model],
            levels=(0.01, 0.02, 0.03, 0.04, 0.05), nrep=5, seed=SEED,
            profile_step=0.05, d_half=2.0)
        for eta, v in noise[model].items():
            print(f"  {model} eta={eta}: d = {v['d_mean']:.4f} +/- {v['d_std']:.4f} um")
    save_json(result_dir / "q2_noise_test.json", noise)

    # 5. 汇总结果
    summary = {"d0_fft": d0, "N0": N0, "N2": N2,
               "angles": {"f1": ANG1, "f2": ANG2},
               "models": results}
    save_json(result_dir / "q2_results.json", summary)

    # 打印汇总对比
    print("\n" + "=" * 60)
    print("两模型结果对比")
    print("=" * 60)
    print(f"{'指标':<10}{'Cauchy':>14}{'Sellmeier1':>14}")
    for key, label in [("d", "d (um)"),
                       ("RMSE", "RMSE"), ("R2", "R^2"),
                       ("AIC", "AIC"), ("BIC", "BIC")]:
        if key in ("RMSE", "R2", "AIC", "BIC"):
            row = f"{label:<10}"
            for m in ["cauchy", "sellmeier1"]:
                row += f"{results[m]['stats'][key]:>14.4f}"
        else:
            row = f"{label:<10}"
            for m in ["cauchy", "sellmeier1"]:
                row += f"{results[m][key]:>14.4f}"
        print(row)
    print("\n完成。结果已写入 outputs/result/ 目录。")


if __name__ == "__main__":
    sys.exit(main())
