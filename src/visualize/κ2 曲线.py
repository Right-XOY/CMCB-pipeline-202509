import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# ---------------------- 仿真数据 ----------------------
kappa2 = np.array([0.000, 0.005, 0.010, 0.020, 0.05, 0.20, 0.50])
delta_d = np.array([-0.063, -1.389, -2.761, -5.504, -13.584, -49.119, -90.427])
std_dev = np.array([0.063, 0.080, 0.100, 0.120, 0.060, 0.093, 0.022])

point_labels = [
    r"$\kappa_2=0$"+"\n无吸收",
    r"$\kappa_2=0.005$"+"\n极弱吸收",
    r"$\kappa_2=0.01$"+"\n弱吸收",
    r"$\kappa_2=0.02$",
    r"$\kappa_2=0.05$"+"\n中等掺杂",
    r"$\kappa_2=0.20$"+"\n重掺杂",
    r"$\kappa_2=0.50$"+"\n强重掺杂",
]

# ---------------------- 画布初始化 ----------------------
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(figsize=(9, 6), dpi=120)

# 1. 仿真点（最高层级）
ax.errorbar(kappa2, delta_d, yerr=std_dev, fmt='o', color='#1f77b4', capsize=4,
            markersize=7, label="仿真结果", zorder=5)

# 2. 平滑样条曲线
x_smooth = np.linspace(kappa2.min(), kappa2.max(), 300)
spl = make_interp_spline(kappa2, delta_d, k=3)
y_smooth = spl(x_smooth)
ax.plot(x_smooth, y_smooth, color='#1f77b4', linestyle='-', linewidth=1.6, alpha=0.7, zorder=2)

# 3. 浅蓝色填充
ax.fill_between(x_smooth, y_smooth, 0, color='#a8d1ff', alpha=0.35, zorder=1)

# 4. 零偏差虚线
ax.axhline(y=0, color='black', linestyle='--', linewidth=1.2, label="零系统偏差", zorder=3)

# ========== 标注位置微调：0.5继续向上 ==========
annot_list = [
    {"idx":0, "dx":22, "dy":5},     # k2=0：向右偏移，离开左上角
    {"idx":4, "dx":18, "dy":10},    # k2=0.05：右上空白
    {"idx":5, "dx":18, "dy":-6},    # k2=0.20
    {"idx":6, "dx":-32, "dy":14},   # k2=0.50：dy从2提升到14，进一步向上
]
for item in annot_list:
    i = item["idx"]
    ax.annotate(
        point_labels[i],
        xy=(kappa2[i], delta_d[i]),
        xytext=(item["dx"], item["dy"]),
        textcoords='offset points',
        fontsize=9.5,
        va="center",
        zorder=6
    )

# ---------------------- 坐标轴设置 ----------------------
ax.set_xlabel(r"衬底消光系数 $\kappa_2$", fontsize=11)
ax.set_ylabel(r"厚度系统偏差 $\Delta d = d_{est}-d_{true}$  (nm)", fontsize=11)
ax.set_xlim(-0.01, 0.53)
ax.grid(alpha=0.25, linestyle="-")
ax.legend(loc="lower left", fontsize=9)

plt.tight_layout()
plt.show()
# plt.savefig("kappa2_error_curve.png", bbox_inches="tight", dpi=150)
