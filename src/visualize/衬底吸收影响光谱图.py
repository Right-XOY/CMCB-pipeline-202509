import matplotlib.pyplot as plt
import numpy as np

def fresnel_complex(n1, n2, theta1_deg, polarization='s'):
    """复数折射率菲涅尔系数，theta1_deg：介质1内角度(度)"""
    theta1 = np.radians(theta1_deg)
    n1c = n1
    n2c = n2
    # Snell
    sin_t2 = n1c * np.sin(theta1) / n2c
    cos_t2 = np.sqrt(1 - sin_t2**2)

    if polarization == 's':
        r = (n1c*np.cos(theta1) - n2c*cos_t2) / (n1c*np.cos(theta1) + n2c*cos_t2)
        t = 2*n1c*np.cos(theta1) / (n1c*np.cos(theta1) + n2c*cos_t2)
    else:
        r = (n2c*np.cos(theta1) - n1c*cos_t2) / (n2c*np.cos(theta1) + n1c*cos_t2)
        t = 2*n1c*np.cos(theta1) / (n2c*np.cos(theta1) + n1c*cos_t2)
    return r, t

def airy_full_model(nu, d, n0, n1, n2, theta0_deg):
    """完整Airy级数，不依赖Stokes关系，支持复数折射率"""
    theta0 = np.radians(theta0_deg)
    # Snell求外延层内折射角
    sin_t1 = n0 * np.sin(theta0)/n1
    theta1 = np.arcsin(sin_t1)
    cos_t1 = np.cos(theta1)

    delta = 4 * np.pi * n1 * d * cos_t1 * nu

    r01_s, t01_s = fresnel_complex(n0, n1, np.degrees(theta1), 's')
    r10_s, t10_s = fresnel_complex(n1, n0, np.degrees(theta0), 's')
    r12_s, t12_s = fresnel_complex(n1, n2, 0, 's')

    r01_p, t01_p = fresnel_complex(n0, n1, np.degrees(theta1), 'p')
    r10_p, t10_p = fresnel_complex(n1, n0, np.degrees(theta0), 'p')
    r12_p, t12_p = fresnel_complex(n1, n2, 0, 'p')

    def total_r(r01, t01, t10, r12, r10):
        numerator = t01 * t10 * r12 * np.exp(1j*delta)
        denominator = 1 - r10 * r12 * np.exp(1j*delta)
        r_total = r01 + numerator / denominator
        return r_total

    rs = total_r(r01_s, t01_s, t10_s, r12_s, r10_s)
    rp = total_r(r01_p, t01_p, t10_p, r12_p, r10_p)
    R = (np.abs(rs)**2 + np.abs(rp)**2)/2
    return R

# ---------------- 参数设置 ----------------
d_true = 4e-4       # 4 μm，单位 cm
n0 = 1.0
n1 = 3.42           # 硅外延，实数无吸收
kappa2_list = [0.0, 0.05, 0.20, 0.50]
color_list = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']
nu = np.linspace(1000, 4000, 800)
theta_inc = 10

# ------------------ 绘图，和上一张图风格严格对齐 ----------------
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(figsize=(9, 6), dpi=120)

for idx, k2 in enumerate(kappa2_list):
    n2 = 3.1 + 1j * k2
    R_curve = airy_full_model(nu, d_true, n0, n1, n2, theta_inc)
    ax.plot(nu, R_curve, color=color_list[idx], linewidth=1.6,
            label=r"$\kappa_2 =$" + f"{k2}")

ax.set_xlabel(r"波数 $\nu$ ($\mathrm{cm^{-1}}$)", fontsize=11)
ax.set_ylabel("反射率 $R$", fontsize=11)
ax.grid(alpha=0.25, linestyle="-")
ax.legend(loc="lower left", fontsize=9)
ax.set_xlim(1000,4000)

plt.tight_layout()
plt.show()
# plt.savefig("substrate_absorb_spectrum.png", bbox_inches="tight", dpi=150)
