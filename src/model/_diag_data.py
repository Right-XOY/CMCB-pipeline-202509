# -*- coding: utf-8 -*-
"""快速数据刻画：SiC 附件1/2 的反射率范围、条纹幅度、FFT 厚度。"""
import sys
import numpy as np
sys.path.insert(0, "src/model")

from preprocess import load_spectra
from q3 import preprocess_sic, detrend_poly, fft_init_d

for name, path in [("附件1(10°)", "data/raw/附件1.xlsx"),
                   ("附件2(15°)", "data/raw/附件2.xlsx")]:
    wn, R = load_spectra(path)
    w, r, cutoff = preprocess_sic(wn, R)
    # 去趋势后的振荡幅度
    osc = detrend_poly(w, r, deg=2)
    amp = float(np.max(osc) - np.min(osc))
    print(f"[{name}] 原始 {len(wn)} pts, 波数 [{wn.min():.0f},{wn.max():.0f}]")
    print(f"   反射率 R: min={r.min():.3f} max={r.max():.3f} mean={r.mean():.3f} (%)")
    print(f"   cutoff={cutoff:.0f}, 处理后 {len(w)} pts")
    print(f"   去趋势(deg2)条纹幅度(峰谷)={amp:.4f} %")
    d0 = fft_init_d(w, osc, n_avg=2.6, theta0_deg=10, search_lo=1500)
    print(f"   FFT d0={d0:.4f} um")
    print()
