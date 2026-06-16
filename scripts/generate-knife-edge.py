import numpy as np
import scipy.special as sc

nu = np.linspace(-3, 3, 800)
# Scipy fresnel usages: S, C
S, C = sc.fresnel(nu)
# F(v) = (1+j)/2 * integral_v^infty exp(-j pi t^2 / 2) dt
# |F(v)| = 0.5 * ((0.5 - C)**2 + (0.5 - S)**2 )
abs_F_sq = 0.5 * ((0.5 - C) ** 2 + (0.5 - S) ** 2)
L_ke = -10 * np.log10(abs_F_sq)  # 10 because F is squared to remove complex values

data = np.column_stack((nu, L_ke))
np.savetxt(
    "data/knife-edge-attenuation.txt",
    data,
    fmt="%.4f",
    delimiter=",",
    header="nu,L",
    comments="",
)
print("Generated knife-edge-attenuation.txt")
