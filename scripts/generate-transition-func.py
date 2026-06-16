import os

import numpy as np
import scipy.special as sc


def F(x):
    factor = np.sqrt(np.pi / 2)
    sqrtx = np.sqrt(x)

    S, C = sc.fresnel(sqrtx / factor)

    return 2j * sqrtx * np.exp(1j * x) * (factor * ((1 - 1j) / 2 - C + 1j * S))


x = np.logspace(-3, 1, 1000)
y = F(x)

A = np.abs(y)
P = np.angle(y, deg=True)

os.makedirs("data", exist_ok=True)
with open("data/transition-function.txt", "w") as f:
    f.write("x,magnitude,phase\n")
    for i in range(len(x)):
        f.write(f"{x[i]:.6e},{A[i]:.6e},{P[i]:.6e}\n")
