import numpy as np

mish = lambda x: x * np.tanh(np.log1p(np.exp(x)))   # x * tanh(softplus(x))
leaky = lambda x: np.where(x > 0, x, 0.1 * x)         # the leaky rectifier of YOLO v1-v3

for v in (-3.0, -1.0, 0.0, 1.0, 3.0):
    print(f"x = {v:4.1f} | Mish {mish(v):7.4f} | leaky {float(leaky(v)):5.2f}")
x = np.linspace(-6, 6, 120001)
i = mish(x).argmin()
print(f"Mish minimum {mish(x[i]):.4f} at x = {x[i]:.3f}")
h = 1e-6
print(f"slope at 0: Mish {(mish(h) - mish(-h)) / (2 * h):.4f}; leaky jumps from 0.1 to 1")
