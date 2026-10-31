"""Figure S1 for PR 09 (Supplementary note on double descent).

Minimum-norm least squares with N random Fourier-type features on a seeded 1-D
regression problem (n = 40 training points). Median test MSE against the noiseless
target versus N, with the interquartile band over 50 repeats.
Writes only to src/assets/pattern-recognition/pr-09-ensembles-validation/.
Run: .venv/bin/python scripts/figures/pr_09_double_descent.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "src/assets/pattern-recognition/pr-09-ensembles-validation"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})

n, K, snr, n_rep = 40, 2000, 20.0, 50     # same settings as the snippet in the tutorial
counts = sorted(set([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 34, 38, 39, 40, 41, 42, 44, 48,
                     55, 65, 80, 100, 130, 200, 350, 600, 1000, 1500, 2000]))
k = np.arange(K)
p = 1.0 / (k + 1) ** 1.5
p /= p.sum()
freq = (k + 1) // 2


def basis(x, idx):
    arg = np.outer(x, freq[idx])
    return np.where(idx % 2 == 0, np.cos(arg), np.sin(arg))


x_test = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
h_test = basis(x_test, k) @ p
sigma = np.sqrt(np.mean(h_test ** 2) / snr)


def one_run(rng):
    x = rng.uniform(0, 2 * np.pi, n)
    y = basis(x, k) @ p + sigma * rng.standard_normal(n)
    order = rng.choice(K, size=max(counts), replace=False, p=p)
    out = []
    for N in counts:
        idx = order[:N]
        A = basis(x, idx) * np.sqrt(p[idx])
        beta = np.linalg.lstsq(A, y, rcond=None)[0]
        pred = basis(x_test, idx) @ (beta * np.sqrt(p[idx]))
        out.append((np.mean((pred - h_test) ** 2), np.mean((A @ beta - y) ** 2)))
    return out


res = np.array([one_run(np.random.default_rng(s)) for s in range(n_rep)])
test, train = res[:, :, 0], res[:, :, 1]
med = np.median(test, 0)
lo, hi = np.percentile(test, [25, 75], axis=0)

fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.fill_between(counts, lo, hi, color="#2563eb", alpha=0.18, lw=0, label="test MSE, interquartile range")
ax.plot(counts, med, "-o", color="#2563eb", ms=3.5, lw=1.8, label="test MSE (median of 50 repeats)")
ax.plot(counts, np.maximum(np.median(train, 0), 1e-8), "--", color="#dc2626", lw=1.5,
        label="training MSE (floored at $10^{-8}$)")
ax.axvline(n, color="#6b7280", lw=1.2, ls=":")
ax.text(n * 1.08, 2e-8 * 3, "$N = n = 40$\ninterpolation threshold", color="#374151", fontsize=9, va="bottom")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("number of features $N$ (model capacity)")
ax.set_ylabel("mean squared error")
ax.set_ylim(1e-8, 2e2)
ax.grid(alpha=0.25)
ax.legend(loc="upper right", fontsize=9, frameon=False)
fig.tight_layout()
path = OUT / "double-descent-toy.png"
fig.savefig(path, dpi=150)
plt.close(fig)
print(path, f"{path.stat().st_size / 1024:.0f} KB")
for N in (8, 40, 2000):
    print(N, f"{med[counts.index(N)]:.2e}")
