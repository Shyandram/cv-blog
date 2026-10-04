"""Original analytical dual bounds and sensitivity examples for OPT04."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "src/assets/optimization/opt-04-duality-kkt"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "axes.facecolor": "white"})

x = np.linspace(-0.5, 3, 500)
lam = np.linspace(0, 3, 500)
fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.6), constrained_layout=True)
ax[0].plot(x, 0.5 * (x - 2)**2, color="#2563eb", label="Primal objective")
ax[0].axvspan(-0.5, 1, color="#2563eb", alpha=.12, label="Feasible x ≤ 1")
ax[0].scatter([1], [.5], color="#ea580c", zorder=5)
ax[0].set(xlabel="Decision x", ylabel="Cost", title="Primal upper bound", ylim=(-.15, 3.3))
ax[0].legend(fontsize=10)
ax[1].plot(lam, lam - .5 * lam**2, label="Dual lower bound", color="#16a34a")
ax[1].axhline(.5, color="#ea580c", linestyle="--", label="Primal optimum")
ax[1].scatter([1], [.5], color="#ea580c", zorder=5)
ax[1].set(xlabel="Multiplier λ ≥ 0", ylabel="Bound", title="Best lower bound meets optimum")
ax[1].legend(fontsize=10)
fig.savefig(OUT / "dual-bounds.png", dpi=150)
plt.close(fig)

b = np.linspace(-1, 4, 500)
value = .5 * np.maximum(2 - b, 0)**2
price = np.maximum(2 - b, 0)
fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.6), constrained_layout=True)
ax[0].plot(b, value, label="Optimal value", color="#2563eb")
ax[0].plot(b, .32 - .8 * (b - 1.2), "--", label="Supporting lower bound", color="#ea580c")
ax[0].scatter([1.2], [.32], color="#ea580c")
ax[0].set(xlabel="Allowed upper bound b", ylabel="Best cost", title="Value of relaxing the constraint", ylim=(-.3, 4.8))
ax[0].legend(fontsize=10)
ax[1].plot(b, price, color="#16a34a")
ax[1].axvline(2, color="gray", linestyle=":")
ax[1].set(xlabel="Allowed upper bound b", ylabel="Optimal multiplier", title="Price vanishes after b = 2")
fig.savefig(OUT / "sensitivity.png", dpi=150)
plt.close(fig)
