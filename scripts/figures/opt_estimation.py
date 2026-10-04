"""Original synthetic robust fitting and analytical experiment design for OPT05."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

OUT = Path(__file__).resolve().parents[2] / "src/assets/optimization/opt-05-fitting-estimation"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "axes.facecolor": "white"})
rng = np.random.default_rng(15)
t = np.linspace(-1, 1, 40)
A = np.column_stack([np.ones_like(t), t])
b = .4 + 1.5*t + rng.normal(0, .08, len(t))
corrupted = [5, 12, 30]
b[corrupted] += 2.5
ls = np.linalg.lstsq(A, b, rcond=None)[0]
delta = .2
def objective(x):
    r = A @ x - b
    return np.where(abs(r) <= delta, .5*r*r, delta*abs(r)-.5*delta**2).sum(), A.T @ np.clip(r, -delta, delta)
fit = minimize(objective, ls, jac=True, method="BFGS", options={"gtol":1e-9})
assert np.linalg.norm(objective(fit.x)[1]) < 1e-6
fig, ax = plt.subplots(figsize=(7.2, 4), constrained_layout=True)
ax.scatter(t, b, color="#64748b", label="Synthetic observations")
ax.scatter(t[corrupted], b[corrupted], color="#dc2626", label="Deliberately shifted responses")
ax.plot(t, A @ ls, color="#ea580c", label="Squared-loss fit")
ax.plot(t, A @ fit.x, color="#2563eb", label="Huber-loss fit")
ax.plot(t, .4 + 1.5*t, "--", color="#16a34a", label="Generating line")
ax.set(xlabel="Input t", ylabel="Response", title="Changing the residual loss changes the fit")
ax.legend(fontsize=10)
fig.savefig(OUT/"robust-fitting.png", dpi=150)
plt.close(fig)

w = np.linspace(.025, .975, 600)
fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.8), constrained_layout=True)
d = -np.log(4*w)-np.log(1-w)
a = 1/(4*w)+1/(1-w)
ax[0].plot(w,d,color="#2563eb")
ax[0].scatter([.5],[-np.log(1)],color="#ea580c")
ax[0].axvline(.5, linestyle=":",color="#ea580c")
ax[0].set(xlabel="Weight on measurement (2, 0)",ylabel="−log det information",title="D-optimal: w = 1/2")
ax[1].plot(w,a,color="#16a34a")
ax[1].scatter([1/3],[2.25],color="#ea580c")
ax[1].axvline(1/3, linestyle=":",color="#ea580c")
ax[1].set(xlabel="Weight on measurement (2, 0)",ylabel="Trace of inverse information",title="A-optimal: w = 1/3",ylim=(1.5,15))
fig.savefig(OUT/"experiment-design.png",dpi=150)
plt.close(fig)
