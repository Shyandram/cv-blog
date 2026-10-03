"""Extra teaching figures for ML 07 (diffusion and flow matching).

Run: MPLCONFIGDIR=/tmp/cv-blog-mpl .venv/bin/python scripts/figures/ml_07_diffusion_flow_matching_extra.py

Every sampler here uses the EXACT denoiser / velocity of a 2-D Gaussian
mixture, computed in closed form, so the figures isolate the behaviour of the
sampling rules themselves (no trained network, no downloads, a few seconds).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "src/assets/ml/ml-07-diffusion-flow-matching"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})
BLUE, ORANGE, GREEN, GRAY = "#2a6fdb", "#e4572e", "#17a398", "#888888"


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(OUT / name)


# ---------------------------------------------------------------- data model
ANG = np.arange(8) * 2 * np.pi / 8
MU = np.c_[2.0 * np.cos(ANG), 2.0 * np.sin(ANG)]      # 8 component means
SIG = 0.15                                             # component std
CLASS = (np.arange(8) % 2)                             # two "classes"


def sample_data(n, rng, comps=None):
    comps = np.arange(8) if comps is None else np.asarray(comps)
    k = rng.choice(comps, size=n)
    return MU[k] + SIG * rng.normal(size=(n, 2))


def gmm_eps(x, abar, comps=None):
    """Exact E[eps | x_t] for VP diffusion of the mixture (optionally restricted)."""
    comps = np.arange(8) if comps is None else np.asarray(comps)
    m = np.sqrt(abar) * MU[comps]                     # (K,2)
    var = abar * SIG**2 + 1 - abar
    d2 = ((x[:, None, :] - m[None]) ** 2).sum(-1)     # (N,K)
    logw = -d2 / (2 * var)
    w = np.exp(logw - logw.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    score = -(w[:, :, None] * (x[:, None, :] - m[None])).sum(1) / var
    return -np.sqrt(1 - abar) * score


def schedule(T=1000, kind="cosine"):
    if kind == "linear":
        beta = np.linspace(1e-4, 0.02, T)
        return np.cumprod(1 - beta)
    s = 0.008
    t = np.arange(T + 1) / T
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    abar = f[1:] / f[0]
    beta = np.clip(1 - abar / np.r_[1.0, abar[:-1]], 0, 0.999)
    return np.cumprod(1 - beta)


# ------------------------------------------------------- figure: schedules
def fig_schedules():
    T = 1000
    t = np.arange(1, T + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for kind, c in [("linear", BLUE), ("cosine", ORANGE)]:
        ab = schedule(T, kind)
        axes[0].plot(t, ab, color=c, label=kind)
        axes[1].plot(t, np.log(ab / (1 - ab)), color=c, label=kind)
    axes[0].set(xlabel="step t", ylabel=r"$\bar\alpha_t$ (signal fraction)",
                title="Cumulative signal level")
    axes[1].set(xlabel="step t", ylabel=r"log SNR = log($\bar\alpha_t/(1-\bar\alpha_t)$)",
                title="Log signal-to-noise ratio")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend()
    save(fig, "noise-schedules.png")


# ------------------------------------------------ DDPM / DDIM samplers (exact)
def ddpm(x, abar, eps_fn):
    T = len(abar)
    prev = np.r_[1.0, abar[:-1]]
    alpha = abar / prev
    beta = 1 - alpha
    rng = np.random.default_rng(1)
    for t in reversed(range(T)):
        e = eps_fn(x, abar[t])
        mean = (x - beta[t] / np.sqrt(1 - abar[t]) * e) / np.sqrt(alpha[t])
        var = beta[t] * (1 - prev[t]) / (1 - abar[t])
        x = mean + (np.sqrt(var) * rng.normal(size=x.shape) if t > 0 else 0)
    return x


def ddim(x, abar, eps_fn, steps, keep=None):
    T = len(abar)
    taus = np.linspace(T - 1, 0, steps).round().astype(int)
    path = [x.copy()]
    for i, t in enumerate(taus):
        e = eps_fn(x, abar[t])
        x0 = (x - np.sqrt(1 - abar[t]) * e) / np.sqrt(abar[t])
        ab_prev = abar[taus[i + 1]] if i + 1 < len(taus) else 1.0
        x = np.sqrt(ab_prev) * x0 + np.sqrt(1 - ab_prev) * e
        path.append(x.copy())
    return x, np.array(path)


def fig_samplers():
    rng = np.random.default_rng(0)
    abar = schedule(1000, "cosine")
    data = sample_data(1500, rng)
    z = rng.normal(size=(1500, 2))
    x_ddpm = ddpm(z.copy(), abar, gmm_eps)
    x_ddim10, path = ddim(z[:40].copy(), abar, gmm_eps, 10)
    x_ddim_all, _ = ddim(z.copy(), abar, gmm_eps, 10)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharex=True, sharey=True)
    axes[0].scatter(*data.T, s=3, color=GRAY)
    axes[0].set_title("Training data (8-component mixture)")
    axes[1].scatter(*x_ddpm.T, s=3, color=BLUE)
    axes[1].set_title("DDPM ancestral sampling, 1000 steps")
    axes[2].scatter(*x_ddim_all.T, s=3, color=ORANGE, alpha=.4)
    for p in range(path.shape[1]):
        axes[2].plot(path[:, p, 0], path[:, p, 1], color="k", lw=.6, alpha=.6)
    axes[2].scatter(*path[0].T, s=8, color="k", marker="x", label="start noise")
    axes[2].set_title("DDIM (eta = 0), 10 steps + paths")
    axes[2].legend(loc="lower right", fontsize=9)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-3.6, 3.6)
        ax.set_ylim(-3.6, 3.6)
        ax.grid(alpha=.2)
    save(fig, "ddpm-ddim-samplers.png")


# ------------------------------------------------------- figure: guidance
def fig_guidance():
    rng = np.random.default_rng(2)
    abar = schedule(1000, "cosine")
    cls = np.where(CLASS == 1)[0]
    z = rng.normal(size=(1200, 2))

    def guided(w):
        return lambda x, ab: gmm_eps(x, ab) + w * (gmm_eps(x, ab, cls) - gmm_eps(x, ab))

    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharex=True, sharey=True)
    for ax, w in zip(axes, [0.0, 1.0, 3.0, 8.0]):
        x, _ = ddim(z.copy(), abar, guided(w), 50)
        ax.scatter(*MU[cls].T, s=60, facecolors="none", edgecolors=ORANGE, lw=1.5)
        ax.scatter(*x.T, s=3, color=BLUE)
        name = "unconditional" if w == 0 else ("conditional" if w == 1 else "guided")
        ax.set_title(f"w = {w:g} ({name})")
        ax.set_aspect("equal")
        ax.set_xlim(-3.8, 3.8)
        ax.set_ylim(-3.8, 3.8)
        ax.grid(alpha=.2)
    fig.suptitle("Classifier-free guidance on a mixture: orange circles mark the 4 modes of class 1",
                 y=1.02)
    save(fig, "cfg-guidance.png")


# ------------------------------------------------- figure: flow matching steps
def fm_velocity(x, t):
    """Exact marginal velocity for x_t=(1-t) z + t x1, z~N(0,I), x1 ~ mixture."""
    m = t * MU
    var = (t * SIG) ** 2 + (1 - t) ** 2
    d2 = ((x[:, None, :] - m[None]) ** 2).sum(-1)
    logw = -d2 / (2 * var)
    w = np.exp(logw - logw.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    # E[x1 | x_t, k] for jointly Gaussian (x1, x_t) given component k
    gain = t * SIG**2 / var
    ex1 = (w[:, :, None] * (MU[None] + gain * (x[:, None, :] - m[None]))).sum(1)
    return (ex1 - x) / (1 - t)


def euler(x, steps, keep_path=False):
    path = [x.copy()]
    h = 1.0 / steps
    for k in range(steps):
        x = x + h * fm_velocity(x, min(k * h, 1 - 1e-4))
        path.append(x.copy())
    return x, np.array(path)


def fig_fm_steps():
    rng = np.random.default_rng(4)
    z = rng.normal(size=(1500, 2))
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharex=True, sharey=True)
    for ax, n in zip(axes, [1, 2, 4, 64]):
        x, _ = euler(z.copy(), n)
        ax.scatter(*x.T, s=3, color=BLUE)
        ax.set_title(f"Euler, {n} step{'s' if n > 1 else ''}")
    _, path = euler(z[:30].copy(), 64)
    for p in range(path.shape[1]):
        axes[3].plot(path[:, p, 0], path[:, p, 1], color="k", lw=.6, alpha=.6)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-3.6, 3.6)
        ax.set_ylim(-3.6, 3.6)
        ax.grid(alpha=.2)
    fig.suptitle("Exact marginal velocity, different step counts (paths drawn for 30 particles)",
                 y=1.02)
    save(fig, "fm-euler-steps.png")


if __name__ == "__main__":
    fig_schedules()
    fig_samplers()
    fig_guidance()
    fig_fm_steps()
