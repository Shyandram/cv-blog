"""Extra figures for PR 08 (approximate inference).

Writes only into src/assets/pattern-recognition/pr-08-approximate-inference/.
Run: .venv/bin/python scripts/figures/pr_08_approximate_inference_extra.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import norm  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "src/assets/pattern-recognition/pr-08-approximate-inference"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.facecolor": "white", "axes.facecolor": "white"})
BLUE, ORANGE, GREEN, GRAY = "#2563eb", "#ea580c", "#16a34a", "#6b7280"


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name, (OUT / name).stat().st_size // 1024, "KB")


# ---------------------------------------------------------------- logistic posterior
def logistic_posterior():
    rng = np.random.default_rng(0)
    n = 40
    X = np.c_[np.ones(n), rng.normal(size=n)]
    y = (rng.random(n) < 1 / (1 + np.exp(-X @ np.array([-0.5, 1.5])))).astype(float)
    prior_var = 4.0

    def log_joint(w):
        s = X @ w
        return np.sum(y * s - np.logaddexp(0, s)) - 0.5 * np.sum(w**2) / prior_var

    def grad(W):
        p = 1 / (1 + np.exp(-W @ X.T))
        return (y - p) @ X - W / prior_var

    g0 = np.linspace(-1.8, 1.2, 181)
    g1 = np.linspace(0.0, 3.6, 181)
    W0, W1 = np.meshgrid(g0, g1, indexing="xy")
    logp = np.array([[log_joint(np.array([a, b])) for a in g0] for b in g1])
    post = np.exp(logp - logp.max())

    w = np.zeros(2)
    for _ in range(25):
        p = 1 / (1 + np.exp(-X @ w))
        H = X.T @ (X * (p * (1 - p))[:, None]) + np.eye(2) / prior_var
        w = w + np.linalg.solve(H, grad(w[None])[0])
    cov_lap = np.linalg.inv(H)

    m, s = np.zeros(2), np.full(2, -1.0)
    for _ in range(3000):
        eps = rng.normal(size=(64, 2))
        g = grad(m + np.exp(s) * eps)
        m += 0.02 * g.mean(axis=0)
        s += 0.02 * ((g * eps).mean(axis=0) * np.exp(s) + 1.0)

    def gauss(mean, cov):
        P = np.linalg.inv(cov)
        d = np.stack([W0 - mean[0], W1 - mean[1]], axis=-1)
        return np.exp(-0.5 * np.einsum("...i,ij,...j->...", d, P, d))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    levels = [0.05, 0.3, 0.7]
    panels = [("Exact posterior (grid)", post, None),
              ("Laplace approximation", gauss(w, cov_lap), w),
              ("Mean-field Gaussian VI", gauss(m, np.diag(np.exp(2 * s))), m)]
    for ax, (title, values, centre) in zip(axes, panels):
        ax.contour(W0, W1, post, levels=levels, colors=GRAY, linestyles="dashed")
        if centre is not None:
            ax.contour(W0, W1, values, levels=levels, colors=BLUE)
            ax.plot(*centre, "o", color=BLUE, ms=4)
        else:
            ax.contourf(W0, W1, post, levels=[0.05, 0.3, 0.7, 1.01], cmap="Blues", alpha=0.7)
        ax.set(title=title, xlabel="w0 (bias)")
    axes[0].set_ylabel("w1 (slope)")
    fig.suptitle("Bayesian logistic regression, 40 points: dashed = exact contours", y=1.02)
    fig.tight_layout()
    save(fig, "logistic-posterior.png")


# ---------------------------------------------------------------- importance sampling
def importance():
    rng = np.random.default_rng(1)
    N = 4000
    t = np.linspace(-3, 5, 400)
    target = norm.pdf(t, 1.5, 0.7)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    ax.plot(t, target, color="black", lw=2, label="target N(1.5, 0.7²)")
    colors = [BLUE, GREEN, ORANGE]
    ess_text = []
    for scale, c in zip([1.0, 3.0, 0.3], colors):
        ax.plot(t, norm.pdf(t, 0, scale), color=c, label=f"proposal N(0, {scale}²)")
        z = rng.normal(0, scale, N)
        lw = norm.logpdf(z, 1.5, 0.7) - norm.logpdf(z, 0, scale)
        wn = np.exp(lw - lw.max())
        wn /= wn.sum()
        ess = 1 / np.sum(wn**2)
        ess_text.append((scale, ess, np.sort(wn)[::-1], c))
    ax.set(title="Target and three proposals", xlabel="z", ylabel="density")
    ax.legend(fontsize=9)
    ax = axes[1]
    for scale, ess, sorted_w, c in ess_text:
        ax.plot(np.arange(1, N + 1), np.cumsum(sorted_w), color=c,
                label=f"N(0, {scale}²): ESS ≈ {ess:.0f} of {N}")
    ax.set(xscale="log", title="Cumulative share of the largest weights",
           xlabel="number of largest weights (log scale)", ylabel="share of total weight")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    save(fig, "importance-weights.png")


# ---------------------------------------------------------------- MH step size
def mh_step_sizes():
    rng = np.random.default_rng(12)

    def log_target(x):
        return -0.5 * ((x - 1.5) / 0.7) ** 2

    def chain(step, n=4000, x=-3.0):
        out, acc = np.empty(n), 0
        for i in range(n):
            prop = x + rng.normal(0, step)
            if np.log(rng.random()) < log_target(prop) - log_target(x):
                x, acc = prop, acc + 1
            out[i] = x
        return out, acc / n

    def acf(x, lags=60):
        x = x - x.mean()
        v = np.dot(x, x)
        return np.array([np.dot(x[: len(x) - k], x[k:]) / v for k in range(lags)])

    fig, axes = plt.subplots(2, 3, figsize=(10.4, 4.9))
    for j, (step, c) in enumerate(zip([0.02, 0.8, 8.0], [ORANGE, BLUE, GREEN])):
        xs, a = chain(step)
        axes[0, j].plot(xs, color=c, lw=0.7)
        axes[0, j].axhline(1.5, color="black", lw=0.8, ls="--")
        axes[0, j].set(title=f"step sd {step}: acceptance {a:.2f}", xlabel="iteration",
                       ylim=(-3.5, 4.5))
        burn = xs[1000:]
        axes[1, j].bar(np.arange(60), acf(burn), color=c, width=0.8)
        axes[1, j].set(xlabel="lag", ylim=(-0.2, 1.05), title="autocorrelation after warm-up")
    axes[0, 0].set_ylabel("x")
    axes[1, 0].set_ylabel("ρ(lag)")
    fig.tight_layout()
    save(fig, "mh-step-size.png")


# ---------------------------------------------------------------- Gibbs
def gibbs_path():
    rng = np.random.default_rng(3)
    rho = 0.85
    sd = np.sqrt(1 - rho**2)
    z = np.array([2.6, -2.2])
    path = [z.copy()]
    draws = []
    for t in range(2000):
        z[0] = rng.normal(rho * z[1], sd)
        path.append(z.copy())
        z[1] = rng.normal(rho * z[0], sd)
        path.append(z.copy())
        draws.append(z.copy())
    path = np.array(path)
    draws = np.array(draws)
    axis = np.linspace(-3.2, 3.2, 200)
    xx, yy = np.meshgrid(axis, axis)
    P = np.linalg.inv(np.array([[1, rho], [rho, 1]]))
    pts = np.stack([xx, yy], -1)
    dens = np.exp(-0.5 * np.einsum("...i,ij,...j->...", pts, P, pts))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax in axes:
        ax.contour(xx, yy, dens, levels=[0.05, 0.3, 0.7], colors=GRAY)
        ax.set_aspect("equal")
        ax.set(xlabel="z1", xlim=(-3.2, 3.2), ylim=(-3.2, 3.2))
    axes[0].plot(path[:41, 0], path[:41, 1], "-", color=BLUE, lw=1)
    axes[0].plot(path[0, 0], path[0, 1], "s", color=ORANGE, ms=6, label="start")
    axes[0].legend(loc="lower right")
    axes[0].set(title="First 20 Gibbs sweeps (axis-aligned moves)", ylabel="z2")
    axes[1].scatter(draws[200:, 0], draws[200:, 1], s=3, alpha=0.4, color=BLUE)
    axes[1].set(title="1,800 draws after warm-up")
    fig.tight_layout()
    save(fig, "gibbs-path.png")


# ---------------------------------------------------------------- KL directions
def kl_directions():
    t = np.linspace(-6, 6, 2001)
    dt = t[1] - t[0]
    p = 0.5 * norm.pdf(t, -2, 0.7) + 0.5 * norm.pdf(t, 2, 0.7)
    mean_p = np.sum(t * p) * dt
    var_p = np.sum((t - mean_p) ** 2 * p) * dt

    def reverse_kl(params):
        m, log_s = params
        q = norm.pdf(t, m, np.exp(log_s))
        mask = q > 1e-300
        return np.sum(q[mask] * (np.log(q[mask]) - np.log(p[mask] + 1e-300))) * dt

    res = minimize(reverse_kl, x0=[1.0, 0.0], method="Nelder-Mead")
    m_r, s_r = res.x[0], np.exp(res.x[1])
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.fill_between(t, p, color=GRAY, alpha=0.35, label="target p (two modes)")
    ax.plot(t, norm.pdf(t, m_r, s_r), color=BLUE, lw=2,
            label=f"argmin KL(q‖p): mean {m_r:.2f}, sd {s_r:.2f}")
    ax.plot(t, norm.pdf(t, mean_p, np.sqrt(var_p)), color=ORANGE, lw=2, ls="--",
            label=f"argmin KL(p‖q): mean {mean_p:.2f}, sd {np.sqrt(var_p):.2f}")
    ax.set(xlabel="z", ylabel="density", title="Fitting one Gaussian to a two-mode target")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(0, 0.62)
    fig.tight_layout()
    save(fig, "kl-directions.png")


if __name__ == "__main__":
    logistic_posterior()
    importance()
    mh_step_sizes()
    gibbs_path()
    kl_directions()
