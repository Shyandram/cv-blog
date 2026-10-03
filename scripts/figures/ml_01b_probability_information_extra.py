"""Figures for ML 01B - Probability and Information Theory.

Run:  .venv/bin/python scripts/figures/ml_01b_probability_information_extra.py
Writes only to src/assets/ml/ml-01b-probability-information/. All data are
synthetic (NumPy random draws or closed-form densities); nothing is downloaded.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "ml" / "ml-01b-probability-information"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "axes.facecolor": "white"})
BLUE, ORANGE, GREEN, RED, GRAY, PURPLE = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#7f7f7f", "#9467bd"


def save(fig, name, max_px=1600):
    """Save at 150 dpi, lowering dpi only if the tight-cropped width would exceed max_px."""
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    width = plt.imread(path).shape[1]
    if width > max_px:
        fig.savefig(path, dpi=150 * (max_px - 10) / width, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"{name}: {path.stat().st_size / 1024:.0f} KB")


def gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def fig_distributions():
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))

    ax = axes[0, 0]
    ax.bar([0, 1], [0.3, 0.7], color=BLUE, width=0.5, label="Bernoulli($\\phi$=0.7)")
    ax.bar([3, 4, 5, 6], [0.1, 0.4, 0.3, 0.2], color=ORANGE, width=0.5, label="Categorical (4 classes)")
    ax.set_xticks([0, 1, 3, 4, 5, 6], ["0", "1", "c1", "c2", "c3", "c4"])
    ax.set_ylabel("probability mass P(x)")
    ax.set_title("Discrete: mass sums to 1")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    x = np.linspace(-4, 4, 600)
    for s, c in [(0.25, RED), (1.0, BLUE), (2.0, GREEN)]:
        ax.plot(x, gauss(x, 0, s), color=c, label=f"N(0, {s}$^2$)")
    ax.axhline(1.0, color=GRAY, ls=":", lw=1)
    ax.set_title("Gaussian density can exceed 1")
    ax.set_ylabel("density p(x)")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    x = np.linspace(-4, 4, 801)
    ax.plot(x, gauss(x, 0, 1), color=BLUE, label="Gaussian N(0,1)")
    ax.plot(x, 0.5 * np.exp(-np.abs(x)), color=ORANGE, label="Laplace(0, 1)")
    xe = x[x >= 0]
    ax.plot(xe, np.exp(-xe), color=GREEN, label="Exponential($\\lambda$=1)")
    ax.set_title("Sharper peaks, heavier tails")
    ax.set_xlabel("x")
    ax.set_ylabel("density p(x)")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    rng = np.random.default_rng(0)
    mu = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.8], [0.8, 1.0]])
    pts = rng.multivariate_normal(mu, cov, size=400)
    g = np.linspace(-3.5, 3.5, 200)
    X1, X2 = np.meshgrid(g, g)
    P = np.linalg.inv(cov)
    d = np.stack([X1, X2], -1)
    q = np.einsum("...i,ij,...j->...", d, P, d)
    dens = np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(np.linalg.det(cov)))
    ax.scatter(pts[:, 0], pts[:, 1], s=6, color=GRAY, alpha=0.5)
    ax.contour(X1, X2, dens, levels=5, colors=BLUE)
    evals, evecs = np.linalg.eigh(cov)
    for lam, v in zip(evals, evecs.T):
        ax.annotate("", xy=2 * np.sqrt(lam) * v, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=RED, lw=2))
    ax.set_aspect("equal")
    ax.set_title("2-D Gaussian, correlation 0.8")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    fig.tight_layout()
    save(fig, "distributions.png")


def fig_mixture():
    rng = np.random.default_rng(1)
    w = np.array([0.5, 0.3, 0.2])
    mus = np.array([-2.0, 0.5, 3.0])
    sig = np.array([0.6, 0.8, 0.5])
    n = 2000
    z = rng.choice(3, size=n, p=w)
    xs = rng.normal(mus[z], sig[z])
    x = np.linspace(-5, 5, 800)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.hist(xs, bins=60, density=True, color=GRAY, alpha=0.4, label="2,000 samples")
    total = np.zeros_like(x)
    for k, c in enumerate([BLUE, ORANGE, GREEN]):
        comp = w[k] * gauss(x, mus[k], sig[k])
        total += comp
        ax.plot(x, comp, color=c, ls="--", label=f"$\\pi_{k + 1}$ N(x; $\\mu_{k + 1}$, $\\sigma_{k + 1}^2$)")
    ax.plot(x, total, color="black", lw=2, label="mixture p(x)")
    ax.set_title("1-D Gaussian mixture")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)

    ax = axes[1]
    centers = np.array([[-2, -1], [2, 0], [0, 2.5]])
    covs = [np.array([[0.5, 0.3], [0.3, 0.4]]), np.array([[0.3, -0.2], [-0.2, 0.6]]), np.eye(2) * 0.25]
    z2 = rng.choice(3, size=600, p=w)
    for k, c in enumerate([BLUE, ORANGE, GREEN]):
        pts = rng.multivariate_normal(centers[k], covs[k], size=(z2 == k).sum())
        ax.scatter(pts[:, 0], pts[:, 1], s=7, color=c, alpha=0.7, label=f"latent z = {k + 1}")
    ax.set_aspect("equal")
    ax.set_title("2-D mixture, colored by hidden z")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "mixture.png")


def fig_base_rate():
    sens, fpr = 0.9, 0.05
    prev = np.logspace(-4, 0, 400)
    ppv = sens * prev / (sens * prev + fpr * (1 - prev))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.semilogx(prev, ppv, color=BLUE, lw=2)
    for p0, c, off in [(0.01, RED, (10, -28)), (0.10, GREEN, (-150, 12))]:
        v = sens * p0 / (sens * p0 + fpr * (1 - p0))
        ax.plot(p0, v, "o", color=c)
        ax.annotate(f"prevalence {p0:.0%}\nP(defect | +) = {v:.3f}", (p0, v),
                    textcoords="offset points", xytext=off, fontsize=9, color=c)
    ax.set_xlabel("prevalence P(defect) (log scale)")
    ax.set_ylabel("P(defect | positive)")
    ax.set_title("Same detector, different base rates")
    ax.grid(alpha=0.3)

    ax = axes[1]
    N = 10000
    d = N * 0.01
    tp, fn = d * sens, d * (1 - sens)
    fp, tn = (N - d) * fpr, (N - d) * (1 - fpr)
    ax.bar(["true positive", "false positive"], [tp, fp], color=[GREEN, RED])
    for i, v in enumerate([tp, fp]):
        ax.text(i, v + 8, f"{v:.0f}", ha="center")
    ax.set_title("Out of 10,000 items at 1% prevalence")
    ax.set_ylabel("number of positive alarms")
    ax.set_ylim(0, 600)
    fig.tight_layout()
    save(fig, "base-rate.png")
    print("TN, FN", tn, fn)


def fig_entropy_sigmoid():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    p = np.linspace(1e-6, 1 - 1e-6, 500)
    h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    ax.plot(p, h, color=BLUE, lw=2)
    ax.set_xlabel("$\\phi$ = P(x = 1)")
    ax.set_ylabel("H(x) in bits")
    ax.set_title("Entropy of a Bernoulli variable")
    ax.grid(alpha=0.3)

    ax = axes[1]
    x = np.linspace(-6, 6, 400)
    ax.plot(x, 1 / (1 + np.exp(-x)), color=BLUE, lw=2, label="sigmoid $\\sigma(x)$")
    ax.plot(x, np.logaddexp(0, x), color=ORANGE, lw=2, label="softplus $\\zeta(x)$")
    ax.plot(x, np.maximum(x, 0), color=GRAY, ls=":", label="max(0, x)")
    ax.set_ylim(-0.5, 6)
    ax.set_xlabel("x")
    ax.set_title("Logistic sigmoid and softplus")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "entropy-sigmoid.png")


def fig_kl_asymmetry():
    x = np.linspace(-8, 8, 4001)
    dx = x[1] - x[0]
    p = 0.5 * gauss(x, -2.5, 0.8) + 0.5 * gauss(x, 2.5, 0.8)
    # Forward KL(p||q) over Gaussians q is minimized by moment matching.
    m_f = (x * p).sum() * dx
    s_f = np.sqrt(((x - m_f) ** 2 * p).sum() * dx)

    def reverse_kl(theta):
        m, log_s = theta
        q = gauss(x, m, np.exp(log_s))
        mask = q > 1e-300
        return (q[mask] * (np.log(q[mask]) - np.log(p[mask] + 1e-300))).sum() * dx

    res = minimize(reverse_kl, x0=[1.0, 0.0], method="Nelder-Mead")
    m_r, s_r = res.x[0], np.exp(res.x[1])
    print(f"forward fit mu={m_f:.3f} sigma={s_f:.3f}; reverse fit mu={m_r:.3f} sigma={s_r:.3f}")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (m, s, title, c) in zip(axes, [
        (m_f, s_f, "minimize KL(p || q): mass-covering", BLUE),
        (m_r, s_r, "minimize KL(q || p): mode-seeking", ORANGE),
    ]):
        ax.plot(x, p, color="black", lw=2, label="target p (two modes)")
        ax.plot(x, gauss(x, m, s), color=c, lw=2, ls="--", label=f"best Gaussian q\n$\\mu$={m:.2f}, $\\sigma$={s:.2f}")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_xlim(-6, 6)
        ax.legend(fontsize=9, loc="upper left")
    axes[0].set_ylabel("density")
    fig.tight_layout()
    save(fig, "kl-asymmetry.png")


if __name__ == "__main__":
    fig_distributions()
    fig_mixture()
    fig_base_rate()
    fig_entropy_sigmoid()
    fig_kl_asymmetry()
