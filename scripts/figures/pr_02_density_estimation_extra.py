"""Extra figures for PR 02 (density estimation).

Run from the repository root:
    MPLCONFIGDIR=/tmp/cv-blog-mpl .venv/bin/python scripts/figures/pr_02_density_estimation_extra.py

Writes only to src/assets/pattern-recognition/pr-02-density-estimation/.
All data are synthetic or scikit-learn toy datasets (fixed seeds); nothing is downloaded.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammaln
from scipy.stats import norm
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier

OUT = Path(__file__).resolve().parents[2] / "src/assets/pattern-recognition/pr-02-density-estimation"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})
C0, C1, C2 = "#1f77b4", "#d62728", "#2ca02c"


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def true_pdf(x):
    return 0.6 * norm.pdf(x, -1.5, 0.4) + 0.4 * norm.pdf(x, 1.5, 0.4)


def sample(rng, n):
    means = np.where(rng.random(n) < 0.6, -1.5, 1.5)
    return means + rng.normal(0, 0.4, n)


def bayes_mean():
    """Posterior over a Gaussian mean as N grows (known variance)."""
    rng = np.random.default_rng(3)
    mu_true, sigma = 0.8, 1.0
    m0, s0 = 0.0, 0.5   # prior N(m0, s0^2)
    data = rng.normal(mu_true, sigma, 100)
    mu = np.linspace(-1.5, 2.0, 600)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(mu, norm.pdf(mu, m0, s0), color="gray", ls="--", label="prior (N = 0)")
    for n, col in [(1, "#9467bd"), (5, C0), (20, C2), (100, C1)]:
        x = data[:n]
        sn2 = 1 / (1 / s0 ** 2 + n / sigma ** 2)
        mn = sn2 * (m0 / s0 ** 2 + x.sum() / sigma ** 2)
        ax.plot(mu, norm.pdf(mu, mn, np.sqrt(sn2)), color=col,
                label=f"N = {n}: ML = {x.mean():.2f}, MAP = {mn:.2f}")
    ax.axvline(mu_true, color="black", lw=0.8)
    ax.set_xlabel(r"mean $\mu$")
    ax.set_ylabel("posterior density")
    ax.set_title("Bayesian estimate of a Gaussian mean (true 0.8)")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, "bayes-mean-posterior.png")


def histograms():
    rng = np.random.default_rng(4)
    x = sample(rng, 180)
    grid = np.linspace(-3.5, 3.5, 500)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    settings = [(0.1, 0.0, "width 0.1"), (0.6, 0.0, "width 0.6, origin 0"),
                (0.6, 0.3, "width 0.6, origin 0.3")]
    for ax, (w, origin, title) in zip(axes, settings):
        edges = np.arange(-3.5 + origin, 3.6 + origin, w)
        ax.hist(x, bins=edges, density=True, color="#9ecae1", edgecolor="white")
        ax.plot(grid, true_pdf(grid), "k--", lw=1)
        ax.set_title(title)
        ax.set_xlabel("x")
    axes[0].set_ylabel("density")
    fig.tight_layout()
    save(fig, "histogram-bins.png")


def parzen_vs_knn():
    rng = np.random.default_rng(4)
    x = np.sort(sample(rng, 180))
    grid = np.linspace(-3.5, 3.5, 700)
    n = len(x)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    # Parzen: box window vs Gaussian window, same width
    h = 0.3
    box = (np.abs(grid[:, None] - x[None]) <= h / 2).sum(1) / (n * h)
    gauss = norm.pdf((grid[:, None] - x[None]) / h).sum(1) / (n * h)
    axes[0].plot(grid, box, color=C0, label="box window, h = 0.3")
    axes[0].plot(grid, gauss, color=C1, label="Gaussian window, h = 0.3")
    axes[0].plot(grid, true_pdf(grid), "k--", lw=1, label="true density")
    axes[0].set_title("Parzen windows")
    # kNN density: k/(N * 2 r_k)
    dist = np.sort(np.abs(grid[:, None] - x[None]), axis=1)
    for k, col in [(5, C2), (30, "#9467bd")]:
        axes[1].plot(grid, k / (n * 2 * dist[:, k - 1]), color=col, label=f"k-NN, k = {k}")
    axes[1].plot(grid, true_pdf(grid), "k--", lw=1)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("k-nearest-neighbor density")
    for ax in axes:
        ax.set_xlabel("x")
        ax.legend(fontsize=8.5)
    axes[0].set_ylabel("density")
    fig.tight_layout()
    save(fig, "parzen-knn.png")


def knn_classifier():
    X, y = make_moons(400, noise=0.3, random_state=0)
    xx, yy = np.meshgrid(np.linspace(-2, 3, 300), np.linspace(-1.6, 2.1, 300))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for ax, k in zip(axes, [1, 15, 101]):
        model = KNeighborsClassifier(n_neighbors=k).fit(X, y)
        z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, z, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.45)
        ax.contour(xx, yy, z, levels=[0.5], colors="black", linewidths=1.2)
        ax.scatter(X[:, 0], X[:, 1], c=np.where(y, C1, C0), s=7)
        ax.set_title(f"k = {k}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    save(fig, "knn-classifier.png")


def curse():
    rng = np.random.default_rng(0)
    dims = [1, 2, 5, 10, 20, 50, 100, 500]
    ratios = []
    for d in dims:
        pts = rng.random((1000, d))
        q = rng.random((50, d))
        dist = np.sqrt(((q[:, None, :] - pts[None]) ** 2).sum(-1))
        ratios.append(np.mean(dist.min(1) / dist.max(1)))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
    axes[0].semilogx(dims, ratios, "o-", color=C0)
    axes[0].set_xlabel("dimension d")
    axes[0].set_ylabel("nearest / farthest distance")
    axes[0].set_title("Distances concentrate (1000 uniform points)")
    axes[0].set_ylim(0, 1)
    # edge length of a sub-cube that captures fraction r of a unit cube
    ds = np.arange(1, 101)
    for r, col in [(0.01, C2), (0.1, C1)]:
        axes[1].plot(ds, r ** (1 / ds), color=col, label=f"capture {int(r*100)}% of data")
    # volume of unit ball relative to enclosing cube [-1,1]^d
    ratio_ball = np.exp(d_ball := (ds / 2) * np.log(np.pi) - gammaln(ds / 2 + 1) - ds * np.log(2))
    axes[1].plot(ds, ratio_ball, color="#9467bd", label="ball volume / cube volume")
    axes[1].set_xlabel("dimension d")
    axes[1].set_ylabel("edge length or fraction")
    axes[1].set_title("Neighborhoods must grow")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    save(fig, "curse-dimensionality.png")
    return dict(zip(dims, np.round(ratios, 3)))


if __name__ == "__main__":
    bayes_mean()
    histograms()
    parzen_vs_knn()
    knn_classifier()
    print(curse())
